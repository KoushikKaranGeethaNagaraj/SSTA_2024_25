import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
from itertools import zip_longest, chain
import re

logger = logging.getLogger(__name__)
class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.num_views = input_param.get('num_views', 2)
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']
        self.img_channel = input_param.get('img_channel', 3)
        self.n_epoch = input_param.get('n_epoch', 1)
        self.n_cl_step = input_param.get('n_cl_step', 1)
        self.cl_mode = input_param.get('cl_mode', 1)
        
    # This fnction returns the total number of sequences
    def total(self):
        return len(self.indices)
    
    #initialises the batch to 0 and shuffles the dataset if train and not if val/test
    def begin(self, do_shuffle=False, epoch=None):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
            pass
        if epoch:
            # self.current_position = int(self.total() * (epoch // self.n_cl_step) * self.n_cl_step / self.n_epoch)
            self.current_position = int(self.total() * epoch / self.n_epoch)
        else:
            self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    #moving to next batch
    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    #checks if all the batches have been completed
    def no_batch_left(self, epoch=None):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        # elif epoch is not None and self.current_position != 0 and \
        #         self.current_position + self.minibatch_size >= \
        #         self.total() * min(((epoch // self.n_cl_step) + 1) * self.n_cl_step / self.n_epoch, 1.):
        #     return True
        elif epoch is not None and self.current_position != 0 and \
                self.current_position + self.minibatch_size >= \
                self.total() * min((epoch + self.n_cl_step) / self.n_epoch, 1.):
            return True
        else:
            return False

    #gets the specific batch for training/validation
    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width,
             self.img_channel * self.num_views)).astype(self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            # print(input_batch[i, :self.current_input_length, :, :, :].shape, data_slice.shape)
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        print("-"*20)
        print("Iterator Name: " + self.name)
        print("    current_position: " + str(self.current_position))
        print("    Minibatch Size: " + str(self.minibatch_size))
        print("    total Size: " + str(self.total()))
        print("    current_input_length: " + str(self.current_input_length))
        print("    Input Data Type: " + str(self.input_data_type))
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))
        print("-"*20)


# This Class is used to load/combine the dataset and get the details of sequences for the VAE/SSTA
class DataProcess:
    def __init__(self, input_param):
        self.input_param = input_param
        self.paths = input_param['paths']
        self.image_width = input_param['image_width'] 
        self.seq_len = input_param['seq_length']
        self.num_views = input_param.get('num_views', 2)
        self.img_channel = input_param.get('img_channel', 3)
        self.baseline = input_param.get('baseline', 'SSTA_view_view')
        self.datautility=DatasetUtility(self.num_views)

    def load_data(self, path,mode="None"):
        # This function takes in (N,Height,Width,ImageChannels) and converts to (N/ssta_views,Height,Width,(ImageChannels*ssta_views))
        frames_np = []
        frames_np=self.datautility.dataset_SSTA_alternator(path)
        #4-1 combines images  if total images were 4000 with 4 views it becomes 1000 images with combination of 4 views(rare usage)
        if self.baseline == 'SSTA_views_1':
            frames_np_4to1 = []  
            for view_idx_start in range(0, len(frames_np), 4):
                temporal1_img = np.concatenate((frames_np[view_idx_start + 0], frames_np[view_idx_start + 1]), axis=1)
                temporal2_img = np.concatenate((frames_np[view_idx_start + 2], frames_np[view_idx_start + 3]), axis=1)
                frame = np.concatenate([temporal1_img, temporal2_img], axis=0)
                frame = cv2.resize(frame, (self.image_width, self.image_width)) / 255.
                if view_idx_start < 100 and False:
                    os.makedirs('tmp', exist_ok=True)
                    # print('frame.shape: ', frame.shape)  # frame.shape:  (128, 128, 3)
                    frame = frame[:, :, ::-1]
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join('tmp', '{}.png'.format(view_idx_start)), frame * 255.)
                frames_np_4to1.append(frame)
            data = np.asarray(frames_np_4to1)
        else:
            #this segment is used the most as the views are appended into data to return in teh combined format (N/ssta_views,Height,Width,(ImageChannels*ssta_views)
            frames_np = np.asarray(frames_np)   
            data = np.zeros((frames_np.shape[0], self.image_width, self.image_width, self.img_channel))
            for i in range(len(frames_np)):
                temp = np.float32(frames_np[i])
                data[i, :, :, :] = cv2.resize(temp, (self.image_width, self.image_width)) / 255
            new_data = np.zeros((frames_np.shape[0] // self.num_views, self.image_width, self.image_width, self.img_channel * self.num_views))
            for i in range(self.num_views):
                new_data[:, :, :, self.img_channel*i:self.img_channel*(i+1)] = data[i::self.num_views][:frames_np.shape[0] // self.num_views]
            data = new_data
        # is it a begin index of sequence
        indices = []
        index = len(data) - 1
        while index >= self.seq_len - 1:
            indices.append(index - self.seq_len + 1)
            index -= 1
        self.processed_dataset_info(path,frames_np,data,indices,mode,self.num_views)
        # indices are the total sequences in the combined dataset that is calculated by Total_images -(num_past+num_step), that is each batch
        return data, indices
    
    def processed_dataset_info(self,path,frames_np,data,indices,mode,ssta_no):
        print("-"*20)
        print('Loaded data from ' + str(path))
        print("Mode: " ,mode,"SSTA_num:",ssta_no)
        print("Dataset Loaded and Alternated: " ,frames_np.shape)
        print("there are " + str(data.shape[0]) + " pictures ")
        print("Combined Dataset for VAE/SSTA " ,data.shape)
        print("there are " + str(len(indices)) + " sequences")
        print("-"*20)


    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths[0],mode="train")
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths[0],mode="test")
        return InputHandle(test_data, test_indices, self.input_param)


#utility class for creating the alternate dataset
class DatasetUtility:
    def __init__(self,total_views):
        self.total_views=total_views
    #utility class for creating the alternate dataset
        
    # This function is used as a key to sort numerically
    def numerical_sort(self,filename):
        """Extracts the numerical part of a filename for sorting."""
        number_extractor = re.compile(r'\d+')
        match = number_extractor.search(filename)
        if match:
            return int(match.group())
        return 0

    #This function takes the dataset, example: ssta_num=4, camera_0,camera_1,camera_2,camera_3, and then converts[0,0,...][1,1,1..][2,2,2..][3,3,3..]to [0,1,2,3,0,1,2,3...]
    def dataset_SSTA_alternator(self,dataset_path):
        # arguments:
        #     dataset_paths-contains a  path to train/val/test dataset 
        #     total_views-SSTA views  
        # returns:Numpy array of the path of dataset passed in as shape (N,Height,Width,ImageChannels)--- N is total number of images from all camera/folders
        dataset_files = [[] for _ in range(self.total_views)]
        for num,camera_view in enumerate(os.listdir(dataset_path)):
            full_path = os.path.join(dataset_path, camera_view)
            dataset_files[num].extend([os.path.join(full_path, file) for file in sorted(os.listdir(full_path),key=self.numerical_sort)])
        ## Combine files alternately
        combined_files_dataset= []
        #alternate the images based on the camera views
        combined_files_dataset = list(chain.from_iterable(filter(None, x) for x in zip_longest(*dataset_files)))
        combined_files_dataset_np = np.stack([np.array(Image.open(path)) for path in combined_files_dataset])
        return combined_files_dataset_np
  





