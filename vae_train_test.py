# coding: utf-8
# https://linuxtut.com/en/fe2d3308b3ba56a80c7a/

import numpy as np
import time
from matplotlib import pyplot as plt
import os, cv2
import sys

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from VAE_model import VanillaVAE
import argparse
import random
from dataset_processing.batch_accessing import data_provider

import lpips
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
loss_fn_alex = lpips.LPIPS(net='alex')

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def run_steps(x_batch, model_0, model_1, args=None):
    '''
    x_batch: [bs, T, 2*2],
    pred_batch: [bs, T, 1*2],
    x_batch: [bs, T, H, W, C], C=3
    pred_batch: [bs, T, H, W, C],
    '''
    num_hidden = [int(x) for x in args.num_hidden.split(',')]
    batch = x_batch.shape[0]
    height = x_batch.shape[2]
    width = x_batch.shape[3]

    h_t_0 = []
    c_t_0 = []
    h_t_1 = []
    c_t_1 = []
    delta_c_list_0 = []
    delta_m_list_0 = []
    delta_c_list_1 = []
    delta_m_list_1 = []

    decouple_loss = []

    for i in range(len(num_hidden)):
        zeros = None  # torch.zeros([batch, num_hidden[i], height, width]).to(args.device)
        h_t_0.append(zeros)
        c_t_0.append(zeros)
        h_t_1.append(zeros)
        c_t_1.append(zeros)
        delta_c_list_0.append(zeros)
        delta_m_list_0.append(zeros)
        delta_c_list_1.append(zeros)
        delta_m_list_1.append(zeros)

    memory_0 = torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    memory_1 = torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    if args.num_views in [4, ]:
        x_0_t, x_1_t, x_2_t, x_3_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)
    else:
        x_0_t, x_1_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)
    pred_batch_0_list = []
    pred_batch_1_list = []
    message_0_list = []
    message_1_list = []
    mu_0_list = []
    log_var_0_list = []
    mu_1_list = []
    log_var_1_list = []
    if args.num_views in [4, ]:
        pred_batch_2_list = []
        pred_batch_3_list = []

    message_0 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
    message_1 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)

    for t in range(args.num_step):
        x_0_t_pred, message_0, mu_0, log_var_0 = \
            model_0(x_0_t[:, t:t + args.num_past], message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0)
        x_1_t_pred, message_1, mu_1, log_var_1 = \
            model_0(x_1_t[:, t:t + args.num_past], message_1, message_0, memory_1, h_t_1, c_t_1, delta_c_list_1, delta_m_list_1)
        if args.num_views in [4, ]:
            x_2_t_pred, message_0, mu_0, log_var_0 = \
                model_0(x_2_t[:, t:t + args.num_past], message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0,
                        delta_m_list_0)
            x_3_t_pred, message_1, mu_1, log_var_1 = \
                model_0(x_3_t[:, t:t + args.num_past], message_1, message_0, memory_1, h_t_1, c_t_1, delta_c_list_1,
                        delta_m_list_1)

        pred_batch_0_list.append(x_0_t_pred)
        pred_batch_1_list.append(x_1_t_pred)
        message_0_list.append(message_0)
        message_1_list.append(message_1)
        mu_0_list.append(mu_0)
        log_var_0_list.append(log_var_0)
        mu_1_list.append(mu_1)
        log_var_1_list.append(log_var_1)
        if args.num_views in [4, ]:
            pred_batch_2_list.append(x_2_t_pred)
            pred_batch_3_list.append(x_3_t_pred)

    pred_batch_0 = torch.cat(pred_batch_0_list, 1)
    pred_batch_1 = torch.cat(pred_batch_1_list, 1)
    if args.num_views in [4, ]:
        pred_batch_2 = torch.cat(pred_batch_2_list, 1)
        pred_batch_3 = torch.cat(pred_batch_3_list, 1)
        pred_batch = torch.cat([pred_batch_0, pred_batch_1, pred_batch_2, pred_batch_3], -1)
    else:
        pred_batch = torch.cat([pred_batch_0, pred_batch_1], -1)
    message_batch_0 = torch.cat(message_0_list, 1)
    message_batch_1 = torch.cat(message_1_list, 1)
    if args.num_views in [4, ]:
        # psuedo
        message_batch = torch.cat([message_batch_0, message_batch_1, message_batch_0, message_batch_1], -1)
    else:
        message_batch = torch.cat([message_batch_0, message_batch_1], -1)

    mu_batch_0 = torch.cat(mu_0_list, 1)
    mu_batch_1 = torch.cat(mu_1_list, 1)
    if args.num_views in [4, ]:
        mu_batch = torch.cat([mu_batch_0, mu_batch_1, mu_batch_0, mu_batch_1], -1)
    else:
        mu_batch = torch.cat([mu_batch_0, mu_batch_1], -1)
    log_var_batch_0 = torch.cat(log_var_0_list, 1)
    log_var_batch_1 = torch.cat(log_var_1_list, 1)
    if args.num_views in [4, ]:
        log_var_batch = torch.cat([log_var_batch_0, log_var_batch_1, log_var_batch_0, log_var_batch_1], -1)
    else:
        log_var_batch = torch.cat([log_var_batch_0, log_var_batch_1], -1)

    return pred_batch, message_batch, mu_batch, log_var_batch

def eval_run_steps(x_batch, model_0,args=None):
    '''
    x_batch: [bs, T, 2*2],
    pred_batch: [bs, T, 1*2],
    x_batch: [bs, T, H, W, C], C=3
    pred_batch: [bs, T, H, W, C],
    '''
    num_hidden = [int(x) for x in args.num_hidden.split(',')]
    batch = x_batch.shape[0]
    height = x_batch.shape[2]
    width = x_batch.shape[3]

    h_t_0 = []
    c_t_0 = []
    h_t_1 = []
    c_t_1 = []
    delta_c_list_0 = []
    delta_m_list_0 = []
    delta_c_list_1 = []
    delta_m_list_1 = []

    for i in range(len(num_hidden)):
        zeros = None  # torch.zeros([batch, num_hidden[i], height, width]).to(args.device)
        h_t_0.append(zeros)
        c_t_0.append(zeros)
        h_t_1.append(zeros)
        c_t_1.append(zeros)
        delta_c_list_0.append(zeros)
        delta_m_list_0.append(zeros)
        delta_c_list_1.append(zeros)
        delta_m_list_1.append(zeros)

    memory_0 = torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    memory_1 = torch.zeros([batch, num_hidden[0], height, width]).to(args.device)

    x_0_t, x_1_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)

    pred_batch_0_list = []
    pred_batch_1_list = []
    message_0_list = []
    message_1_list = []
    mu_0_list = []
    log_var_0_list = []
    mu_1_list = []
    log_var_1_list = []


    message_0 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
    message_1 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
    
    
    for t in range(x_batch.shape[1]):
        x_0_t_pred, message_0, mu_0, log_var_0 = \
            model_0(x_0_t[:, t:t+1,:,:,:], message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0)
        x_1_t_pred, message_1, mu_1, log_var_1 = \
            model_0(x_1_t[:, t:t+1,:,:,:], message_1, message_0, memory_1, h_t_1, c_t_1, delta_c_list_1, delta_m_list_1)
   

        pred_batch_0_list.append(x_0_t_pred)
        pred_batch_1_list.append(x_1_t_pred)
        message_0_list.append(message_0)
        message_1_list.append(message_1)
        mu_0_list.append(mu_0)
        log_var_0_list.append(log_var_0)
        mu_1_list.append(mu_1)
        log_var_1_list.append(log_var_1)
    

    pred_batch_0 = torch.cat(pred_batch_0_list, 1)
    pred_batch_1 = torch.cat(pred_batch_1_list, 1)
    pred_batch = torch.cat([pred_batch_0, pred_batch_1], -1)
    
    message_batch_0 = torch.cat(message_0_list, 1)
    message_batch_1 = torch.cat(message_1_list, 1)
    message_batch = torch.cat([message_batch_0, message_batch_1], -1)

    mu_batch_0 = torch.cat(mu_0_list, 1)
    mu_batch_1 = torch.cat(mu_1_list, 1)
    mu_batch = torch.cat([mu_batch_0, mu_batch_1], -1)

    log_var_batch_0 = torch.cat(log_var_0_list, 1)
    log_var_batch_1 = torch.cat(log_var_1_list, 1)
    log_var_batch = torch.cat([log_var_batch_0, log_var_batch_1], -1)

    return pred_batch, message_batch, mu_batch, log_var_batch


def training(bs, n_epoch, act,args):
    #generate data/batch loaders for training and test dataset
    train_input_handle, test_input_handle = data_provider(
        args.data_name, args.train_data_paths, args.valid_data_paths, args.bs, args.img_width,
        seq_length=args.num_step + args.num_past, is_training=True, num_views=args.num_views, img_channel=args.img_channel,
        baseline=args.baseline, args=args)

    #saving path directory
    save_dir_path = args.vae_save_dir
    print("savefilepath",save_dir_path)
    if os.path.exists(save_dir_path) == False:
        os.makedirs(save_dir_path, exist_ok=True)

    #basing on channels vae
    input_dim = 3
    #spliting the hidden units string to list of integers
    h_units = [int(x) for x in args.num_hidden.split(',')]
    
    if args.mode == 'eval' and args.ckpt_dir is not None:
        model_0_path = os.path.join(args.ckpt_dir, "vae.pt")
        model_0 = torch.load(model_0_path)
        print('Loaded model_0 from {}'.format(model_0_path))
    elif args.mode == 'continue_train' and args.ckpt_dir is not None:
        model_0_path = os.path.join(args.ckpt_dir, "vae.pt")
        model_0 = torch.load(model_0_path)
        print('Loaded model_0 from {}'.format(model_0_path))
    else:
        model_0 = VanillaVAE(input_dim, h_units, act, args)
        print('Created VAE model_0')
    model_0 = model_0.to(args.device)


    #optimiser and loss function
    optimizer = optim.Adam(list(model_0.parameters()),args.adam_lr)
    MSE = nn.MSELoss()
    train_loss ,test_loss ,train_recon_loss,train_kl_loss ,test_recon_loss ,test_kl_loss=[],[],[],[],[],[]
    best_eval_loss = np.inf
    train_input_handle.print_stat()
    for epoch in range(1, n_epoch + 1):
        if args.mode!="eval":
            print('Training ...')
            print("--------------",epoch,"------------------")
            model_0.train()
            sum_loss ,sum_recon_loss ,sum_kl_loss = 0,0,0
            train_input_handle.begin(do_shuffle=True)
            N=0
            while (train_input_handle.no_batch_left() == False ):
                ims = train_input_handle.get_batch()
                # print("input",ims.shape)
                train_input_handle.next()
                x_batch = ims[:, :]
                gt_batch = ims[:, args.num_past:]
                x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
                gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))
                optimizer.zero_grad()
                pred_batch, message_batch, mu_batch, log_var_batch = run_steps(x_batch, model_0, None,
                                                    args=args)
                #loss function 
                reconstruction_loss = MSE(pred_batch, gt_batch)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var_batch - mu_batch ** 2 - log_var_batch.exp(),
                                                    dim=list(range(1, len(mu_batch.shape)))), dim=0)

                loss = reconstruction_loss + args.kld_weight * kld_loss
                #Gradient Descent
                loss.backward()
                optimizer.step()
                sum_loss += loss.data * bs
                sum_recon_loss += reconstruction_loss.data * bs
                sum_kl_loss += kld_loss.data * bs
                N+=bs

            ave_loss = sum_loss / N 
            ave_recon_loss = sum_recon_loss / N 
            ave_kl_loss = sum_kl_loss / N

            ## If not using cuda:
            if args.device=="cpu:0":
                train_loss.append(ave_loss.cpu())
                train_recon_loss.append(ave_recon_loss.cpu())
                train_kl_loss.append(ave_kl_loss.cpu())

            train_loss.append(ave_loss)
            train_recon_loss.append(ave_recon_loss)
            train_kl_loss.append(ave_kl_loss)
            train_stats = {'average_loss': ave_loss, 'average_reconstruction_loss': ave_recon_loss, 'average_kld_loss': ave_kl_loss}
            print("Training Loss at epoch",epoch,":",train_stats)
            print('Evaluating ...')
            with torch.no_grad():
                test_sum_loss ,test_sum_recon_loss ,test_sum_kl_loss = [],[],[]
                test_input_handle.begin(do_shuffle=False)

                while (test_input_handle.no_batch_left() == False and args.mode != 'eval'):
                    ims = test_input_handle.get_batch()
                    test_input_handle.next()
                    x_batch = ims[:, :]
                    gt_batch = ims[:, args.num_past:]
                    x_test_torch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
                    t_test_torch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))

                    y_test_torch, message_torch, mu_batch, log_var_batch = run_steps(x_test_torch, model_0, None,args=args)
                    reconstruction_loss = MSE(y_test_torch, t_test_torch[:, :y_test_torch.shape[1]])
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var_batch - mu_batch ** 2 - log_var_batch.exp(),
                                                        dim=list(range(1, len(mu_batch.shape)))), dim=0)

                    loss = reconstruction_loss + args.kld_weight * kld_loss
                    test_sum_loss.append(loss.detach())
                    test_sum_recon_loss.append(reconstruction_loss.detach())
                    test_sum_kl_loss.append(kld_loss.detach())
                    
                avg_te_sum_loss = torch.mean(torch.tensor(test_sum_loss))
                avg_te_recon_loss = torch.mean(torch.tensor(test_sum_recon_loss))
                avg_te_kl_loss = torch.mean(torch.tensor(test_sum_kl_loss))
                test_loss.append(avg_te_sum_loss.data)
                test_recon_loss.append(avg_te_recon_loss.data)
                test_kl_loss.append(avg_te_kl_loss.data)
                test_stats = {'average_eval_loss': avg_te_sum_loss, 'average_eval_recon_loss': avg_te_recon_loss, 'average_val_kld_loss': avg_te_kl_loss}
                print("Evaluating Loss at epoch",epoch,":",test_stats)

            if avg_te_sum_loss < best_eval_loss and epoch>1 :
                print('avg_te_sum_loss: {}, best_eval_loss: {}'.format(avg_te_sum_loss, best_eval_loss))
                best_eval_loss = avg_te_sum_loss
                print(save_dir_path,"XXXXXXXXXXXXXXXXSAVING BEST MODEL VAEXXXXXXXXXXXXXXXXXXXXXXXXXX")
                train_root_res_path=os.path.join(save_dir_path, "best_model_vae")
                train_root_res_path=os.path.join(train_root_res_path, str(epoch))
                os.makedirs(train_root_res_path, exist_ok=True)
                torch.save(model_0, os.path.join(train_root_res_path, "vae.pt"))
            
            if epoch%2==0:#10
                print("XXXXXXXXXXXXXXXX VAE TRAIN Model XXXXXXXXXXXXXXXXXXXXXXXXXX")
                backup_train_path=os.path.join(save_dir_path, "backup_train_vae")
                backup_train_path=os.path.join(backup_train_path,str(epoch))
                os.makedirs(backup_train_path, exist_ok=True)
                torch.save(model_0, os.path.join(backup_train_path, "vae.pt")) 

        #generate and save vae images    
        if (args.mode=="eval" and epoch==n_epoch) or (args.mode!="eval" and epoch%1==0):
            batch_id = 0
            res_path = os.path.join(save_dir_path, "vae_evaluation_images")
            res_path = os.path.join(res_path, args.mode)
            res_path = os.path.join(res_path, str(epoch))
            os.makedirs(res_path, exist_ok=True)
        
            test_input_handle.begin(do_shuffle=False)
            test_images_count=0
            while (test_input_handle.no_batch_left() == False):
                batch_id = batch_id + 1
                ims = test_input_handle.get_batch()
                # print("imsshape",ims.shape)
                test_input_handle.next()
                temp_x_test = ims[:, :]
                t_test = ims[:, :]
                # print("test",x_test.shape)
                with torch.no_grad():
                    y_test, message, _, _ = eval_run_steps(torch.from_numpy(temp_x_test.astype(np.float32)).to(args.device),
                                                model_0,args=args)
                y_test = y_test.detach().cpu().numpy()
                message = message.detach().cpu().numpy()

                path = os.path.join(res_path,str(batch_id))
                os.makedirs(path,exist_ok=True)
                for view_idx in range(args.num_views):
                    for i in range(y_test.shape[1]):
                        name = 'gt_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        img_gt = np.uint8(t_test[0, i, :, :, (view_idx * args.img_channel):((view_idx + 1) * args.img_channel)] * 255)
                        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(file_name, img_gt)

                    for i in range(y_test.shape[1]):
                        name = 'pd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        img_pd = y_test[0, i, :, :,
                                (view_idx * args.img_channel):((view_idx + 1) * args.img_channel)]
                        img_pd = np.uint8(img_pd * 255)
                        img_pd = cv2.cvtColor(img_pd, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(file_name, img_pd)

                    for i in range(y_test.shape[1]):
                        name = 'msg_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        img_pd = message[0, i, :, :,
                                (view_idx * 1) * args.vae_latent_dim:((view_idx + 1) * args.vae_latent_dim)]
                        img_pd = np.uint8(img_pd * 255)
                        # print("messageshape",img_pd.shape)
                        cv2.imwrite(file_name, img_pd)
                test_images_count+=test_images_count+y_test.shape[1]
                if test_images_count>=args.display_test_images_count:
                    break



if __name__ == "__main__":

    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--act', type=str, default="relu", help='relu')
    parser.add_argument('--mode', type=str, default="train", help='train/continue_train / eval')
    parser.add_argument('--eval_mode', type=str, default='multi_step_eval', help='multi_step_eval / single_step_eval')
    parser.add_argument('--eval_num_step', type=int, default=15)
    parser.add_argument('--log_per_epoch', type=int, default=15)
    parser.add_argument('--num_step', type=int, default=1)
    parser.add_argument('--num_past', type=int, default=1)
    parser.add_argument('--num_cl_step', type=int, default=100)
    parser.add_argument('--n_epoch', type=int, default=100, help='200')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--Nte', type=int, default=20, help='200')
    parser.add_argument('--kld_weight', type=float, default=0.0000, help='0.00005')

    parser.add_argument('--data_name', type=str, default='model_vae')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 ; cpu:0')
    #change this to train
    parser.add_argument('--train_data_paths', type=str, default=r"Pneuma_dataset/pneuma_T2NOD_20181029_0_1_D2_0900_0930/train",help='r"DATASET\train" or r"DATASET\val"')
    parser.add_argument('--valid_data_paths', type=str, default=r"Pneuma_dataset/pneuma_T2NOD_20181029_0_1_D2_0900_0930/test",help='r"DATASET\val" or r"DATASET\test"')
    parser.add_argument('--display_test_images_count',type=int, default=50) 
    # RGB dataset
    parser.add_argument('--img_width', type=int, default=256, help='img width')
    parser.add_argument('--num_views', type=int, default=2, help='num views')
    parser.add_argument('--img_channel', type=int, default=3, help='img channel')

    parser.add_argument('--baseline', type=str, default='SSTA_view_view',
                        help="SSTA_view_view,SSTA_views_1")# This parameter is used to prepare the dataset,alternate or making 4 view dataset to 1 view dataset(SSTA_views_1)
    parser.add_argument('--vae_save_dir', type=str, default= r"vae_trained/vae_sim_model_32_16_8_latent000000rand")
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--num_hidden', type=str, default='32,16,8', help='64,64,64,64')
    parser.add_argument('--vae_latent_dim', type=int, default=4)
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--adam_lr',type=float, default=0.0005)                        
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--version', type=str, default='predrnn', help='version')
    parser.add_argument('--cl_mode', type=str, default='sliding_window', help='full_history, sliding_window')
    #TO load the vae and test
    parser.add_argument('--ckpt_dir', type=str,default="vae_sim_model_32_16_8_latent00.0000rand/best_model_vae/100", help='checkpoint dir: dir/vae.pt')

    args = parser.parse_args()
    args.vae_save_dir = os.path.join(args.vae_save_dir)

    training(args.bs, args.n_epoch, args.act, args)
