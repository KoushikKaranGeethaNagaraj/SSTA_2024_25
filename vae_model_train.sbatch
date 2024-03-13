#!/bin/bash
#SBATCH -Avae_train_model
#SBATCH --account=gts-skousik3
#SBATCH -N1 --gres=gpu:V100:2
#SBATCH -t15
#SBATCH -qinferno
#SBATCH -oReport-%vae_report_new2.out
#SBATCH --mail-type=BEGIN,END,FAIL #SBATCH --mail-user=knagaraj31@gatech.edu
srun python /storage/home/hcoda1/9/knagaraj31/p-skousik3-0/SSTA_2022/SSTA_ITSC_2022-main/vae.py
