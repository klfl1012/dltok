#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J dl_job_32gb_epochs:100_spatial_resolution_256_timesteps2000_3000_4000 
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

source /dtu/blackhole/1b/191611/DL/DL/bin/activate
cd /dtu/blackhole/1b/191611/Data

# python main.py --mode train     --max_epochs 5     --batch_size 8     --seq_len 4  --spatial_resolution 128     --hidden_channels 16     --n_modes "4,4,4"  --patience 3     --num_predictions_to_log 1 --use_wandb
# python main.py --mode train  --model "diffusion"  --dataset_name "DiffusionDataset" --max_epochs 100     --batch_size 4 --spatial_resolution 512  --num_predictions_to_log 1 --diffusion_dim 64 --diffusion_dim_mults "1,2,4,8" --diffusion_flash_attn --diffusion_auto_normalize --use_wandb
python main.py --mode train  --diffusion_timesteps 2000 --diffusion_sampling_timesteps 2000 --model "diffusion"  --dataset_name "DiffusionDataset" --max_epochs 100     --batch_size 4 --spatial_resolution 256  --num_predictions_to_log 1 --diffusion_dim 64 --diffusion_dim_mults "1,2,4,8" --diffusion_flash_attn --use_wandb
python main.py --mode train  --diffusion_timesteps 3000 --diffusion_sampling_timesteps 3000 --model "diffusion"  --dataset_name "DiffusionDataset" --max_epochs 100     --batch_size 4 --spatial_resolution 256  --num_predictions_to_log 1 --diffusion_dim 64 --diffusion_dim_mults "1,2,4,8" --diffusion_flash_attn --use_wandb
python main.py --mode train  --diffusion_timesteps 4000 --diffusion_sampling_timesteps 4000 --model "diffusion"  --dataset_name "DiffusionDataset" --max_epochs 100     --batch_size 4 --spatial_resolution 256  --num_predictions_to_log 1 --diffusion_dim 64 --diffusion_dim_mults "1,2,4,8" --diffusion_flash_attn --use_wandb