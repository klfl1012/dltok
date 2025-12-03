#!/bin/sh
## ------------- specify queue name ----------------
#BSUB -q c02516
## ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
## ------------- specify job name ----------------
#BSUB -J vae_training
## ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
## ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=32GB]"
## ------------- specify wall-clock time (max allowed is 12:00)----------------
#BSUB -W 12:00
#BSUB -o OUTPUT_FILE_TRAINING%J.out
#BSUB -e OUTPUT_FILE_TRAINING%J.err

source /zhome/d1/3/223803/test/bin/activate
cd /zhome/d1/3/223803/dltok/conditional_diffusion || exit 1

python 34_train_multiscale_vae.py \
	--data-dir /dtu/blackhole/1b/223803/tcv_data \
	--variables n phi \
	--epochs 200 \
	--batch-size 8 \
	--lr 1e-3 \
	--kl-weight 1e-4 \
	--output /dtu/blackhole/1b/223803/runs/multiscale_vae_elbow \
	--use-amp \
	--patience 20