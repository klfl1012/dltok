#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J vae_big_model
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 12:00
#BSUB -o OUTPUT_FILE_BIG_MODEL%J.out
#BSUB -e OUTPUT_FILE_BIG_MODEL%J.err

source /zhome/d1/3/223803/test/bin/activate
cd /zhome/d1/3/223803/dltok/conditional_diffusion || exit 1

python 34_train_multiscale_vae.py \
	--data-dir /dtu/blackhole/1b/223803/tcv_data \
	--variables n phi \
	--epochs 200 \
	--batch-size 6 \
	--lr 1e-3 \
	--kl-weight 1e-4 \
	--latent-dim 256 \
	--base-channels 64 \
	--loss-type elbow \
	--output /dtu/blackhole/1b/223803/runs/vae_big_model \
	--use-amp \
	--patience 20
