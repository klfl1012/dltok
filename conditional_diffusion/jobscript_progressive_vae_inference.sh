#!/bin/sh
## ------------- specify queue name ----------------
#BSUB -q c02516
## ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
## ------------- specify job name ----------------
#BSUB -J progressive_vae_inference
## ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
## ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=16GB]"
## ------------- specify wall-clock time ----------------
#BSUB -W 2:00
#BSUB -o OUTPUT_FILE_PROGRESSIVE_VAE_INF%J.out
#BSUB -e OUTPUT_FILE_PROGRESSIVE_VAE_INF%J.err

source /zhome/d1/3/223803/test/bin/activate
cd /zhome/d1/3/223803/dltok/conditional_diffusion || exit 1

python 20_inference_progressive_vae.py \
	--checkpoint /dtu/blackhole/1b/223803/runs/progressive_vae/best_model.pt \
	--data-dir /dtu/blackhole/1b/223803/bout_data \
	--probe-dir /dtu/blackhole/1b/223803/probe_data \
	--output /dtu/blackhole/1b/223803/results/progressive_vae_inference \
	--variables n te ti phi \
	--max-resolution 256 \
	--device cuda \
	--num-timesteps 93
