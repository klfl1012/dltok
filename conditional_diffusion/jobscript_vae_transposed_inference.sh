#!/bin/sh
## ------------- specify queue name ----------------
#BSUB -q c02516
## ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
## ------------- specify job name ----------------
#BSUB -J vae_transposed_inference
## ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
## ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"
## ------------- specify wall-clock time (max allowed is 12:00)----------------
#BSUB -W 12:00
#BSUB -o OUTPUT_FILE_VAE_TRANSPOSED_INFERENCE%J.out
#BSUB -e OUTPUT_FILE_VAE_TRANSPOSED_INFERENCE%J.err

source /zhome/d1/3/223803/test/bin/activate
cd /zhome/d1/3/223803/dltok/conditional_diffusion || exit 1

python 30_inference_vae_transposed.py \
	--checkpoint /dtu/blackhole/1b/223803/runs/vae_transposed_heavy/best_model.pt \
	--data-dir /dtu/blackhole/1b/223803/bout_data \
	--probe-dir /dtu/blackhole/1b/223803/probe_data \
	--output /dtu/blackhole/1b/223803/results/vae_transposed_inference \
	--variables n te ti phi \
	--resize 256 \
	--noise-std 0.1 \
	--device cuda \
	--denoise-iterations 1 2 5 10
