#!/bin/sh
## ------------- specify queue name ----------------
#BSUB -q c02516
## ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
## ------------- specify job name ----------------
#BSUB -J temp_prog_vae_inf
## ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
## ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=16GB]"
## ------------- specify wall-clock time ----------------
#BSUB -W 2:00
#BSUB -o OUTPUT_FILE_TEMPORAL_PROG_VAE_INF%J.out
#BSUB -e OUTPUT_FILE_TEMPORAL_PROG_VAE_INF%J.err

source /zhome/d1/3/223803/test/bin/activate
cd /zhome/d1/3/223803/dltok/conditional_diffusion || exit 1

python 32_inference_temporal_progressive_vae.py \
	--checkpoint /dtu/blackhole/1b/223803/runs/temporal_progressive_vae/best_model.pt \
	--data-dir /dtu/blackhole/1b/223803/bout_data \
	--probe-dir /dtu/blackhole/1b/223803/probe_data \
	--output /dtu/blackhole/1b/223803/results/temporal_progressive_vae_inference \
	--variables n te ti phi \
	--max-resolution 256 \
	--device cuda \
	--num-samples 100
