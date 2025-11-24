#!/bin/sh
## ------------- specify queue name ----------------
#BSUB -q c02516
## ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
## ------------- specify job name ----------------
#BSUB -J temporal_prog_vae
## ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
## ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"
## ------------- specify wall-clock time (max allowed is 12:00)----------------
#BSUB -W 12:00
#BSUB -o OUTPUT_FILE_TEMPORAL_PROG_VAE%J.out
#BSUB -e OUTPUT_FILE_TEMPORAL_PROG_VAE%J.err

source /zhome/d1/3/223803/test/bin/activate
cd /zhome/d1/3/223803/dltok/conditional_diffusion || exit 1

python 31_train_temporal_progressive_vae.py \
	--epochs 200 \
	--batch-size 2 \
	--lr 1e-3 \
	--device cuda \
	--output /dtu/blackhole/1b/223803/runs/temporal_progressive_vae \
	--probe-dir /dtu/blackhole/1b/223803/probe_data \
	--data-dir /dtu/blackhole/1b/223803/bout_data \
	--sequence-length 10 \
	--spatial-size 256 \
	--target-sizes 64 128 256 \
	--hidden-dims 64 128 \
	--scale-weights "64:0.5,128:1.0,256:2.0" \
	--interpolate-probes \
	--use-amp \
	--save-every 20
