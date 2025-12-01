#!/bin/sh
## ------------- specify queue name ----------------
#BSUB -q c02516
## ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
## ------------- specify job name ----------------
#BSUB -J probe_recon
## ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
## ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=32GB]"
## ------------- specify wall-clock time (max allowed is 12:00)----------------
#BSUB -W 12:00
#BSUB -o OUTPUT_FILE_PROBE_RECON%J.out
#BSUB -e OUTPUT_FILE_PROBE_RECON%J.err

source /zhome/d1/3/223803/test/bin/activate
cd /zhome/d1/3/223803/dltok/conditional_diffusion || exit 1

python 37_train_probe_reconstruction.py \
	--vae-checkpoint /dtu/blackhole/1b/223803/runs/vae_kl_low/best_model.pt \
	--sim-data-dir /dtu/blackhole/1b/223803/tcv_probe_data/simulation \
	--probe-data-dir /dtu/blackhole/1b/223803/tcv_probe_data/probes \
	--variables n phi \
	--seq-len 10 \
	--num-probes 64 \
	--epochs 200 \
	--batch-size 16 \
	--lr 1e-3 \
	--kl-weight 1e-4 \
	--output /dtu/blackhole/1b/223803/runs/probe_reconstruction_kl_low \
	--use-amp \
	--patience 20
