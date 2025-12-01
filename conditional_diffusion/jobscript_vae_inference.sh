#!/bin/sh
## ------------- specify queue name ----------------
#BSUB -q c02516
## ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
## ------------- specify job name ----------------
#BSUB -J vae_inference
## ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
## ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=16GB]"
## ------------- specify wall-clock time (max allowed is 12:00)----------------
#BSUB -W 1:00
#BSUB -o OUTPUT_FILE_VAE_INFERENCE%J.out
#BSUB -e OUTPUT_FILE_VAE_INFERENCE%J.err

source /zhome/d1/3/223803/test/bin/activate
cd /zhome/d1/3/223803/dltok/conditional_diffusion || exit 1

python 35_inference_multiscale_vae.py \
    --model-path /dtu/blackhole/1b/223803/runs/multiscale_vae_elbow/best_model.pt \
    --output /dtu/blackhole/1b/223803/results/multiscale_vae_elbow_512 \
    --data-dir /dtu/blackhole/1b/223803/tcv_data \
    --variables n phi
