# Deep Learning for Fusion Plasma Dynamics

This repository contains a course project developed at the **Technical University of Denmark (DTU)** for **02456 – Deep Learning (Fall 2025)**.  
The project investigates deep learning methods for **modeling, forecasting, and reconstructing tokamak plasma dynamics** using high-resolution simulation data.

## Project Summary

Fusion plasma simulations are computationally expensive and diagnostic measurements are sparse. This project explores whether modern deep learning models can act as efficient surrogates for plasma simulation and reconstruction.

The work focuses on three tasks:

- **Plasma Forecasting (Fourier Neural Operator)**  
  Multi-step spatiotemporal prediction with resolution-invariant generalization (zero-shot super-resolution).

- **Prediction Refinement (Diffusion Models)**  
  Diffusion-based denoising to sharpen neural operator predictions; effective at coarse scales but limited by artifacts.

- **State Reconstruction from Sparse Probes (VAE)**  
  Multi-scale VAE trained on full plasma frames, combined with a probe encoder to reconstruct global plasma states from sparse measurements.

## Data

- Tokamak (TCV) plasma simulation data (2D fields at 512×512 resolution)
- Training on DTU HPC 
