#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from tqdm import tqdm
from datetime import datetime
import os
from dotenv import load_dotenv

from copy import deepcopy
from itertools import product

from model_registry import build_model, available_models, load_model_from_checkpoint
from dataloader import build_dataloader

load_dotenv()


def _build_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description='Train or run inference with plasma simulation models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference'],
        default='train',
        help='Mode: train or inference'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='fno',
        choices=available_models(),
        help='Model architecture to use'
    )
    
    # Data parameters
    parser.add_argument(
        '--seq_len',
        type=int,
        default=100,
        help='Sequence length for time series'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training/inference'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--spatial_resolution',
        type=int,
        default=256,
        help='Target spatial resolution for downsampling (e.g., 256 to downsample 1024x1024 to 256x256)'
    )
    
    # Training parameters
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (overrides model default)'
    )
    parser.add_argument(
        '--loss_function',
        type=str,
        default=None,
        choices=['MSE', 'MSE+Grad'],
        help='Loss function (overrides model default)'
    )
    
    # Model hyperparameters (optional overrides)
    parser.add_argument(
        '--n_modes',
        type=str,
        default=None,
        help='Fourier modes as comma-separated tuple, e.g. "8,8,8" (FNO only)'
    )
    parser.add_argument(
        '--hidden_channels',
        type=int,
        default=None,
        help='Hidden channels width (FNO only)'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=None,
        help='Number of FNO layers (FNO only)'
    )
    
    # Callbacks
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test step after training'
    )
    
    # Paths
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for inference'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save inference outputs'
    )
    
    # Logging
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--num_predictions_to_log',
        type=int,
        default=1,
        help='Number of sequence predictions to log per validation epoch (each prediction shows all timesteps)'
    )
    parser.add_argument(
        '--disable_val_image_logging',
        action='store_true',
        help='Disable image logging during validation (overrides model default)'
    )
    parser.add_argument(
        '--enable_inference_image_logging',
        action='store_true',
        help='Enable image logging during inference/predict using the same sequence visualization'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        default=False,
        help='Use Weights & Biases for logging instead of TensorBoard'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='plasma-simulation',
        help='WandB project name'
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='WandB entity (username or team name)'
    )

    # Ablation study controls
    parser.add_argument(
        '--ablation_study',
        action='store_true',
        help='Enable automated ablation study across seq_len and/or spatial_resolution (train mode only)'
    )
    parser.add_argument(
        '--ablation_mode',
        type=str,
        choices=['seq_len', 'spatial_resolution', 'grid'],
        default='grid',
        help='Ablation strategy: vary seq_len only, spatial_resolution only, or run the Cartesian grid'
    )

    return parser.parse_args()


def _train(args):
    print(f'\nInitializing training for model "{args.model}"...\n')
    
    # Initialize and load overhead
    with tqdm(total=3, desc='Initializing', unit='step', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]') as pbar:

        pbar.set_description('Initializing: Building dataloaders')
        train_loader, val_loader, test_loader = build_dataloader(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
            spatial_resolution=args.spatial_resolution,
        )
        pbar.update(1)
        
        pbar.set_description('Initializing: Building model')
        overrides = {}
        if args.learning_rate is not None:
            overrides['learning_rate'] = args.learning_rate
        if args.loss_function is not None:
            overrides['loss_function'] = args.loss_function
        if args.n_modes is not None:
            overrides['n_modes'] = tuple(map(int, args.n_modes.split(',')))
        if args.hidden_channels is not None:
            overrides['hidden_channels'] = args.hidden_channels
        if args.n_layers is not None:
            overrides['n_layers'] = args.n_layers
        if args.num_predictions_to_log is not None:
            overrides['num_predictions_to_log'] = args.num_predictions_to_log
        if args.disable_val_image_logging:
            overrides['enable_val_image_logging'] = False
        if args.enable_inference_image_logging:
            overrides['enable_inference_image_logging'] = True
        
        model, model_config = build_model(args.model, **overrides)
        pbar.update(1)
        
        pbar.set_description('Initializing: Setting up trainer')
        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f'{args.model}-{{epoch:02d}}-{{val_MSE_loss:.4f}}',
            monitor='val_MSE_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        
        if args.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor='val_MSE_loss',
                patience=args.patience,
                mode='min',
                verbose=True,
            )
            callbacks.append(early_stop_callback)
        
        # Setup logger (WandB or TensorBoard)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.model}_res{args.spatial_resolution}_seq{args.seq_len}_{timestamp}"
        
        if args.use_wandb:
            api_key = os.environ.get('WANDB_API_KEY')
            if api_key:
                os.environ['WANDB_API_KEY'] = api_key
            else:
                print("Warning: No WandB API key found in .env file or environment variables.")
                print("Please set WANDB_API_KEY in your .env file or as an environment variable.")
                print("Falling back to TensorBoard logging.")
                args.use_wandb = False
        
        if args.use_wandb:
            logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                save_dir=args.log_dir,
                log_model=True,  
            )
            print(f"Using Weights & Biases logging (Project: {args.wandb_project})")
        else:
            logger = TensorBoardLogger(
                save_dir=args.log_dir,
                name=run_name,
                default_hp_metric=False,
            )
            print("Using TensorBoard logging")
        
        # Log hyperparameters
        logger.log_hyperparams({
            'model': args.model,
            'seq_len': args.seq_len,
            'batch_size': args.batch_size,
            'spatial_resolution': args.spatial_resolution,
            'max_epochs': args.max_epochs,
            'seed': args.seed,
            **model_config['kwargs'],
        })
        
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices='auto',
            log_every_n_steps=10,
            enable_progress_bar=True,
        )
        pbar.update(1)
    
    print(f'\nStarting training ({args.max_epochs} epochs)...\n')
    trainer.fit(model, train_loader, val_loader)
    
    if args.test:
        print(f'\nTesting best model...')
        trainer.test(model, test_loader, ckpt_path='best')
    
    print(f'\nTraining complete! Best checkpoint: {checkpoint_callback.best_model_path}\n')
    
    if args.use_wandb:
        import wandb
        wandb.finish()


def _run_ablation(args):
    """Execute a small ablation study across seq_len and/or spatial resolution."""

    @dataclass(frozen=True)
    class AblationDefaults:
        seq_lens: tuple[int, ...] = (32, 64, 96, 128)
        spatial_resolutions: tuple[int, ...] = (128, 192, 256)

    seq_values = AblationDefaults.seq_lens
    res_values = AblationDefaults.spatial_resolutions

    if args.ablation_mode == 'seq_len':
        combos = [(seq, args.spatial_resolution) for seq in seq_values]
    elif args.ablation_mode == 'spatial_resolution':
        combos = [(args.seq_len, res) for res in res_values]
    else:  
        combos = list(product(seq_values, res_values))

    total_runs = len(combos)
    print(f"\nStarting ablation study ({total_runs} runs) with mode='{args.ablation_mode}'\n")

    for idx, (seq_len, spat_res) in enumerate(combos, start=1):
        print(f"\n[Ablation {idx}/{total_runs}] seq_len={seq_len}, spatial_resolution={spat_res}\n")
        run_args = deepcopy(args)
        run_args.ablation_study = False  # Prevent recursion
        run_args.seq_len = seq_len
        run_args.spatial_resolution = spat_res
        _train(run_args)


def _inference(args):

    if not args.checkpoint:
        raise ValueError('--checkpoint is required for inference mode')
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    print(f'\nRunning inference with checkpoint: {checkpoint_path.name}\n')
    
    with tqdm(total=3, desc='Setup', unit='step') as pbar:
        pbar.set_description('Loading data')
        _, _, test_loader = build_dataloader(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
            spatial_resolution=args.spatial_resolution,
        )
        pbar.update(1)
        
        pbar.set_description('Loading model')
        model = load_model_from_checkpoint(str(checkpoint_path), model_name=args.model)
        pbar.update(1)
        
        pbar.set_description('Setting up trainer')
        trainer = L.Trainer(
            accelerator='auto',
            devices='auto',
            enable_progress_bar=True,
        )
        pbar.update(1)
    
    print(f'\nRunning predictions...\n')
    predictions = trainer.predict(model, test_loader)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'predictions.pt'
    
    torch.save(predictions, output_file)
    print(f'\nPredictions saved to: {output_file}\n')


def main():
    args = _build_args()
    
    if args.mode == 'train':
        if args.ablation_study:
            _run_ablation(args)
        else:
            _train(args)
    elif args.mode == 'inference':
        if args.ablation_study:
            raise ValueError('--ablation_study is only supported in train mode')
        _inference(args)


if __name__ == '__main__':
    main()
