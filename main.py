#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm
from datetime import datetime

from model_registry import build_model, available_models, load_model_from_checkpoint
from dataloader import build_dataloader

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
        '--num_images_to_log',
        type=int,
        default=1,
        help='Number of prediction images to log per validation epoch'
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
        
        # Setup TensorBoard logger
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.model}_res{args.spatial_resolution}_seq{args.seq_len}_{timestamp}"
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=run_name,
            default_hp_metric=False,
        )
        
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
        
        # Store num_images_to_log in model for validation_step
        model.num_images_to_log = args.num_images_to_log
        
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
        _train(args)
    elif args.mode == 'inference':
        _inference(args)


if __name__ == '__main__':
    main()
