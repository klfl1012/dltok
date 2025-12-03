import argparse
from dataclasses import dataclass
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from datetime import datetime
import os
from dotenv import load_dotenv
from copy import deepcopy
import wandb
from model_registry import (
    build_model,
    available_models,
    load_model_from_checkpoint,
    load_checkpoint_data,
)
from dataloader import build_dataloader

load_dotenv()

def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')

def parse_noise_std(value):
    """Parses noise_std input to float or tuple of two floats."""
    # Split input by commas to handle tuple-like input
    if ',' in value:
        # Attempt to convert to a tuple of floats
        try:
            return tuple(float(x) for x in value.split(','))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid input format: {value}. Expected 'float,float'.")
    else:
        # Attempt to convert to a float
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float format: {value}.")
        
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
        '--dataset_name',
        type=str,
        default='PlasmaDataset',
        help='Which dataset to use - "PlasmaDataset" for FNO and "DiffusionDataset" for denoising tasks'
    )

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

    parser.add_argument(
        '--noise_mean',
        type=float,
        default=0.0,
        help='The gaussian noise mean for the diffusion dataset'
    )
    parser.add_argument(
        '--noise_std',
        type=parse_noise_std,
        default=0.02,
        help='The gaussian noise std for the diffusion dataset. Can either be a float or a tuple. If tuple, it is a range of values the std can take'
    )
    parser.add_argument(
        '--diffusion_dataset_use_random_crop',
        action='store_true',
        default=True,
        help='Whether to use random cropping in DiffusionDataset'
    )
    parser.add_argument(
        '--diffusion_dataset_min_crop',
        type=int,
        default=32,
        help='Size of the minimum random crop in DiffusionDataset'
    )
    parser.add_argument(
        '--diffusion_dataset_max_crop',
        type=int,
        default=None,
        help='Size of the maximum random crop in DiffusionDataset'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        default=False,
        help='Use Weights & Biases for logging instead of TensorBoard'
    )
    parser.add_argument(
        '--normalize',
        type=_str2bool,
        nargs='?',
        const=True,
        default=True,
        metavar='{true,false}',
        help='Enable or disable channel-wise normalization (pass true/false)'
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
        choices=['MSE', 'L2', 'MSE+Grad', 'H1', 'LpLoss', 'MHD'],
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
    parser.add_argument(
        '--rank',
        type=float,
        default=None,
        help='Low-rank factorization ratio for FNO/TFNO (e.g. 0.05)'
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
        default=str(os.getenv('CHECKPOINT_ROOT', 'checkpoints')),
        # default=str(Path('/dtu/blackhole/1b/191611/DL/ckpts/')),
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
        '--enable_val_image_logging',
        action='store_true',
        default=False,
        help='Enable image logging during validation (overrides model default)'
    )
    parser.add_argument(
        '--enable_inference_image_logging',
        action='store_true',
        help='Enable image logging during inference/predict using the same sequence visualization'
    )

    parser.add_argument(
        '--timestep_to_show',
        type=str,
        default=None,
        help='Which timestep(s) to show when logging images: None (default), "last", "random", an int index, or comma-separated list of ints'
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

    parser.add_argument(
        '--run_inference_after_train',
        action='store_true',
        help='After training finishes, immediately run inference using the best checkpoint'
    )
    parser.add_argument(
        '--ignore_checkpoint_data_config',
        action='store_true',
        help='Use CLI data params during inference instead of those saved in the checkpoint'
    )

    # Ablation study controls
    parser.add_argument(
        '--ablation_study',
        action='store_true',
        help='Enable automated ablation study across seq_len and/or spatial_resolution (train mode only)'
    )

    return parser.parse_args()


def _train(args):
    print(f'\nInitializing training for model "{args.model}"...\n')
    wandb_closed = False
    
    print('Building dataloaders...')
    train_loader, val_loader, test_loader = build_dataloader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        spatial_resolution=args.spatial_resolution,
        normalize=args.normalize,
        dataset_name=args.dataset_name,
    )

    sample_sequence, _ = train_loader.dataset[0]
    # sample_sequence shape: [T, C, X, Y] or [T, X, Y]
    num_channels = sample_sequence.shape[1] if sample_sequence.ndim == 4 else 1

    print('Building model...')
    overrides = {}
    # ensure model is created with matching input/output channels
    overrides['in_channels'] = num_channels
    overrides['out_channels'] = num_channels
    data_config = {
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,
        'spatial_resolution': args.spatial_resolution,
        'seed': args.seed,
        'normalize': args.normalize,
        'num_channels': num_channels,
    }
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
    if args.rank is not None:
        overrides['rank'] = args.rank
    if args.num_predictions_to_log is not None:
        overrides['num_predictions_to_log'] = args.num_predictions_to_log
    if args.enable_val_image_logging:
        overrides['enable_val_image_logging'] = True
    if args.enable_inference_image_logging:
        overrides['enable_inference_image_logging'] = True
    overrides['data_config'] = data_config

    model, model_config = build_model(args.model, **overrides)
    object.__setattr__(model, 'sample_to_show', args.timestep_to_show)
    monitor_metric = f"val_{model.loss_name}_loss"

    print(
        "Configured run:" 
        f" seq_len={args.seq_len},"
        f" spatial_resolution={args.spatial_resolution},"
        f" n_layers={model_config['kwargs'].get('n_layers')},"
        f" hidden_channels={model_config['kwargs'].get('hidden_channels')},"
        f" n_modes={model_config['kwargs'].get('n_modes')}"
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.model}_res{args.spatial_resolution}_seq{args.seq_len}_{timestamp}"

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print('Setting up trainer...')
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f'{run_name}-best',
        monitor=monitor_metric,
        mode='min',
        save_top_k=1,
        save_last=False,
    )
    callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=args.patience,
            mode='min',
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Setup logger (WandB or TensorBoard)
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
        'normalize': args.normalize,
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
    
    print(f'\nStarting training ({args.max_epochs} epochs)...\n')
    trainer.fit(model, train_loader, val_loader)
    
    if args.test:
        print(f'\nTesting best model...')
        trainer.test(model, test_loader, ckpt_path='best')
    
    if args.run_inference_after_train:
        if args.use_wandb and not wandb_closed:
            wandb.finish()
            wandb_closed = True
        checkpoint_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        if checkpoint_path:
            print(f'\nRunning post-training inference with checkpoint: {Path(checkpoint_path).name}\n')
            inference_args = deepcopy(args)
            inference_args.mode = 'inference'
            inference_args.checkpoint = checkpoint_path
            inference_args.run_inference_after_train = False
            _inference(inference_args)
        else:
            print('\nWarning: No checkpoint available for post-training inference. Skipping.\n')

    print(f'\nTraining complete! Best checkpoint: {checkpoint_callback.best_model_path}\n')
    
    if args.use_wandb and not wandb_closed:
        wandb.finish()


def _run_ablation(args):
    """Execute a small ablation study across seq_len and/or spatial resolution."""

    @dataclass(frozen=True)
    class AblationDefaults:
        seq_lens: tuple[int, ...] = (1, 2, 4, 5)

    seq_values = AblationDefaults.seq_lens

    print(f"\nStarting ablation study ({len(seq_values)} runs)'\n")

    for idx, seq_len in enumerate(seq_values, start=1):
        print(f"\n[Ablation {idx}/{len(seq_values)}] seq_len={seq_len}\n")
        run_args = deepcopy(args)
        run_args.ablation_study = False # Prevent recursion
        run_args.seq_len = seq_len
        _train(run_args)


def _inference(args):

    if not args.checkpoint:
        raise ValueError('--checkpoint is required for inference mode')
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    print(f'\nRunning inference with checkpoint: {checkpoint_path.name}\n')

    checkpoint_data = load_checkpoint_data(str(checkpoint_path))
    checkpoint_hparams = checkpoint_data.get('hyper_parameters', {})
    checkpoint_data_config = checkpoint_hparams.get('data_config')

    if checkpoint_data_config and not args.ignore_checkpoint_data_config:
        for key in ('seq_len', 'spatial_resolution', 'batch_size', 'seed'):
            value = checkpoint_data_config.get(key)
            if value is not None:
                setattr(args, key, value)
            args.normalize = checkpoint_data_config.get('normalize', args.normalize)
        print(
            'Using data config from checkpoint '
            f"(seq_len={args.seq_len}, spatial_resolution={args.spatial_resolution}, "
            f"batch_size={args.batch_size}, seed={args.seed})"
        )
    elif not checkpoint_data_config:
        print('Note: checkpoint does not contain data configuration metadata; using CLI values.')
    else:
        print('Ignoring checkpoint data configuration per --ignore_checkpoint_data_config flag.')
    
    print('Building dataloader...')
    _, _, test_loader = build_dataloader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        spatial_resolution=args.spatial_resolution,
        normalize=args.normalize,
        dataset_name=args.dataset_name,
    )

    print('Loading model weights...')
    model = load_model_from_checkpoint(
        str(checkpoint_path),
        model_name=args.model,
        checkpoint_data=checkpoint_data,
    )
    object.__setattr__(model, 'sample_to_show', args.timestep_to_show)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.model}_inference_res{args.spatial_resolution}_seq{args.seq_len}_{timestamp}"
    logger = None

    if args.use_wandb:
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        else:
            print("Warning: No WandB API key found for inference logging. Falling back to TensorBoard.")
            args.use_wandb = False

    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            save_dir=args.log_dir,
            log_model=False,
        )
        print(f"Using Weights & Biases logging for inference (Project: {args.wandb_project})")
    else:
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=run_name,
            default_hp_metric=False,
        )
        print("Using TensorBoard logging for inference")

    logger.log_hyperparams({
        'mode': 'inference',
        'model': args.model,
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,
        'spatial_resolution': args.spatial_resolution,
        'seed': args.seed,
            'normalize': args.normalize,
        'checkpoint': str(checkpoint_path),
    })

    trainer = L.Trainer(
        accelerator='auto',
        devices='auto',
        logger=logger,
        enable_progress_bar=True,
    )

    if args.use_wandb or args.enable_inference_image_logging:
        object.__setattr__(model, 'enable_inference_image_logging', True)

    print('\nEvaluating on test set...')
    test_results = trainer.test(model, dataloaders=test_loader, ckpt_path=None)
    if test_results:
        summary = ', '.join(f"{k}={v:.6f}" for k, v in test_results[0].items())
        print(f"Test metrics: {summary}")

    print('\nRunning predictions...\n')
    predictions = trainer.predict(model, test_loader)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'predictions.pt'
    
    torch.save(predictions, output_file)
    print(f'\nPredictions saved to: {output_file}\n')

    if args.use_wandb:
        artifact = wandb.Artifact(name=f'{run_name}-predictions', type='predictions')
        artifact.add_file(str(output_file))
        logger.experiment.log_artifact(artifact)
        wandb.finish()


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
