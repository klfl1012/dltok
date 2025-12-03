from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import lightning as L
from model import FNOModel, TFNOModel
import torch


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_class: type[L.LightningModule]
    description: str = ''
    default_params: Optional[dict] = None


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    'fno': ModelSpec(
        name='FNO',
        model_class=FNOModel,
        description='Fourier Neural Operator with neuralop backend for plasma simulation.',
        default_params={
            'n_modes': (8, 8, 8),
            'hidden_channels': 64,
            'in_channels': 1,  
            'out_channels': 1, 
            'n_layers': 4,
            'stabilizer': None,
            'norm': None,
            'preactivation': False,
            'fno_skip': 'linear',
            'separable': False,
            'factorization': None,
            'rank': 1.0,
            'fixed_rank_modes': False,
            'implementation': 'factorized',
            'decomposition_kwargs': None,
            'domain_padding': None,
            'learning_rate': 1e-3,
            'loss_function': 'MSE',
            'num_predictions_to_log': 1,
            'log_images_every_n_epochs': 1,
            'max_image_logging_epochs': None,
            'enable_val_image_logging': False,
            'enable_inference_image_logging': False,
            'data_config': None,
        },
    ),
    'tfno': ModelSpec(
        name='TFNO',
        model_class=TFNOModel,
        description='Temporal Fourier Neural Operator with neuralop backend for plasma simulation.',
        default_params={
            'n_modes': (32, 32, 8),
            'hidden_channels': 96,
            'in_channels': 1,
            'out_channels': 1,
            'n_layers': 4,
            'stabilizer': None,
            'norm': None,
            'preactivation': False,
            'fno_skip': 'linear',
            'separable': False,
            'factorization': 'tucker',
            'rank': 0.05,
            'fixed_rank_modes': False,
            'implementation': 'factorized',
            'decomposition_kwargs': None,
            'domain_padding': None,
            'learning_rate': 1e-3,
            'loss_function': 'MSE',
            'num_predictions_to_log': 1,
            'log_images_every_n_epochs': 1,
            'max_image_logging_epochs': None,
            'enable_val_image_logging': False,
            'enable_inference_image_logging': False,
            'data_config': None,
        },
    ),
}

def available_models() -> Tuple[str, ...]:
    return tuple(MODEL_REGISTRY.keys())


def resolve_model(name: str) -> ModelSpec:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise KeyError(
            f'Unknown model "{name}". Available models: {available}'
        )
    return MODEL_REGISTRY[key]


def build_model(
    name: str,
    **model_kwargs
) -> Tuple[L.LightningModule, dict]:

    spec = resolve_model(name)
    params = spec.default_params.copy() if spec.default_params else {}
    params.update(model_kwargs)
    
    model = spec.model_class(**params)
    
    model_config = {
        'name': name,
        'kwargs': params,
    }
    
    print(f'Built {spec.name}: {spec.description}')
    print(f'  Modes: {params.get("n_modes", "N/A")}')
    print(f'  Hidden channels: {params.get("hidden_channels", "N/A")}')
    print(f'  Layers: {params.get("n_layers", "N/A")}')
    print(f'  Learning rate: {params["learning_rate"]}, Loss: {params["loss_function"]}')
    
    return model, model_config


def rebuild_model_from_config(model_config: dict) -> L.LightningModule:
    return build_model(
        name=model_config['name'],
        **model_config['kwargs']
    )[0]


def load_checkpoint_data(
    checkpoint_path: str,
    map_location: torch.device | str | None = None,
    weights_only: bool = False,
) -> dict:
    if map_location is not None:
        resolved_location = map_location
    elif torch.cuda.is_available():
        resolved_location = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        resolved_location = torch.device('mps')
    else:
        resolved_location = torch.device('cpu')
    try:
        return torch.load(
            checkpoint_path,
            map_location=resolved_location,
            weights_only=weights_only,
        )
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        return torch.load(
            checkpoint_path,
            map_location=resolved_location,
        )


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str = 'fno',
    checkpoint_data: dict | None = None,
    map_location: torch.device | str | None = None,
    weights_only: bool = False,
) -> L.LightningModule:
    spec = resolve_model(model_name)

    checkpoint = checkpoint_data or load_checkpoint_data(
        checkpoint_path,
        map_location=map_location,
        weights_only=weights_only,
    )
    state_dict = checkpoint.get('state_dict', {})
    state_dict.pop('_metadata', None)

    hyper_parameters = checkpoint.get('hyper_parameters', {})
    model = spec.model_class(**hyper_parameters)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model



__all__ = [
    'ModelSpec',
    'available_models',
    'resolve_model',
    'build_model',
    'rebuild_model_from_config',
    'load_model_from_checkpoint',
    'load_checkpoint_data',
    'MODEL_REGISTRY',
]