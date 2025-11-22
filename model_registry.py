from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import lightning as L
from model import FNOModel


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
            'in_channels': 1,  # Single channel input (density field)
            'out_channels': 1,  # Single channel output
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


def load_model_from_checkpoint(checkpoint_path: str, model_name: str = 'fno') -> L.LightningModule:
    spec = resolve_model(model_name)
    model = spec.model_class.load_from_checkpoint(checkpoint_path)
    return model



__all__ = [
    'ModelSpec',
    'available_models',
    'resolve_model',
    'build_model',
    'rebuild_model_from_config',
    'load_model_from_checkpoint',
    'MODEL_REGISTRY',
]