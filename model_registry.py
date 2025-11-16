from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import lightning as L
from models import FNOModel


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
        description='Fourier Neural Operator for plasma simulation.',
        default_params={
            'in_neurons': 32,
            'hidden_neurons': 64,
            'out_neurons': 32,
            'modesSpace': 8,
            'modesTime': 8,
            'input_size': 4,
            'learning_rate': 1e-3,
            'restart_at_epoch_n': 0,
            'loss_function': 'L2',
        },
    ),
}

def available_models() -> Tuple[str, ...]:
    """Return tuple of available model names."""
    return tuple(MODEL_REGISTRY.keys())


def resolve_model(name: str) -> ModelSpec:
    """Get model specification by name."""
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
    print(f'  Neurons: in={params["in_neurons"]}, hidden={params["hidden_neurons"]}, out={params["out_neurons"]}')
    print(f'  Modes: space={params["modesSpace"]}, time={params["modesTime"]}')
    print(f'  Learning rate: {params["learning_rate"]}, Loss: {params["loss_function"]}')
    
    return model, model_config


def rebuild_model_from_config(model_config: dict) -> L.LightningModule:
    return build_model(
        name=model_config['name'],
        **model_config['kwargs']
    )[0]



__all__ = [
    'ModelSpec',
    'available_models',
    'resolve_model',
    'build_model',
    'rebuild_model_from_config',
    'MODEL_REGISTRY',
]