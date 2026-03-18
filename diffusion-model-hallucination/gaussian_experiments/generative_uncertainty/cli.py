import click
import yaml
from pathlib import Path
import os
import torch

from .ensemble_weights import (
    build_last_layer_laplace_models,
    get_diffusion,
)

from .ensemble_sampling import (
    gen_deep_ensemble_samples,
    gen_llla_ensemble_samples,
)

@click.group()
@click.option("--config", default="config.yml")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    config_data = {}
    if os.path.exists(config):
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f) or {} 
    else:
        click.echo(f"Warning: Config file '{config}' not found. Using defaults.", err=True)
    ctx.obj.update(config_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ctx.obj['device'] = device

@cli.command()
@click.pass_obj
def generate_llla_models(obj):
    laplace_ensemble_config = obj['laplace-ensemble']
    deep_ensemble_config = obj['deep-ensemble']

    diffusion = get_diffusion()

    build_last_layer_laplace_models(
        trained_models_dir=deep_ensemble_config['trained_models_dir'],
        llla_sampled_models_dir=laplace_ensemble_config['llla_sampled_models_dir'],
        diffusion=diffusion,
        device=obj['device'],
        sel_generation=deep_ensemble_config['sel_generation'],
        M=deep_ensemble_config['M'],
        laplace_batches=laplace_ensemble_config['laplace_batches'],
        laplace_batch_size=laplace_ensemble_config['laplace_batch_size'],
        weight_sampling_seed=laplace_ensemble_config['weight_sampling_seed'],
    )

@cli.command()
@click.pass_obj
def generate_llla_samples(obj):
    sampling_config = obj['sampling']
    gen_llla_ensemble_samples(
        num_samples=sampling_config['num_samples'],
        batch_size=sampling_config['batch_size'],
        device=obj['device'],
        samples_cache_dir=sampling_config['samples_cache_dir'],
        llla_sampled_models_dir=obj['laplace-ensemble']['llla_sampled_models_dir'],
        trained_models_dir=obj['deep-ensemble']['trained_models_dir'],
        sel_generation=obj['deep-ensemble']['sel_generation'],
        M=obj['deep-ensemble']['M'],
    )

@cli.command()
@click.pass_obj
def generate_deep_ensemble_samples(obj):
    sampling_config = obj['sampling']
    gen_deep_ensemble_samples(
        num_samples=sampling_config['num_samples'],
        batch_size=sampling_config['batch_size'],
        device=obj['device'],
        samples_cache_dir=sampling_config['samples_cache_dir'],
        trained_models_dir=obj['deep-ensemble']['trained_models_dir'],
        sel_generation=obj['deep-ensemble']['sel_generation'],
        M=obj['deep-ensemble']['M'],
    )