import click
import yaml
import os
import torch
from pathlib import Path

from .ensemble_weights import (
    build_laplace_ensemble,
    get_diffusion,
)

from .ensemble_sampling import (
    gen_deep_ensemble_samples,
    gen_la_ensemble_samples,
    gen_lora_ensemble_samples,
)

from .lora_ensemble import build_lora_ensemble
from .model_loading import load_base_model
import numpy as np


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
def generate_la_models(obj):
    lc = obj['laplace-ensemble']
    dc = obj['deep-ensemble']

    build_laplace_ensemble(
        trained_models_dir=dc['trained_models_dir'],
        la_sampled_models_dir=lc['la_sampled_models_dir'],
        diffusion=get_diffusion(),
        device=obj['device'],
        sel_generation=dc['sel_generation'],
        M=dc['M'],
        laplace_batches=lc['laplace_batches'],
        laplace_batch_size=lc['laplace_batch_size'],
        weight_sampling_seed=lc['weight_sampling_seed'],
        sample_temperature=lc['sample_temperature'],
        prior_precision=lc.get('prior_precision', 1e-2),
        last_layer_name=lc.get('last_layer_name', 'out_fc'),
        subset=lc.get('subset', 'last_layer'),
        m=lc.get('m', 1000),
        subset_seed=lc.get('subset_seed', 42),
        curvature=lc.get('curvature', 'ef'),
        approximation=lc.get('approximation', 'diagonal'),
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

@cli.command('build-lora')
@click.pass_obj
def build_lora(obj):
    lc = obj['lora-ensemble']
    dc = obj['deep-ensemble']
    
    chkpt_dir = Path(dc['trained_models_dir'].format(seed=0))
    real_data_path = chkpt_dir / "real_dataset.npy"
    real_data = np.load(str(real_data_path))

    base_model = load_base_model(
        trained_models_dir=dc['trained_models_dir'],
        sel_generation=dc['sel_generation'],
        device=obj['device']
    )

    build_lora_ensemble(
        base_model=base_model,
        diffusion=get_diffusion(),
        real_data=real_data,
        save_dir=lc['lora_sampled_models_dir'],
        M=lc['M'],
        r=lc['r'],
        alpha=lc['alpha'],
        epochs=lc['epochs'],
        batch_size=lc['batch_size'],
        lr=lc['lr'],
        device=obj['device']
    )

@cli.command('sample')
@click.option('--ensemble-type', type=click.Choice(['deep', 'la', 'lora']), required=True)
@click.pass_obj
def sample(obj, ensemble_type):
    sampling_config = obj['sampling']
    if ensemble_type == 'deep':
        gen_deep_ensemble_samples(
            num_samples=sampling_config['num_samples'],
            batch_size=sampling_config['batch_size'],
            device=obj['device'],
            samples_cache_dir=sampling_config['samples_cache_dir'],
            trained_models_dir=obj['deep-ensemble']['trained_models_dir'],
            sel_generation=obj['deep-ensemble']['sel_generation'],
            M=obj['deep-ensemble']['M'],
        )
    elif ensemble_type == 'la':
        gen_la_ensemble_samples(
            num_samples=sampling_config['num_samples'],
            batch_size=sampling_config['batch_size'],
            device=obj['device'],
            samples_cache_dir=sampling_config['samples_cache_dir'],
            la_sampled_models_dir=obj['laplace-ensemble']['la_sampled_models_dir'],
            trained_models_dir=obj['deep-ensemble']['trained_models_dir'],
            sel_generation=obj['deep-ensemble']['sel_generation'],
            M=obj['deep-ensemble']['M'],
            prior_precision=obj['laplace-ensemble'].get('prior_precision', 1e-2),
            approximation=obj['laplace-ensemble'].get('approximation', 'diagonal'),
            curvature=obj['laplace-ensemble'].get('curvature', 'ef'),
            subset=obj['laplace-ensemble'].get('subset', 'last_layer'),
            m=obj['laplace-ensemble'].get('m', 1000),
            sample_temperature=obj['laplace-ensemble'].get('sample_temperature', 1.0),
        )
    elif ensemble_type == 'lora':
        lc = obj['lora-ensemble']
        gen_lora_ensemble_samples(
            num_samples=sampling_config['num_samples'],
            batch_size=sampling_config['batch_size'],
            device=obj['device'],
            samples_cache_dir=sampling_config['samples_cache_dir'],
            lora_sampled_models_dir=lc['lora_sampled_models_dir'],
            trained_models_dir=obj['deep-ensemble']['trained_models_dir'],
            sel_generation=obj['deep-ensemble']['sel_generation'],
            M=lc['M'],
            r=lc['r'],
            alpha=lc['alpha'],
        )