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
    gen_llla_ensemble_samples,
    gen_lora_ensemble_samples,
)

from .flare import generate_flare_scores as _generate_flare_scores
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
def generate_llla_models(obj):
    lc = obj['laplace-ensemble']
    dc = obj['deep-ensemble']

    build_laplace_ensemble(
        trained_models_dir=dc['trained_models_dir'],
        llla_sampled_models_dir=lc['llla_sampled_models_dir'],
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
        max_posterior_std=lc.get('max_posterior_std', 1.0),
        std_reference_subnetwork_size=lc.get('std_reference_subnetwork_size', 1000),
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
@click.option('--ensemble-type', type=click.Choice(['deep', 'llla', 'lora']), required=True)
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
    elif ensemble_type == 'llla':
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


@cli.command()
@click.pass_obj
def generate_flare_scores(obj):
    lc = obj['laplace-ensemble']
    dc = obj['deep-ensemble']

    chkpt_dir = Path(dc['trained_models_dir'].format(seed=0))
    real_data_path = chkpt_dir / "real_dataset.npy"

    _generate_flare_scores(
        n_score_samples=lc['n_score_samples'],
        device=obj['device'],
        flare_samples_cache_dir=lc['flare_samples_cache_dir'],
        trained_models_dir=dc['trained_models_dir'],
        sel_generation=dc['sel_generation'],
        real_data_path=str(real_data_path),
        subset=lc.get('subset', 'last_layer'),
        m=lc.get('m', 1000),
        prior_precision=lc.get('prior_precision', 1e-2),
        last_layer_name=lc.get('last_layer_name', 'out_fc'),
        seed=lc.get('subset_seed', 42),
        max_posterior_std=lc.get('max_posterior_std', 1.0),
        std_reference_subnetwork_size=lc.get('std_reference_subnetwork_size', 1000),
        n_batches=lc['laplace_batches'],
        batch_size=lc['laplace_batch_size'],
    )