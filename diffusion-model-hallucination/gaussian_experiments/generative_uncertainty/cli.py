import click
import os
import torch
from pathlib import Path

from .ensemble_weights import get_diffusion


from .model_loading import load_base_model
from .config import load_config, AppConfig
import numpy as np

from dataclasses import dataclass
@dataclass
class AppContext:
    config: AppConfig
    device: torch.device

@click.group()
@click.option("--config", default="config.yml")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    # config_data = {}
    loaded_config = None
    if os.path.exists(config):
        with open(config, 'r') as f:
            # config_data = yaml.safe_load(f) or {}
            loaded_config = load_config(config)
            
    else:
        click.echo(f"Warning: Config file '{config}' not found. Using defaults.", err=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ctx.obj = AppContext(config=loaded_config, device=device)

@cli.command('print-config')
@click.pass_obj
def print_config(app_ctx: AppContext):
    print("Current Configuration:")
    print(app_ctx.device)
    print(app_ctx.config.model_dump_json(indent=2))

@cli.command()
@click.pass_obj
def generate_la_models(app_ctx: AppContext):
    from generative_uncertainty.ensemble_weights import build_laplace_ensemble
    build_laplace_ensemble(
        la=app_ctx.config.laplace_ensemble,
        de=app_ctx.config.deep_ensemble,
        device=app_ctx.device,
        diffusion=get_diffusion()
    )

@cli.command('build-lora')
@click.pass_obj
def build_lora(app_ctx: AppContext):
    from .lora_ensemble import build_lora_ensemble
    lc = app_ctx.config.lora_ensemble
    dc = app_ctx.config.deep_ensemble

    chkpt_dir = Path(dc.trained_models_dir.format(seed=0))
    real_data_path = chkpt_dir / "real_dataset.npy"
    real_data = np.load(str(real_data_path))

    base_model = load_base_model(
        trained_models_dir=dc.trained_models_dir,
        sel_generation=dc.sel_generation,
        device=app_ctx.device
    )

    build_lora_ensemble(
        base_model=base_model,
        diffusion=get_diffusion(),
        real_data=real_data,
        save_dir=lc.lora_sampled_models_dir,
        M=lc.M,
        r=lc.r,
        alpha=lc.alpha,
        epochs=lc.epochs,
        batch_size=lc.batch_size,
        lr=lc.lr,
        device=app_ctx.device
    )

@cli.command('build-laplace-lora')
@click.pass_obj
def build_laplace_lora(app_ctx: AppContext):
    from .laplace_lora import build_laplace_lora_ensemble
    build_laplace_lora_ensemble(
        la=app_ctx.config.laplace_lora_ensemble,
        de=app_ctx.config.deep_ensemble,
        device=app_ctx.device,
        diffusion=get_diffusion()
    )

@cli.command('sample')
@click.option('--ensemble-type', type=click.Choice(['deep', 'la', 'lora', 'la-lora']), required=True)
@click.pass_obj
def sample(app_ctx: AppContext, ensemble_type):
    sampling_config = app_ctx.config.sampling
    if ensemble_type == 'deep':
        from .ensemble_sampling import gen_deep_ensemble_samples
        gen_deep_ensemble_samples(
            sampling_config = sampling_config,
            device=app_ctx.device,
            deep_ensemble_config=app_ctx.config.deep_ensemble,
        )
    elif ensemble_type == 'la':
        from .ensemble_sampling import gen_la_ensemble_samples
        gen_la_ensemble_samples(
            sampling_config = sampling_config,
            device=app_ctx.device,
            la_ensemble_config=app_ctx.config.la_ensemble,
            deep_ensemble_config=app_ctx.config.deep_ensemble
        )
    elif ensemble_type == 'lora':
        from .ensemble_sampling import gen_lora_ensemble_samples
        gen_lora_ensemble_samples(
            sampling_config = sampling_config,
            device=app_ctx.device,
            lora_ensemble_config=app_ctx.config.lora_ensemble,
            deep_ensemble_config=app_ctx.config.deep_ensemble
        )
    elif ensemble_type == 'la-lora':
        from .ensemble_sampling import gen_la_lora_ensemble_samples 
        gen_la_lora_ensemble_samples(
            sampling_config = sampling_config,
            device=app_ctx.device,
            laplace_lora_config=app_ctx.config.laplace_lora_ensemble,
            deep_ensemble_config=app_ctx.config.deep_ensemble
        )