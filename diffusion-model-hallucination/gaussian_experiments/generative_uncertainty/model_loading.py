import torch
from pathlib import Path
from ddpm_torch.toy import Decoder
from .lora_ensemble import inject_lora
from .utils import get_param_str_la, get_param_str_la_lora
from .config import LaplaceEnsembleConfig, LaplaceLoraEnsembleConfig, DeepEnsembleConfig, LoraEnsembleConfig

def load_model_from_checkpoint(chkpt_path, device):
    model = Decoder(in_features=2, mid_features=128, num_temporal_layers=3)
    try:
        checkpoint = torch.load(chkpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.to(device)
    model.eval()
    return model


def load_lora_model_from_checkpoint(chkpt_path, device, r: int, alpha: float):
    model = Decoder(in_features=2, mid_features=128, num_temporal_layers=3)
    inject_lora(model, r=r, alpha=alpha)
    try:
        checkpoint = torch.load(chkpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.to(device)
    model.eval()
    return model

def load_deep_ensemble_models(de_config: DeepEnsembleConfig, device: torch.device):
    total_models = de_config.M + 1
    print("Loading ensemble models from independent checkpoints...")
    models = []
    for model_seed in range(total_models):
        chkpt_dir = Path(de_config.trained_models_dir.format(seed=model_seed))
        chkpt_path = chkpt_dir / f"ddpm_gaussian25_gen_{de_config.sel_generation}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}")
        models.append(load_model_from_checkpoint(chkpt_path=chkpt_path, device=device))
    return models

# def load_la_sampled_models(la_sampled_models_dir, M, device, prior_precision, approximation, curvature, subset, m, temperature):
def load_la_sampled_models(la_config: LaplaceEnsembleConfig, device: torch.device, de_config: DeepEnsembleConfig):
    models = []
    for model_id in range(de_config.M):
        param_str = get_param_str_la(la_config)
        chkpt_path = Path(la_config.la_sampled_models_dir) / f"la_sample_{model_id}_{param_str}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing LA sampled model checkpoint: {chkpt_path}")
        models.append(load_model_from_checkpoint(chkpt_path=chkpt_path, device=device))
    return models

def load_la_lora_sampled_models(la_lora_config: LaplaceLoraEnsembleConfig, device: torch.device, de_config: DeepEnsembleConfig):
    models = []
    param_str = get_param_str_la_lora(la_lora_config)
    chkpt_dir = Path(la_lora_config.la_lora_sampled_models_dir)
    for model_id in range(de_config.M):
        candidates = [
            chkpt_dir / f"la_lora_sample_{model_id}_{param_str}.pt",
            chkpt_dir / f"lora_la_sample_{model_id}_{param_str}.pt",
        ]
        chkpt_path = next((p for p in candidates if p.exists()), None)
        if chkpt_path is None:
            raise FileNotFoundError(
                "Missing LA LoRA sampled model checkpoint. Checked: "
                + ", ".join(str(p) for p in candidates)
            )
        models.append(
            load_lora_model_from_checkpoint(
                chkpt_path=chkpt_path,
                device=device,
                r=la_lora_config.r,
                alpha=la_lora_config.alpha,
            )
        )
    return models


# def load_lora_ensemble_models(lora_sampled_models_dir, base_model, M, r, alpha, device):
def load_lora_ensemble_models(lora_config: LoraEnsembleConfig, de_config: DeepEnsembleConfig, device: torch.device):
    models = []
    for model_id in range(de_config.M):
        chkpt_path = Path(lora_config.lora_sampled_models_dir) / f"lora_sample_{model_id}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing LoRA sampled model checkpoint: {chkpt_path}")
        models.append(
            load_lora_model_from_checkpoint(
                chkpt_path=chkpt_path,
                device=device,
                r=lora_config.r,
                alpha=lora_config.alpha,
            )
        )
    return models
    # from .lora_ensemble import inject_lora
    # import copy
    # models = []
    # for model_id in range(de_config.M):
    #     chkpt_path = Path(lora_config.lora_sampled_models_dir) / f"lora_sample_{model_id}.pt"
    #     if not chkpt_path.exists():
    #         raise FileNotFoundError(f"Missing LoRA sampled model checkpoint: {chkpt_path}")
    #     # model = copy.deepcopy(base_model)
    #     # inject_lora(model, r=lora_config.r, alpha=lora_config.alpha)
    #     try:
    #         checkpoint = torch.load(chkpt_path, map_location=device, weights_only=False)
    #     except TypeError:
    #         checkpoint = torch.load(chkpt_path, map_location=device)
    #     model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    #     model.to(device)
    #     model.eval()
    #     models.append(model)
    # return models

def load_base_model(de_config: DeepEnsembleConfig, device: torch.device):
    chkpt_dir = Path(de_config.trained_models_dir.format(seed=0))
    chkpt_path = chkpt_dir / f"ddpm_gaussian25_gen_{de_config.sel_generation}.pt"
    if not chkpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}")
    return load_model_from_checkpoint(chkpt_path=chkpt_path, device=device)
