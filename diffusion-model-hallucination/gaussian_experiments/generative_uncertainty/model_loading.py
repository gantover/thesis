import torch
from pathlib import Path
from ddpm_torch.toy import Decoder

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

def load_deep_ensemble_models(trained_models_dir, sel_generation, M, device):
    total_models = M + 1
    print("Loading ensemble models from independent checkpoints...")
    models = []
    for model_seed in range(total_models):
        chkpt_dir = Path(trained_models_dir.format(seed=model_seed))
        chkpt_path = chkpt_dir / f"ddpm_gaussian25_gen_{sel_generation}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}")
        models.append(load_model_from_checkpoint(chkpt_path=chkpt_path, device=device))
    return models

def load_la_sampled_models(la_sampled_models_dir, M, device, prior_precision, approximation, curvature, subset, m):
    models = []
    for model_id in range(M):
        m_str = f"_m{m}" if subset == "random" else ""
        param_str = f"prior{prior_precision}_approx{approximation}_curv{curvature}_subset{subset}{m_str}"
        chkpt_path = Path(la_sampled_models_dir) / f"la_sample_{model_id}_{param_str}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing LA sampled model checkpoint: {chkpt_path}")
        models.append(load_model_from_checkpoint(chkpt_path=chkpt_path, device=device))
    return models

def load_lora_ensemble_models(lora_sampled_models_dir, base_model, M, r, alpha, device):
    from .lora_ensemble import inject_lora
    import copy
    models = []
    for model_id in range(M):
        chkpt_path = Path(lora_sampled_models_dir) / f"lora_sample_{model_id}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing LoRA sampled model checkpoint: {chkpt_path}")
        model = copy.deepcopy(base_model)
        inject_lora(model, r=r, alpha=alpha)
        try:
            checkpoint = torch.load(chkpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        model.to(device)
        model.eval()
        models.append(model)
    return models

def load_base_model(trained_models_dir, sel_generation, device):
    chkpt_dir = Path(trained_models_dir.format(seed=0))
    chkpt_path = chkpt_dir / f"ddpm_gaussian25_gen_{sel_generation}.pt"
    if not chkpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}")
    return load_model_from_checkpoint(chkpt_path=chkpt_path, device=device)
