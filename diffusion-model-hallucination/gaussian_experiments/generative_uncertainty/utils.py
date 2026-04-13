from pathlib import Path
from .config import LaplaceEnsembleConfig, LaplaceLoraEnsembleConfig

# def get_param_str_la(prior_precision, approximation, curvature, subset, m=None, temperature=1.0):
def get_param_str_la(la_config: LaplaceEnsembleConfig):
    m_str = f"_m{la_config.m}" if la_config.subset == "random" else ""
    temp_str = f"_temp{la_config.temperature}" if la_config.temperature != 1.0 else ""
    curv_str = f"_curv{la_config.curvature}" 
    param_str = f"prior{la_config.prior_precision}{temp_str}_approx{la_config.approximation}{curv_str}_subset{la_config.subset}{m_str}"
    return param_str

def get_param_str_la_lora(lora_config: LaplaceLoraEnsembleConfig):
    temp_str = f"_temp{lora_config.temperature}" if lora_config.temperature != 1.0 else ""
    param_str = f"prior{lora_config.prior_precision}{temp_str}_approx{lora_config.approximation}_curv{lora_config.curvature}_r{lora_config.r}_alpha{lora_config.alpha}"
    return param_str

def get_clean_label(filepath):
    name = Path(filepath).stem
    prefix = "la_ensemble_samples_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if not name or name == "la_ensemble_samples":
        return "LA (Default)"
        
    name = name.replace('prior', 'Prior: ')
    name = name.replace('_temp', ', Temp: ')
    name = name.replace('_approx', ', Approx: ')
    name = name.replace('_curv', ', Curv: ')
    name = name.replace('_subset', ', Subset: ')
    name = name.replace('_m', ', m: ')
    return name