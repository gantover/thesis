from pathlib import Path

def get_param_str(prior_precision, approximation, curvature, subset, m=None, temperature=1.0):
    m_str = f"_m{m}" if subset == "random" else ""
    temp_str = f"_temp{temperature}" if temperature != 1.0 else ""
    curv_str = f"_curv{curvature}" if approximation != "icla" else ""
    param_str = f"prior{prior_precision}{temp_str}_approx{approximation}{curv_str}_subset{subset}{m_str}"
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