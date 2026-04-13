import yaml
from pydantic import BaseModel

class DeepEnsembleConfig(BaseModel):
    trained_models_dir: str
    sel_generation: int
    M: int

class LaplaceEnsembleConfig(BaseModel):
    la_sampled_models_dir: str
    last_layer_name: str
    laplace_batches: int
    laplace_batch_size: int
    prior_precision: float
    temperature: float
    approximation: str
    curvature: str
    subset: str
    m: int
    weight_sampling_seed: int
    subset_seed: int

class LaplaceLoraEnsembleConfig(BaseModel):
    la_lora_sampled_models_dir: str
    # M: int
    r: int
    alpha: float
    map_epochs: int
    map_lr: float
    laplace_batches: int
    laplace_batch_size: int
    prior_precision: float
    temperature: float
    weight_sampling_seed: int
    curvature: str
    approximation: str

class LoraEnsembleConfig(BaseModel):
    lora_sampled_models_dir: str
    # M: int
    r: int
    alpha: float
    epochs: int
    lr: float
    batch_size: int

class SamplingConfig(BaseModel):
    num_samples: int
    batch_size: int
    samples_cache_dir: str

class AppConfig(BaseModel):
    laplace_ensemble: LaplaceEnsembleConfig
    laplace_lora_ensemble: LaplaceLoraEnsembleConfig
    lora_ensemble: LoraEnsembleConfig
    deep_ensemble: DeepEnsembleConfig
    sampling: SamplingConfig

# 2. Load the YAML and parse it into the model
def load_config(file_path: str) -> AppConfig:
    with open(file_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    # Pydantic will validate the types and build the object
    return AppConfig(**raw_config)

# # 3. Use it with full LSP support!
# config = load_config("config.yml")

# # Your IDE will now autocomplete this:
# print(config.laplace_ensemble.laplace_batch_size) 
# print(config.lora_ensemble.epochs)