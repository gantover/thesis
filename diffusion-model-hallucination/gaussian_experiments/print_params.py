import yaml
import torch
from pathlib import Path
from generative_uncertainty.model_loading import load_base_model

config = "config.yml"
with open(config, 'r') as f:
    config_data = yaml.safe_load(f)

device = torch.device('cpu')
trained_models_dir = config_data['deep-ensemble']['trained_models_dir']
sel_generation = config_data['deep-ensemble']['sel_generation']

try:
    model = load_base_model(trained_models_dir, sel_generation, device)
    print("=== Model Architecture ===")
    print(model)
    print("\n=== Parameter Breakdown ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
            
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}")
except Exception as e:
    print(e)
