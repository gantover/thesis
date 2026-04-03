import copy
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ddpm_torch.modules import Linear as CustomLinear
from .ensemble_weights import _get_diffusion_target

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(torch.empty(linear_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.empty(r, linear_layer.out_features))
        self.scaling = alpha / r
        
        # Correctly initialize lora_A with fan_in = in_features
        # kaiming_uniform_ expects weight shape (out_features, in_features), but our A is (in_features, r)
        # We can just use uniform_ with bounds 1 / sqrt(in_features)
        bound = 1 / math.sqrt(linear_layer.in_features)
        nn.init.uniform_(self.lora_A, -bound, bound)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

def inject_lora(model, r=4, alpha=1.0):
    """Recursively replaces all nn.Linear and CustomLinear layers in the model with LoRALinear in place."""
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, CustomLinear)):
            setattr(model, name, LoRALinear(module, r, alpha))
        else:
            inject_lora(module, r, alpha)

def build_lora_ensemble(
    base_model,
    diffusion,
    real_data,
    save_dir,
    M=5,
    r=4,
    alpha=1.0,
    epochs=10,
    batch_size=2048,
    lr=1e-3,
    device="cpu"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard Dataset for pure SGD (no replacement needed)
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(real_data, dtype=torch.float32))
    
    models = []
    for i in range(M):
        print(f"Training LoRA ensemble member {i+1}/{M}...")
        
        # Fresh independent adapter
        model_copy = copy.deepcopy(base_model)
        inject_lora(model_copy, r=r, alpha=alpha)
        model_copy.to(device)
        model_copy.train()
        
        # Only optimize the LoRA tiny parameter matrices
        optimizer = torch.optim.Adam([p for p in model_copy.parameters() if p.requires_grad], lr=lr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for (x_0,) in loader:
                x_0 = x_0.to(device)
                
                # Forward Diffusion
                t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device)
                noise = torch.randn_like(x_0)
                x_t = diffusion.q_sample(x_0, t, noise=noise)
                
                # Training Target
                target = _get_diffusion_target(diffusion, x_0, x_t, noise, t)
                
                # Parameter Efficient Tuning
                optimizer.zero_grad()
                out = model_copy(x_t, t) 
                loss = torch.nn.functional.mse_loss(out, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % max(1, (epochs // 5)) == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(loader):.4f}")
        
        # Save cache
        model_copy.eval()
        cache_path = save_dir / f"lora_sample_{i}.pt"
        torch.save(model_copy.state_dict(), cache_path)
        print(f"  -> Saved LoRA member to {cache_path}")
        models.append(model_copy)
        
    return models
