import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ddpm_torch.modules import Linear as CustomLinear
from .laplace_hessian import _get_diffusion_target

try:
    from peft import get_peft_model, BOFTConfig
except ImportError:
    raise ImportError("Please install peft using `pip install peft` to use the OFT/BOFT ensemble module.")

def convert_to_standard_linear(model):
    """Replaces CustomLinear with standard nn.Linear so peft can recognize them."""
    for name, module in model.named_children():
        if isinstance(module, CustomLinear):
            # Create a standard linear layer with the same shape
            std_linear = nn.Linear(module.in_features, module.out_features, bias=(module.bias is not None))
            # Copy over the pre-trained weights
            std_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                std_linear.bias.data = module.bias.data.clone()
            # Replace it in the model
            setattr(model, name, std_linear)
        elif len(list(module.children())) > 0:
            convert_to_standard_linear(module)

def inject_oft(model, boft_block_size=4, boft_n_butterfly_factor=2):
    """Converts model to standard PyTorch layers and injects BOFT via PEFT."""
    convert_to_standard_linear(model)
    
    # Only target layers where the `in_features` are 128 (meaning divisible by block size = 4).
    # Since `in_features` of your data is 2, `in_fc` and `out_fc` (which has out=2, but in=128) 
    # might cause issues if peft assumes output size needs to be divisible by block size.
    # To be extremely safe, we target just the temporal layers which are 128 -> 128
    config = BOFTConfig(
        boft_block_size=boft_block_size,
        boft_n_butterfly_factor=boft_n_butterfly_factor,
        target_modules=r".*temp_fc.*fc.*", 
        modules_to_save=[], # Freeze the base model
    )
    
    peft_model = get_peft_model(model, config)
    return peft_model


def build_oft_ensemble(
    base_model,
    diffusion,
    real_data,
    save_dir,
    M=5,
    boft_block_size=4,
    boft_n_butterfly_factor=2,
    epochs=10,
    batch_size=2048,
    lr=1e-3,
    device="cpu"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(real_data, dtype=torch.float32))
    
    models = []
    for i in range(M):
        print(f"Training OFT (BOFT) ensemble member {i+1}/{M}...")
        
        # Fresh independent adapter
        model_copy = copy.deepcopy(base_model)
        model_copy = inject_oft(model_copy, boft_block_size=boft_block_size, boft_n_butterfly_factor=boft_n_butterfly_factor)
        model_copy.to(device)
        model_copy.train()
        
        # Determine parameters to optimize (only BOFT params)
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
        cache_path = save_dir / f"oft_sample_{i}.pt"
        
        # PEFT automatically handles saving state dict cleanly
        model_copy.save_pretrained(cache_path)
        print(f"  -> Saved OFT member adapter to {cache_path}")
        models.append(model_copy)
        
    return models
