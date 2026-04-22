import os
import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from diffusers import LDMSuperResolutionPipeline, DDPMScheduler
from peft import LoraConfig, BOFTConfig, get_peft_model
from tqdm import tqdm

@click.command()
@click.option("--data_dir", default="/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train", help="Path to ImageNet train")
@click.option("--out_dir", default="./ensemble_weights", help="Output directory for model weights")
@click.option("--m_models", default=5, help="Number of ensemble models (M)")
@click.option("--subset_size", default=10000, help="Number of random images per model")
@click.option("--epochs", default=2, help="Epochs per ensemble member")
@click.option("--batch_size", default=8, help="Batch size (8 fits well on A100 40GB for LDM)")
@click.option("--peft_method", type=click.Choice(["lora", "boft"]), default="lora")
def train_ensemble(data_dir, out_dir, m_models, subset_size, epochs, batch_size, peft_method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Base Model Components (we only need VQVAE and UNet for training)
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    pipe = LDMSuperResolutionPipeline.from_pretrained(model_id).to(device)
    vqvae, unet, scheduler = pipe.vqvae, pipe.unet, pipe.scheduler
    vqvae.requires_grad_(False) # Freeze VQVAE
    
    # 2. Dataset Setup: Crop to 256x256 (Target), scale to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 3. PEFT Config (Targeting Attention and Convs to mimic Deep Ensembles)
    # target_modules = ["to_q", "to_k", "to_v", "to_out.0", "conv1", "conv2", "conv_in", "conv_out"]

    target_modules = [
        "to_q", "to_k", "to_v", "to_out.0", # Grabs all Attention blocks
        "conv",                             # Grabs conv1, conv2, conv_in, conv_out, conv_shortcut
        "proj"                              # Grabs linear projection layers
    ]
    if peft_method == "lora":
        peft_config = LoraConfig(r=64, lora_alpha=32, target_modules=target_modules)
    else:
        peft_config = BOFTConfig(boft_block_size=8, target_modules=target_modules)

    # 4. Sequential Ensemble Training
    for m in range(m_models):
        print(f"\n--- Training Ensemble Member {m}/{m_models - 1} ---")
        
        # 1. Inject fresh PEFT adapter into UNet
        peft_unet = get_peft_model(unet, peft_config)
        peft_unet.train()
        optimizer = torch.optim.AdamW(peft_unet.parameters(), lr=5e-5)
        
        # Create random subset for this model
        indices = torch.randperm(len(full_dataset))[:subset_size].tolist()
        subset = Subset(full_dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)

        for epoch in range(epochs):
            loss_m = 0
            for step, (high_res_imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                high_res_imgs = high_res_imgs.to(device)
                low_res_imgs = F.interpolate(high_res_imgs, size=(64, 64), mode="bicubic", align_corners=False)

                with torch.no_grad():
                    latents = vqvae.encode(high_res_imgs).latents
                    
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                latent_model_input = torch.cat([noisy_latents, low_res_imgs], dim=1)
                noise_pred = peft_unet(latent_model_input, timesteps).sample
                
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_m += loss.item()

            print(f"Epoch {epoch+1} Loss: {loss_m/len(dataloader):.4f}")

        # 2. Save PEFT weights
        save_path = os.path.join(out_dir, str(m))
        peft_unet.save_pretrained(save_path)
        
        # 3. THE CLEAN FIX: Unload adapter and scrub the metadata tag manually
        peft_unet = peft_unet.unload()
        if hasattr(unet, 'peft_config'):
            del unet.peft_config

if __name__ == "__main__":
    train_ensemble()