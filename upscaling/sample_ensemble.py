import os
import click
import torch
from torchvision.transforms.functional import to_pil_image
from diffusers import LDMSuperResolutionPipeline
from PIL import Image
from peft import PeftModel

@click.command()
@click.option("--weights_dir", default="./ensemble_weights", help="Directory containing the M model weights")
@click.option("--out_dir", default="./sampled_images", help="Base directory for output images")
@click.option("--low_res_folder", default="./eval_dataset/low_res", help="Path to folder containing 64x64 images to upscale")
@click.option("--m_models", default=2, help="Number of ensemble models (M)")
@click.option("--num_inference_steps", default=50, help="Diffusion steps")
@click.option("--sample_base", is_flag=True, help="Whether to sample the base model (MAP estimate) without any adapters")
def sample_ensemble(weights_dir, out_dir, low_res_folder, m_models, num_inference_steps, sample_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base pipeline once
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    pipe = LDMSuperResolutionPipeline.from_pretrained(model_id).to(device)
    pipe.set_progress_bar_config(disable=True) # Cleaner output

    # Fetch images to process
    img_names = [f for f in os.listdir(low_res_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # ==========================================
    # NEW: Sample the Base Model (MAP Estimate)
    # ==========================================
    if sample_base:
        print("\n--- Sampling with Base Model (MAP Estimate) ---")
        base_target_folder = os.path.join(out_dir, "base", "imgs")
        os.makedirs(base_target_folder, exist_ok=True)
        
        for img_name in img_names:
            img_path = os.path.join(low_res_folder, img_name)
            low_res_img = Image.open(img_path).convert("RGB")
            
            # Predict using the unadapted, naked UNet
            upscaled_image = pipe(low_res_img, num_inference_steps=num_inference_steps, eta=1).images[0]
            
            save_path = os.path.join(base_target_folder, img_name)
            upscaled_image.save(save_path)
            print(f"Saved Base MAP: {save_path}")
    # ==========================================

    for m in range(m_models):
        print(f"\n--- Sampling with Ensemble Member {m} ---")
        
        # Target folder: output_dir/m/imgs/
        target_folder = os.path.join(out_dir, str(m), "imgs")
        os.makedirs(target_folder, exist_ok=True)
        
        # Load specific adapter weights
        adapter_path = os.path.join(weights_dir, str(m))
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter {m} not found at {adapter_path}. Skipping.")
            continue
            
        # pipe.unet.load_adapter(adapter_path)
        # Manually wrap the pipeline's UNet with the PEFT weights
        pipe.unet = PeftModel.from_pretrained(pipe.unet, adapter_path)

        # Upscale dataset
        for img_name in img_names:
            img_path = os.path.join(low_res_folder, img_name)
            low_res_img = Image.open(img_path).convert("RGB")
            
            # Predict
            upscaled_image = pipe(low_res_img, num_inference_steps=num_inference_steps, eta=1).images[0]
            
            # Save
            save_path = os.path.join(target_folder, img_name)
            upscaled_image.save(save_path)
            print(f"Saved: {save_path}")

        # Unload adapter to prepare for the next model's weights
        # pipe.unet.unload_adapter()
        # Unload adapter and manually scrub the metadata tag (exactly like training)
        pipe.unet = pipe.unet.unload()
        if hasattr(pipe.unet, 'peft_config'):
            del pipe.unet.peft_config

if __name__ == "__main__":
    sample_ensemble()