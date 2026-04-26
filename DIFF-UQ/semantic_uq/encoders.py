import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import numpy as np

try:
    import clip
except ImportError:
    clip = None

try:
    import open_clip
except ImportError:
    open_clip = None


def vgg16_preprocess(image: Image.Image) -> torch.Tensor:
    # We keep the pure 128x128 resolution
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    # Standard ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image - mean) / std


def dinov2_preprocess(image: Image.Image) -> torch.Tensor:
    # Resize directly to the ViT standard without cropping away the edges
    image = image.resize((224, 224), resample=Image.BICUBIC)

    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    return image


def extract_clip_params(preprocess, device):
    """
    Extracts size, mean, and std from a transforms.Compose object 
    and returns a pure-GPU batched preprocessing function.
    """
    size = 224
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    if hasattr(preprocess, 'transforms'):
        for t in preprocess.transforms:
            if hasattr(t, 'size'):
                size = t.size if isinstance(t.size, int) else t.size[0]
            if hasattr(t, 'mean'): mean = t.mean
            if hasattr(t, 'std'): std = t.std
            
    def preprocess_gpu(image_batch: torch.Tensor) -> torch.Tensor:
        # image_batch must be a (B, C, H, W) tensor in [0.0, 1.0]
        x = F.interpolate(image_batch, size=(size, size), mode='bicubic', align_corners=False)
        mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
        return (x - mean_t) / std_t
        
    return preprocess_gpu


def build_encoder(encoder_name: str, device: torch.device):
    if encoder_name == "clip":
        if clip is None:
            raise ImportError("Please install `clip` to use the CLIP encoder.")
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            return model.encode_image(image_batch)

        preprocess_gpu = extract_clip_params(preprocess, device)
        return preprocess, encode_batch, preprocess_gpu
    
    # SigLIP (ViT-B/16 tuned on WebLI)
    if encoder_name == "siglip":
        if open_clip is None:
            raise ImportError("Please install `open_clip_torch` first.")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
        model.eval().to(device)

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            features = model.encode_image(image_batch)
            features = F.normalize(features, p=2, dim=1)
            return features

        preprocess_gpu = extract_clip_params(preprocess, device)
        return preprocess, encode_batch, preprocess_gpu

    # OpenCLIP ViT-H/14
    if encoder_name == "openclip_h14":
        if open_clip is None:
            raise ImportError("Please install `open_clip_torch` first.")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        model.eval().to(device)

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            features = model.encode_image(image_batch)
            features = F.normalize(features, p=2, dim=1)
            return features

        preprocess_gpu = extract_clip_params(preprocess, device)
        return preprocess, encode_batch, preprocess_gpu

    if encoder_name == "dinov2_vits14_reg":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        model.eval().to(device)
        preprocess = dinov2_preprocess

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            features = model.forward_features(image_batch)
            return features["x_norm_clstoken"]
            
        def preprocess_gpu(image_batch: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(image_batch, size=(224, 224), mode='bicubic', align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            return (x - mean) / std

        return preprocess, encode_batch, preprocess_gpu
    
    if encoder_name == "vgg16":
        full_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval().to(device)
        feature_extractor = full_vgg.features
        avgpool = full_vgg.avgpool
        classifier = full_vgg.classifier[:5] 
        preprocess = vgg16_preprocess

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            x = feature_extractor(image_batch)
            x = avgpool(x)
            x = torch.flatten(x, 1)
            features = classifier(x)
            features = F.normalize(features, p=2, dim=1)
            return features

        def preprocess_gpu(image_batch: torch.Tensor) -> torch.Tensor:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            return (image_batch - mean) / std

        return preprocess, encode_batch, preprocess_gpu

    raise ValueError(f"Unknown encoder: {encoder_name}")
