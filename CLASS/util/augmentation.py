import torch
import torchvision.transforms as T
from einops import rearrange


# Normalization values for different models, for images in the [0, 255] range.
# The 'default' key will be used for any model name other than 'clip'.
NORMALIZATION_STATS = {
    'default': {
        'mean': [0.485 * 255, 0.456 * 255, 0.406 * 255],
        'std': [0.229 * 255, 0.224 * 255, 0.225 * 255],
    },
    'clip': {
        'mean': [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
        'std': [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    },
}

class AddGaussianNoise(torch.nn.Module):
    """
    Adds Gaussian noise to a tensor. Suitable for tensors in the [0, 255] range.
    """
    def __init__(self, mean=0., std=10.):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        return (img + torch.randn_like(img) * self.std + self.mean).clamp(0, 255)


def permute_images(tensor):
    """Changes tensor layout from HWC to CHW."""
    return rearrange(tensor, "... h w c -> ... c h w")


def unpermute_images(tensor):
    """Changes tensor layout from CHW to HWC."""
    return rearrange(tensor, "... c h w -> ... h w c")


def center_transform(vision_model=None, final_crop_size=(224, 224)):
    """
    Creates a center crop and normalization transform.

    Args:
        vision_model (str, optional): If 'clip', uses CLIP-specific normalization.
                                    Otherwise, uses default values. Defaults to None.
        final_crop_size (tuple): The final size of the cropped image.

    Returns:
        torchvision.transforms.Compose: A composition of transforms.
    """
    norm_key = 'clip' if vision_model and vision_model.lower() == 'clip' else 'default'
    stats = NORMALIZATION_STATS[norm_key]

    return T.Compose(
        [
            T.CenterCrop(size=final_crop_size),
            T.Normalize(mean=stats['mean'], std=stats['std']),
        ]
    )


def crop_transform(vision_model=None, final_crop_size=(224, 224)):
    """
    Creates a random crop, adds Gaussian noise, and normalizes the image.

    Args:
        vision_model (str, optional): If 'clip', uses CLIP-specific normalization.
                                    Otherwise, uses default values. Defaults to None.
        final_crop_size (tuple): The final size of the cropped image.

    Returns:
        torchvision.transforms.Compose: A composition of transforms.
    """
    norm_key = 'clip' if vision_model and vision_model.lower() == 'clip' else 'default'
    stats = NORMALIZATION_STATS[norm_key]

    return T.Compose([
        T.RandomCrop(size=final_crop_size),
        AddGaussianNoise(std=5.),
        T.Normalize(mean=stats['mean'], std=stats['std']),
    ])