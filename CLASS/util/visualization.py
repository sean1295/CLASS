import numpy as np
import torchvision.utils as vutils
import torch

def create_image_grid(images, n, completion_status=None, border_width=4):
    """
    Create image grid using torchvision.utils.make_grid with colored borders
    
    Args:
        images: numpy array of shape (N, H, W, C) or (N, C, H, W), or torch tensor
        n: grid size (n x n)
        completion_status: list/array of boolean values indicating completion status
                          True = completed (green border), False = in progress (red border)
                          If None, no borders are added
        border_width: width of the border in pixels
    
    Returns:
        numpy array representing the image grid with colored borders
    """
    import torch
    import torchvision.utils as vutils
    import numpy as np
    
    # Ensure we have enough images
    num_images = n * n
    if len(images) < num_images:
        raise ValueError(
            f"Not enough images. Need {num_images} images but only got {len(images)}"
        )
    
    images = images[:num_images]
    
    if isinstance(images, np.ndarray):
        if images.shape[-1] == 3:
            images = images.transpose(0, 3, 1, 2)
        
        images = torch.from_numpy(images).float()
        if images.max() > 1.0:
            images = images / 255.0
    
    if completion_status is not None:
        completion_status = completion_status[:num_images]  # Take only required number
        images_with_borders = []
        
        for i, img in enumerate(images):
            img_with_border = img.clone()
            
            if completion_status[i]:
                border_color = [0.0, 1.0, 0.0]  # RGB in [0, 1] range
            else:
                border_color = [1.0, 0.0, 0.0]  # RGB in [0, 1] range
            
            # Add border to each channel
            for c in range(3):  # Assuming RGB images
                img_with_border[c, :border_width, :] = border_color[c]
                img_with_border[c, -border_width:, :] = border_color[c]
                img_with_border[c, :, :border_width] = border_color[c]
                img_with_border[c, :, -border_width:] = border_color[c]
            
            images_with_borders.append(img_with_border)
        
        images = torch.stack(images_with_borders)
    
    grid = vutils.make_grid(
        images, 
        nrow=n,  # number of images per row
        padding=0,  # small padding between images to separate borders
        normalize=False,  # don't normalize since we handle it above
        value_range=None  # use the full range of input values
    ).numpy()
    
    # Convert back to uint8 format
    grid = (grid * 255).astype(np.uint8)
    
    return grid