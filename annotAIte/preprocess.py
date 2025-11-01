"""
Image preprocessing functions for the annotAIte pipeline.
"""

import numpy as np
import torch
from typing import Union
from scipy import ndimage


def preprocess_image(
    img: Union[np.ndarray, torch.Tensor],
    gaussian_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Convert input image to a torch.Tensor with shape (1, C, H, W), float32 in [0,1],
    apply Gaussian blur to reduce high-frequency noise.
    
    Args:
        img: Input image as HxWxC (numpy uint8 or float/torch) or (C,H,W).
             If single channel grayscale, will be converted to 3-channel RGB.
             TODO: Implement grayscale-to-RGB conversion if needed.
        gaussian_sigma: Standard deviation for Gaussian blur.
        
    Returns:
        tensor_img: torch.Tensor of shape (1, C, H, W), dtype=torch.float32, values in [0,1].
    """
    # Convert to numpy if torch tensor
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()
    else:
        img_np = np.asarray(img)
    
    # Normalize uint8 to [0,1]
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = img_np.astype(np.float32)
    
    # Ensure image is in (H, W, C) format
    if img_np.ndim == 2:  # Grayscale (H, W)
        img_np = np.stack([img_np, img_np, img_np], axis=-1)  # (H, W, 3)
    elif img_np.ndim == 3 and img_np.shape[0] < img_np.shape[2]:  # (C, H, W)
        img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
    
    # Ensure 3 channels
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    elif img_np.shape[2] > 3:
        # Take first 3 channels if more than 3
        img_np = img_np[:, :, :3]
    
    # Apply Gaussian blur
    if gaussian_sigma > 0:
        blurred_channels = []
        for c in range(img_np.shape[2]):
            blurred_channels.append(
                ndimage.gaussian_filter(img_np[:, :, c], sigma=gaussian_sigma)
            )
        img_np = np.stack(blurred_channels, axis=-1)
    
    # Convert to torch tensor and reshape to (1, C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    return img_tensor

