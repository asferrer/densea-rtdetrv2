""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import PIL

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms.v2 import functional as F

from typing import Any, Dict

from ...core import register

@register()
class RandomPhotometricDistort(nn.Module):
    def __init__(self, *args, p=0.5, **kwargs):
        super().__init__()
        self.transform = T.ColorJitter(*args, **kwargs)
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            img = self.transform(img)
        return img, target

@register()
class RandomGaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.2):
        super().__init__()
        self.transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            img = self.transform(img)
        return img, target

@register()
class RandomNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.01, p=0.2):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor) and img.is_floating_point():
                noise = torch.normal(mean=self.mean, std=self.std, size=img.shape, device=img.device)
                img = img + noise
        return img, target

@register()
class RandomRotation(nn.Module):
    def __init__(self, degrees, p=0.5, **kwargs):
        super().__init__()
        self.transform = T.RandomRotation(degrees=degrees, **kwargs)
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            return self.transform(img, target)
        return img, target

@register()
class RandomZoomOut(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = T.RandomZoomOut(*args, **kwargs)
    def forward(self, img, target):
        return self.transform(img, target)

@register()
class RandomIoUCrop(nn.Module):
    def __init__(self, *args, p=0.8, **kwargs):
        super().__init__()
        self.transform = T.RandomIoUCrop(*args, **kwargs)
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            return self.transform(img, target)
        return img, target

@register()
class RandomHorizontalFlip(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = T.RandomHorizontalFlip(*args, **kwargs)
    def forward(self, img, target):
        return self.transform(img, target)

@register()
class Resize(nn.Module):
    def __init__(self, size, **kwargs):
        super().__init__()
        self.transform = T.Resize(size=size, **kwargs)
    def forward(self, img, target):
        return self.transform(img, target)

@register()
class SanitizeBoundingBoxes(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = T.SanitizeBoundingBoxes(*args, **kwargs)
    def forward(self, img, target):
        return self.transform(img, target)

@register()
class ConvertBoxes(nn.Module):
    def __init__(self, fmt='cxcywh', normalize=True):
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def forward(self, img, target):
        # First, make sure the boxes exist and have the right type.
        if 'boxes' in target and isinstance(target['boxes'], BoundingBoxes):
            boxes = target['boxes']
            
            # 1. Convert format
            boxes = F.convert_bounding_box_format(boxes, new_format=self.fmt)
            
            # 2. Normalize manually to avoid the ambiguous call
            if self.normalize:
                # Get the canvas size from the BoundingBoxes object itself
                h, w = boxes.canvas_size
                if self.fmt == 'cxcywh':
                    # Normalize cx, x, w
                    boxes[:, 0] /= w
                    boxes[:, 2] /= w
                    # Normalize cy, y, h
                    boxes[:, 1] /= h
                    boxes[:, 3] /= h
                else: # assumes xyxy
                    boxes[:, 0::2] /= w
                    boxes[:, 1::2] /= h

            target['boxes'] = boxes
        return img, target
# ------------------------------------------

@register()
class ToImage(T.ToImage):
    pass
        
@register()
class ToDtype(T.ToDtype):
    def __init__(self, dtype, scale=False):
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        super().__init__(dtype=torch_dtype, scale=scale)


@register()
class SyntheticDomainAdaptation(nn.Module):
    """
    Transform to close the domain gap between synthetic and real images.

    Supports independent per-channel RGB adjustments based on a domain gap analysis.

    Usage:
        # With parameters from a domain gap analysis:
        - {type: SyntheticDomainAdaptation,
           brightness_factor: [0.80, 0.90],
           red_adjustment: [0.85, 0.95],
           green_adjustment: [0.90, 1.00],
           blue_adjustment: [0.75, 0.85],
           p: 0.7}

        # Modo legacy (backward compatible):
        - {type: SyntheticDomainAdaptation,
           brightness_factor: [0.75, 0.90],
           blue_reduction: [0.75, 0.90],
           green_reduction: [0.85, 0.95],
           p: 0.7}
    """
    def __init__(self,
                 brightness_factor=(0.7, 0.9),      # Global brightness adjustment
                 red_adjustment=None,               # Channel-specific R adjustment
                 green_adjustment=None,             # Channel-specific G adjustment
                 blue_adjustment=None,              # Channel-specific B adjustment
                 # Legacy parameters (backward compatibility)
                 blue_reduction=None,               # Legacy: reduce blue
                 green_reduction=None,              # Legacy: reduce green
                 p=0.7):
        super().__init__()
        self.brightness_factor = brightness_factor

        # If the new parameters (red/green/blue_adjustment) are used, use them
        # Otherwise, use legacy (blue_reduction, green_reduction)
        if red_adjustment is not None:
            self.red_adjustment = red_adjustment
            self.green_adjustment = green_adjustment if green_adjustment is not None else (1.0, 1.0)
            self.blue_adjustment = blue_adjustment if blue_adjustment is not None else (1.0, 1.0)
        else:
            # Legacy mode: brightness + green/blue reduction only
            self.red_adjustment = (1.0, 1.0)  # No adjustment
            self.green_adjustment = green_reduction if green_reduction is not None else (0.85, 0.95)
            self.blue_adjustment = blue_reduction if blue_reduction is not None else (0.7, 0.9)

        self.p = p

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor):
                # Sample random factors for each channel
                b_factor = torch.empty(1).uniform_(*self.brightness_factor).item()
                r_factor = torch.empty(1).uniform_(*self.red_adjustment).item()
                g_factor = torch.empty(1).uniform_(*self.green_adjustment).item()
                blue_factor = torch.empty(1).uniform_(*self.blue_adjustment).item()

                # Apply per-channel adjustments
                if img.dim() == 3:
                    if img.shape[0] == 3:  # CHW (C, H, W)
                        img = img.clone()
                        img[0] = img[0] * b_factor * r_factor      # R: brightness x R adjustment
                        img[1] = img[1] * b_factor * g_factor      # G: brightness x G adjustment
                        img[2] = img[2] * b_factor * blue_factor   # B: brightness x B adjustment
                    elif img.shape[2] == 3:  # HWC (H, W, C)
                        img = img.clone()
                        img[:, :, 0] = img[:, :, 0] * b_factor * r_factor
                        img[:, :, 1] = img[:, :, 1] * b_factor * g_factor
                        img[:, :, 2] = img[:, :, 2] * b_factor * blue_factor

                # Clamp to keep a valid range
                img = torch.clamp(img, 0, 1 if img.max() <= 1 else 255)

        return img, target


@register()
class RandomBrightnessContrast(nn.Module):
    """
    More aggressive brightness and contrast adjustment for domain adaptation.
    """
    def __init__(self, brightness=(0.6, 1.2), contrast=(0.8, 1.2), p=0.5):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor):
                # Brightness adjustment
                b = torch.empty(1).uniform_(*self.brightness).item()
                img = img * b

                # Contrast adjustment
                c = torch.empty(1).uniform_(*self.contrast).item()
                mean = img.mean()
                img = (img - mean) * c + mean

                img = torch.clamp(img, 0, 1 if img.max() <= 1 else 255)

        return img, target