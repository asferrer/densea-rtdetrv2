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
        # Primero, nos aseguramos de que las cajas existan y sean del tipo correcto.
        if 'boxes' in target and isinstance(target['boxes'], BoundingBoxes):
            boxes = target['boxes']
            
            # 1. Convertir formato
            boxes = F.convert_bounding_box_format(boxes, new_format=self.fmt)
            
            # 2. Normalizar manualmente para evitar la llamada ambigua
            if self.normalize:
                # Obtenemos el tamaño del lienzo desde el propio objeto BoundingBoxes
                h, w = boxes.canvas_size
                if self.fmt == 'cxcywh':
                    # Normalizar cx, x, w
                    boxes[:, 0] /= w
                    boxes[:, 2] /= w
                    # Normalizar cy, y, h
                    boxes[:, 1] /= h
                    boxes[:, 3] /= h
                else: # asume xyxy
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
    Transform específico para cerrar el domain gap entre imágenes sintéticas y reales.

    NUEVO: Soporta ajustes independientes por canal RGB basados en análisis de domain gap.

    Uso:
        # Con parámetros de domain_gap_analysis.json:
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
                 brightness_factor=(0.7, 0.9),      # Ajuste global de brillo
                 red_adjustment=None,               # NUEVO: Ajuste específico canal R
                 green_adjustment=None,             # NUEVO: Ajuste específico canal G
                 blue_adjustment=None,              # NUEVO: Ajuste específico canal B
                 # Legacy parameters (backward compatibility)
                 blue_reduction=None,               # Legacy: reducir azul
                 green_reduction=None,              # Legacy: reducir verde
                 p=0.7):
        super().__init__()
        self.brightness_factor = brightness_factor

        # Si se usan parámetros nuevos (red/green/blue_adjustment), usarlos
        # Si no, usar legacy (blue_reduction, green_reduction)
        if red_adjustment is not None:
            self.red_adjustment = red_adjustment
            self.green_adjustment = green_adjustment if green_adjustment is not None else (1.0, 1.0)
            self.blue_adjustment = blue_adjustment if blue_adjustment is not None else (1.0, 1.0)
        else:
            # Modo legacy: solo brillo + reducción de green/blue
            self.red_adjustment = (1.0, 1.0)  # Sin ajuste
            self.green_adjustment = green_reduction if green_reduction is not None else (0.85, 0.95)
            self.blue_adjustment = blue_reduction if blue_reduction is not None else (0.7, 0.9)

        self.p = p

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor):
                # Generar factores aleatorios para cada canal
                b_factor = torch.empty(1).uniform_(*self.brightness_factor).item()
                r_factor = torch.empty(1).uniform_(*self.red_adjustment).item()
                g_factor = torch.empty(1).uniform_(*self.green_adjustment).item()
                blue_factor = torch.empty(1).uniform_(*self.blue_adjustment).item()

                # Aplicar ajustes por canal
                if img.dim() == 3:
                    if img.shape[0] == 3:  # CHW (C, H, W)
                        img = img.clone()
                        img[0] = img[0] * b_factor * r_factor      # R: brillo × ajuste_R
                        img[1] = img[1] * b_factor * g_factor      # G: brillo × ajuste_G
                        img[2] = img[2] * b_factor * blue_factor   # B: brillo × ajuste_B
                    elif img.shape[2] == 3:  # HWC (H, W, C)
                        img = img.clone()
                        img[:, :, 0] = img[:, :, 0] * b_factor * r_factor
                        img[:, :, 1] = img[:, :, 1] * b_factor * g_factor
                        img[:, :, 2] = img[:, :, 2] * b_factor * blue_factor

                # Clamp para mantener rango válido
                img = torch.clamp(img, 0, 1 if img.max() <= 1 else 255)

        return img, target


@register()
class RandomBrightnessContrast(nn.Module):
    """
    Ajuste más agresivo de brillo y contraste para domain adaptation.
    """
    def __init__(self, brightness=(0.6, 1.2), contrast=(0.8, 1.2), p=0.5):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor):
                # Ajuste de brillo
                b = torch.empty(1).uniform_(*self.brightness).item()
                img = img * b

                # Ajuste de contraste
                c = torch.empty(1).uniform_(*self.contrast).item()
                mean = img.mean()
                img = (img - mean) * c + mean

                img = torch.clamp(img, 0, 1 if img.max() <= 1 else 255)

        return img, target