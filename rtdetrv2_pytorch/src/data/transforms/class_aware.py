"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Class-aware augmentations for handling class imbalance in object detection.
These transforms apply additional augmentations when minority classes are present.
"""

import torch
import torch.nn as nn
import random
import numpy as np

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F

from ...core import register

__all__ = [
    'ClassAwarePhotometricDistort',
    'ClassAwareMixUp',
    'ClassAwareMultiScale',
    'ClassAwareCopyPaste'
]


@register()
class ClassAwarePhotometricDistort(nn.Module):
    """
    Applies stronger photometric augmentations when target classes are present.

    When an image contains objects from specified target classes (e.g., Can, Metal_Debris),
    this transform applies more aggressive color/brightness augmentations to increase
    the variety of training samples for these minority classes.

    Args:
        target_classes: List of class IDs to trigger extra augmentation
        base_p: Base probability of applying augmentation (default: 0.5)
        extra_p: Extra probability added when target classes present (default: 0.3)
        brightness: Brightness jitter range (default: 0.3)
        contrast: Contrast jitter range (default: 0.3)
        saturation: Saturation jitter range (default: 0.3)
        hue: Hue jitter range (default: 0.1)
    """

    def __init__(
        self,
        target_classes=None,
        base_p=0.5,
        extra_p=0.3,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ):
        super().__init__()
        self.target_classes = set(target_classes) if target_classes else set()
        self.base_p = base_p
        self.extra_p = extra_p

        # Normal augmentation
        self.normal_transform = T.ColorJitter(
            brightness=brightness * 0.5,
            contrast=contrast * 0.5,
            saturation=saturation * 0.5,
            hue=hue * 0.5
        )

        # Stronger augmentation for target classes
        self.strong_transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def _has_target_class(self, target):
        """Check if target contains any of the specified classes."""
        if 'labels' not in target:
            return False
        labels = target['labels']
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        return bool(set(labels) & self.target_classes)

    def forward(self, img, target, dataset=None):
        has_target = self._has_target_class(target)

        # Determine probability based on whether target classes are present
        p = self.base_p + self.extra_p if has_target else self.base_p

        if torch.rand(1) < p:
            transform = self.strong_transform if has_target else self.normal_transform
            img = transform(img)

        return img, target, dataset


@register()
class ClassAwareMultiScale(nn.Module):
    """
    Applies multi-scale augmentation with extra scales for images containing target classes.

    This helps the model learn to detect minority class objects at various scales.

    Args:
        target_classes: List of class IDs to trigger extra scales
        base_scales: List of scale factors for normal images (default: [0.8, 1.0, 1.2])
        extra_scales: Additional scales when target classes present (default: [0.6, 0.7, 1.3, 1.4])
        p: Probability of applying scale augmentation (default: 0.5)
    """

    def __init__(
        self,
        target_classes=None,
        base_scales=None,
        extra_scales=None,
        p=0.5
    ):
        super().__init__()
        self.target_classes = set(target_classes) if target_classes else set()
        self.base_scales = base_scales or [0.8, 1.0, 1.2]
        self.extra_scales = extra_scales or [0.6, 0.7, 1.3, 1.4]
        self.p = p

    def _has_target_class(self, target):
        if 'labels' not in target:
            return False
        labels = target['labels']
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        return bool(set(labels) & self.target_classes)

    def forward(self, img, target, dataset=None):
        if torch.rand(1) > self.p:
            return img, target, dataset

        has_target = self._has_target_class(target)
        scales = self.base_scales + self.extra_scales if has_target else self.base_scales
        scale = random.choice(scales)

        if scale != 1.0:
            # Get original size
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size

            new_h = int(h * scale)
            new_w = int(w * scale)

            # Resize image and boxes
            img = F.resize(img, [new_h, new_w])

            if 'boxes' in target:
                boxes = target['boxes']
                boxes = boxes * scale
                target['boxes'] = boxes

        return img, target, dataset


@register()
class ClassAwareMixUp(nn.Module):
    """
    MixUp augmentation that prioritizes mixing with images containing target classes.

    When an image without target classes is processed, this transform has a higher
    chance of mixing it with an image that contains target classes, effectively
    increasing exposure to minority class features.

    Args:
        target_classes: List of class IDs to prioritize in mixing
        alpha: Beta distribution parameter for mixing ratio (default: 0.5)
        p: Probability of applying mixup (default: 0.3)
    """

    def __init__(
        self,
        target_classes=None,
        alpha=0.5,
        p=0.3
    ):
        super().__init__()
        self.target_classes = set(target_classes) if target_classes else set()
        self.alpha = alpha
        self.p = p
        self._target_class_indices = None
        self._all_indices = None

    def _build_target_indices(self, dataset):
        """Build index of images containing target classes."""
        if self._target_class_indices is not None:
            return

        coco = dataset.coco
        cat_ids = set()

        # Map class labels to category IDs if needed
        for cat in coco.dataset['categories']:
            if cat['id'] in self.target_classes:
                cat_ids.add(cat['id'])

        # Find images with target classes
        img_ids_with_target = set()
        for ann in coco.dataset['annotations']:
            if ann['category_id'] in cat_ids:
                img_ids_with_target.add(ann['image_id'])

        # Convert to dataset indices
        self._target_class_indices = []
        self._all_indices = list(range(len(dataset)))

        for idx, img_id in enumerate(dataset.ids):
            if img_id in img_ids_with_target:
                self._target_class_indices.append(idx)

        print(f"[ClassAwareMixUp] Found {len(self._target_class_indices)} images with target classes")

    def forward(self, img, target, dataset=None):
        if dataset is None or torch.rand(1) > self.p:
            return img, target, dataset

        self._build_target_indices(dataset)

        if not self._target_class_indices:
            return img, target, dataset

        # Prefer mixing with target class images
        if torch.rand(1) < 0.7:  # 70% chance to mix with target class image
            mix_idx = random.choice(self._target_class_indices)
        else:
            mix_idx = random.choice(self._all_indices)

        # Load mix image
        mix_img, mix_target = dataset.load_item(mix_idx)
        if mix_target is None:
            return img, target, dataset

        # Convert to tensor if needed
        if not isinstance(mix_img, torch.Tensor):
            mix_img = F.to_image(mix_img)
            mix_img = F.to_dtype(mix_img, torch.float32, scale=True)

        if not isinstance(img, torch.Tensor):
            img = F.to_image(img)
            img = F.to_dtype(img, torch.float32, scale=True)

        # Resize mix image to match
        mix_img = F.resize(mix_img, img.shape[-2:])

        # Compute mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure main image dominates

        # Mix images
        img = lam * img + (1 - lam) * mix_img

        # Combine targets (keep both sets of boxes/labels)
        if 'boxes' in target and 'boxes' in mix_target:
            # Scale mix boxes if images were different sizes
            combined_boxes = torch.cat([target['boxes'], mix_target['boxes']], dim=0)
            combined_labels = torch.cat([target['labels'], mix_target['labels']], dim=0)
            target['boxes'] = combined_boxes
            target['labels'] = combined_labels

        return img, target, dataset


@register()
class ClassAwareCopyPaste(nn.Module):
    """
    Copy-Paste augmentation that copies objects from target classes onto other images.

    This is one of the most effective augmentations for minority classes as it directly
    increases the number of training samples with target class objects.

    Args:
        target_classes: List of class IDs to copy-paste
        p: Probability of applying copy-paste (default: 0.3)
        max_paste: Maximum number of objects to paste (default: 3)
    """

    def __init__(
        self,
        target_classes=None,
        p=0.3,
        max_paste=3
    ):
        super().__init__()
        self.target_classes = set(target_classes) if target_classes else set()
        self.p = p
        self.max_paste = max_paste
        self._target_objects = None

    def _build_object_bank(self, dataset):
        """Build bank of target class objects for pasting."""
        if self._target_objects is not None:
            return

        self._target_objects = []
        coco = dataset.coco

        # Find annotations for target classes
        for ann in coco.dataset['annotations']:
            if ann['category_id'] in self.target_classes:
                self._target_objects.append({
                    'image_id': ann['image_id'],
                    'bbox': ann['bbox'],  # [x, y, w, h]
                    'category_id': ann['category_id']
                })

        print(f"[ClassAwareCopyPaste] Built bank with {len(self._target_objects)} target class objects")

    def forward(self, img, target, dataset=None):
        # This is a placeholder - full implementation requires mask support
        # For bounding box detection, we simply return the original
        # A full implementation would:
        # 1. Extract object regions from source images
        # 2. Paste them onto the current image
        # 3. Update bounding boxes and labels

        return img, target, dataset
