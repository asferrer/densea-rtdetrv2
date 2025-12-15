"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Class-balanced sampling for handling class imbalance in object detection.
"""

import torch
from torch.utils.data import Sampler
from collections import Counter, defaultdict
import numpy as np
import random

from ..core import register

__all__ = ['ClassBalancedSampler']


@register()
class ClassBalancedSampler(Sampler):
    """
    Sampler that oversamples images containing minority classes to balance class representation.

    This sampler increases the frequency of images containing specified target classes,
    which helps improve detection of underrepresented classes.

    Args:
        dataset: Dataset with COCO-style annotations (injected by DataLoader)
        target_classes: List of class names to oversample (e.g., ['Can', 'Metal_Debris'])
        oversample_factor: Factor by which to oversample images with target classes (default: 3.0)
        shuffle: Whether to shuffle the indices (default: True)
        seed: Random seed for reproducibility (default: None)

    Example YAML configuration:
        train_dataloader:
            type: DataLoader
            dataset: ...
            sampler:
                type: ClassBalancedSampler
                target_classes: ['Can', 'Metal_Debris']
                oversample_factor: 3.0
    """

    # No usamos __inject__ porque el dataset se pasa desde el DataLoader
    # __inject__ = ['dataset']

    def __init__(
        self,
        dataset=None,
        target_classes=None,
        oversample_factor=3.0,
        shuffle=True,
        seed=None
    ):
        self._dataset = dataset
        self.target_classes = target_classes or []
        self.oversample_factor = oversample_factor
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._initialized = False
        self.indices = []
        self.target_images = []
        self.normal_images = []

        # Si el dataset ya está disponible, inicializar inmediatamente
        if dataset is not None:
            self._build_index()
            self._initialized = True

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        """Setter para cuando el dataset se asigna después de la creación."""
        self._dataset = value
        if value is not None and not self._initialized:
            self._build_index()
            self._initialized = True

    def _build_index(self):
        """Build mapping from class to image indices and compute sampling weights."""
        coco = self.dataset.coco

        # Get category name to id mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in coco.dataset['categories']}

        # Get target class IDs
        target_class_ids = set()
        for class_name in self.target_classes:
            if class_name in cat_name_to_id:
                target_class_ids.add(cat_name_to_id[class_name])
            else:
                print(f"[ClassBalancedSampler] Warning: class '{class_name}' not found in dataset")

        # Build image-to-classes mapping
        img_to_classes = defaultdict(set)
        for ann in coco.dataset['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            img_to_classes[img_id].add(cat_id)

        # Count images containing target classes
        self.target_images = []  # Indices of images with target classes
        self.normal_images = []  # Indices of images without target classes

        for idx, img_id in enumerate(self.dataset.ids):
            classes_in_image = img_to_classes.get(img_id, set())
            if classes_in_image & target_class_ids:  # Intersection
                self.target_images.append(idx)
            else:
                self.normal_images.append(idx)

        # Calculate class statistics
        class_counts = Counter()
        for ann in coco.dataset['annotations']:
            class_counts[ann['category_id']] += 1

        # Print summary
        print(f"\n[ClassBalancedSampler] Inicializado:")
        print(f"  - Total de imágenes: {len(self.dataset)}")
        print(f"  - Imágenes con clases objetivo: {len(self.target_images)}")
        print(f"  - Imágenes sin clases objetivo: {len(self.normal_images)}")
        print(f"  - Factor de sobremuestreo: {self.oversample_factor}x")

        if self.target_classes:
            print(f"  - Clases objetivo: {self.target_classes}")
            for class_name in self.target_classes:
                if class_name in cat_name_to_id:
                    cat_id = cat_name_to_id[class_name]
                    count = class_counts.get(cat_id, 0)
                    print(f"    - {class_name}: {count} anotaciones")

        # Build final indices list with oversampling
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Rebuild the sampling indices with oversampling."""
        # Start with all normal images
        self.indices = list(self.normal_images)

        # Add oversampled target images
        oversample_count = int(len(self.target_images) * self.oversample_factor)
        if self.target_images:
            # Repeat target images to achieve desired oversample factor
            repeated_targets = self.target_images * int(self.oversample_factor)
            # Add any remaining to reach exact count
            remaining = oversample_count - len(repeated_targets)
            if remaining > 0:
                repeated_targets.extend(random.sample(self.target_images, min(remaining, len(self.target_images))))
            self.indices.extend(repeated_targets)

        print(f"[ClassBalancedSampler] Total índices de entrenamiento: {len(self.indices)} "
              f"(original: {len(self.dataset)}, sobremuestreado: +{len(self.indices) - len(self.dataset)})")

    def _ensure_initialized(self):
        """Ensure the sampler is initialized before use."""
        if not self._initialized and self._dataset is not None:
            self._build_index()
            self._initialized = True

    def set_epoch(self, epoch):
        """Set epoch for reproducible shuffling."""
        self._epoch = epoch

    def __iter__(self):
        """Iterate over sampled indices."""
        self._ensure_initialized()
        indices = self.indices.copy()

        if self.shuffle:
            # Use epoch-based seed for reproducibility
            if self.seed is not None:
                g = torch.Generator()
                g.manual_seed(self.seed + self._epoch)
                perm = torch.randperm(len(indices), generator=g).tolist()
            else:
                perm = torch.randperm(len(indices)).tolist()
            indices = [indices[i] for i in perm]

        return iter(indices)

    def __len__(self):
        """Return total number of samples (including oversampled)."""
        self._ensure_initialized()
        return len(self.indices)


@register()
class ClassAwareRandomSampler(Sampler):
    """
    Alternative sampler that samples images based on class frequency weights.

    Instead of duplicating images, this sampler uses weighted random sampling
    where images with rare classes have higher probability of being selected.

    Args:
        dataset: Dataset with COCO-style annotations (injected by DataLoader)
        target_classes: Dict mapping class names to weight multipliers (e.g., {'Can': 5.0, 'Metal_Debris': 5.0})
        num_samples: Number of samples per epoch (default: same as dataset length)
        replacement: Whether to sample with replacement (default: True)
    """

    # No usamos __inject__ porque el dataset se pasa desde el DataLoader
    # __inject__ = ['dataset']

    def __init__(
        self,
        dataset=None,
        target_classes=None,
        num_samples=None,
        replacement=True
    ):
        self._dataset = dataset
        self.target_classes = target_classes or {}
        self._num_samples = num_samples
        self.replacement = replacement
        self._epoch = 0
        self._initialized = False
        self.weights = None

        # Si el dataset ya está disponible, inicializar inmediatamente
        if dataset is not None:
            self._compute_weights()
            self._initialized = True

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        """Setter para cuando el dataset se asigna después de la creación."""
        self._dataset = value
        if value is not None and not self._initialized:
            self._compute_weights()
            self._initialized = True

    @property
    def num_samples(self):
        if self._num_samples is not None:
            return self._num_samples
        if self._dataset is not None:
            return len(self._dataset)
        return 0

    def _ensure_initialized(self):
        """Ensure the sampler is initialized before use."""
        if not self._initialized and self._dataset is not None:
            self._compute_weights()
            self._initialized = True

    def _compute_weights(self):
        """Compute sampling weights for each image."""
        coco = self.dataset.coco

        # Get category name to id mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in coco.dataset['categories']}

        # Convert target_classes to use category IDs
        target_weights = {}
        for class_name, weight in self.target_classes.items():
            if class_name in cat_name_to_id:
                target_weights[cat_name_to_id[class_name]] = weight

        # Build image-to-classes mapping
        img_to_classes = defaultdict(set)
        for ann in coco.dataset['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            img_to_classes[img_id].add(cat_id)

        # Compute weight for each image
        self.weights = []
        for idx, img_id in enumerate(self.dataset.ids):
            classes_in_image = img_to_classes.get(img_id, set())

            # Start with base weight of 1.0
            weight = 1.0

            # Add bonus weight for each target class present
            for cat_id in classes_in_image:
                if cat_id in target_weights:
                    weight = max(weight, target_weights[cat_id])

            self.weights.append(weight)

        self.weights = torch.tensor(self.weights, dtype=torch.float64)

        print(f"\n[ClassAwareRandomSampler] Inicializado:")
        print(f"  - Imágenes con peso > 1.0: {(self.weights > 1.0).sum().item()}")
        print(f"  - Peso máximo: {self.weights.max().item():.2f}")
        print(f"  - Peso medio: {self.weights.mean().item():.2f}")

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        self._ensure_initialized()
        g = torch.Generator()
        g.manual_seed(self._epoch)

        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=g
        ).tolist()

        return iter(indices)

    def __len__(self):
        self._ensure_initialized()
        return self.num_samples
