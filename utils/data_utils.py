import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# constannts from config file
from utils.config import DATA_DIR, CINIC_MEAN, CINIC_STD, IMAGE_SIZE, RANDOM_SEED

def get_transforms(augment_type="basic"):
    base_tf = [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=CINIC_MEAN, std=CINIC_STD)
    ]

    # no new images, some of them just get transformed (model only see the transformed versions) 
    if augment_type == "advanced":
        advanced_tf = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomGrayscale(p=0.1),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1)        
            ]
        return T.Compose(advanced_tf + base_tf)
    
    return T.Compose(base_tf)

# specific data loaders for different stages of experiments
def get_dataloaders(config, split='train', num_workers=2):
    aug_type = config.get("augmentations", "basic")
    batch_size = config.get("batch_size", 64)
    subset_ratio = config.get("subset_ratio", 1.0)
    samples_per_class = config.get("samples_per_class", None)
    validation_set_size = config.get("validation_set_size", "full")

    transform = get_transforms(aug_type)
    path = os.path.join(DATA_DIR, split)
    full_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    if split == 'train' and samples_per_class is not None:
        targets = np.array(full_dataset.targets)
        indices = []
        for cls in range(len(full_dataset.classes)):
            cls_indices = np.where(targets == cls)[0]
            n = min(samples_per_class, len(cls_indices))
            chosen = np.random.choice(cls_indices, size=n, replace=False)
            indices.extend(chosen.tolist())
        dataset = Subset(full_dataset, indices)
        print(f"Few-shot mode: {samples_per_class} images per class, {len(indices)} total.")

    elif split == 'train' and subset_ratio < 1.0:
        targets = torch.tensor(full_dataset.targets)
        num_classes = len(torch.unique(targets))
        num_samples_per_class = int(len(full_dataset) * subset_ratio / num_classes)
        indices = []
        for class_idx in range(num_classes):
            class_indices = torch.where(targets == class_idx)[0]
            perm = torch.randperm(len(class_indices))[:num_samples_per_class]
            indices.append(class_indices[perm])
        indices = torch.cat(indices)
        dataset = Subset(full_dataset, indices)
        print(f"Few-shot mode (stratified): {len(indices)} images ({subset_ratio*100}% of training set), {num_samples_per_class} per class.")

    elif split == 'test' and validation_set_size == "reduced":
        np.random.seed(RANDOM_SEED)
        targets = np.array(full_dataset.targets)
        indices = []
        for cls in range(len(full_dataset.classes)):
            cls_indices = np.where(targets == cls)[0]
            chosen = np.random.choice(cls_indices, size=min(1000, len(cls_indices)), replace=False)
            indices.extend(chosen.tolist())
        dataset = Subset(full_dataset, indices)
        print(f"Reduced validation set: 1000 images per class, {len(indices)} total.")
        loader = DataLoader(
        dataset,
        batch_size=batch_size if split == 'train' else 256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2

        )
        return loader

    else:
        dataset = full_dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size if split == 'train' else min(batch_size * 2, 256),
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return loader

def apply_cutmix(inputs, targets, alpha=1.0):
    if inputs.size(0) <= 1:
        return inputs, targets, targets, 1.0

    indices = torch.randperm(inputs.size(0))
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    
    W, H = inputs.size()[2], inputs.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[indices, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return inputs, targets, shuffled_targets, lam