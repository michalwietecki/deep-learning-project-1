import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# constannts from config file
from utils.config import DATA_DIR, CINIC_MEAN, CINIC_STD, IMAGE_SIZE

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
def get_dataloaders(config, split='train', num_workers=4):
    aug_type = config.get("augmentations", "basic")
    batch_size = config.get("batch_size", 64)
    subset_ratio = config.get("subset_ratio", 1.0)
    
    transform = get_transforms(aug_type)
    
    path = os.path.join(DATA_DIR, split)
    full_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    
    # few-shot mechanism
    if split == 'train' and subset_ratio < 1.0:
        num_samples = int(len(full_dataset) * subset_ratio)
        
        # random indices
        indices = torch.randperm(len(full_dataset))[:num_samples]
        dataset = Subset(full_dataset, indices)
        print(f"Few-shot mode: Use {num_samples} images ({subset_ratio*100}% of all available training examples).")
    else:
        dataset = full_dataset

    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=False # for gpu optimization
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