import random
import numpy as np
import torch
import os

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

    print(f"Full reproducibility with seed set at: {seed}")

# global constants to import
#DATA_DIR = './data/'
DATA_DIR = '/kaggle/input/datasets/mengcius/cinic10' #TEMPORAL CHANGE TO KAGGLE
TRAINED_MODELS_DIR = './trained_models/'
IMAGE_SIZE = 224
NUM_CLASSES = 10
RANDOM_SEED = 420

# CINIC-10 dataset statistics, for nromalization
CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]

# experiments configuration
#models, epochs, lr, batch_size, optimizer, augmentation, scheduler, weight_decay, use_cutmix
EXPERIMENTS = {
    # stage 1: 3 models, for future comparison
    "stage_1": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "augmentations": "none"
    },
        "stage_2_test_num_workers_4": {
        "models": ["baseline_cnn", "efficientnet_b0"],
        "epochs": 1,
        "lr": 0.001,
        "batch_size": 64,
        "optimizer": "ADAM",
        "augmentations": "none",
        "scheduler" : "StepLR"
    },
        "stage_2_test_num_workers_4_batch_128": {
        "models": ["baseline_cnn",],
        "epochs": 2,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "augmentations": "none",
        "scheduler" : "StepLR"
    },
    "stage_2_stepLR": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "augmentations": "none",
        "scheduler" : "StepLR"
    },
    "stage_2_MultiStepLR": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "augmentations": "none",
        "scheduler" : "MultiStepLR"
    },
    "stage_2_CosineAnnealingLR": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "augmentations": "none",
        "scheduler" : "CosineAnnealingLR"
    },
    "stage_2_ReduceLROnPlateau": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "augmentations": "none",
        "scheduler" : "ReduceLROnPlateau"
    },
    "stage_2_LR_0.003": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.003,
        "batch_size": 128,
        "optimizer": "ADAM",
        "augmentations": "none",
    },
        "stage_2_LR_0.0003": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.0003,
        "batch_size": 128,
        "optimizer": "ADAM",
        "augmentations": "none",
    },
        "stage_3_dropout": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay" : 0.0,
        "dropout": 0.3,
    },
        "stage_3_wd": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 1e-4,
        "dropout": 0.0,
    },
        "stage_3_mix_soft": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 1e-4,
        "dropout": 0.3,
    },
        "stage_3_mix_hard": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 5e-4,
        "dropout": 0.5,
    },

}
