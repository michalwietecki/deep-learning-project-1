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
        "dropout": 0.2,
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
        "dropout": 0.2,
    },
        "stage_3_mix_hard": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 5e-4,
        "dropout": 0.3,
    },
        "stage_4_mix_hard_aug": {
        "models": ["baseline_cnn", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "augmentations": "advanced",
        "use_cutmix": False
    },
    "stage_4_mix_hard_cut": {
        "models": ["baseline_cnn", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "augmentations": "basic",
        "use_cutmix": True
    },
    "stage_4_mix_hard_cut_efficientnet_only": {
        "models": ["efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "augmentations": "basic",
        "use_cutmix": True
    },
    "stage_4_mix_hard_aug_cut": {
        "models": ["baseline_cnn", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "augmentations": "advanced",
        "use_cutmix": True
    },
        "stage_4_mix_hard_aug_cut_grad_accumulation": {
        "models": ["baseline_cnn", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 32,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "augmentations": "advanced",
        "use_cutmix": True,
        "grad_accumulation_steps": 4
    },
    "stage_4_mix_soft_aug": {
        "models": ["efficientnet_b0"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "augmentations": "advanced",
        "use_cutmix": False
    },
    "stage_4_mix_soft_cut": {
        "models": ["efficientnet_b0"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "augmentations": "basic",
        "use_cutmix": True
    },
       "stage_4_mix_soft_aug_cut": {
        "models": ["efficientnet_b0"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "ADAM",
        "scheduler" : "CosineAnnealingLR",
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "augmentations": "advanced",
        "use_cutmix": True
    },



    # ========= stage 5


    # Stage FS-1: baseline few-shot - jak bardzo spada jakość przy małych danych?
    # Porównanie wszystkich 3 modeli przy tym samym ograniczeniu
    "stage_5_1__10_percent": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 64,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "none",
        "subset_ratio": 0.1,
        "validation_set_size": "reduced"
    },
    "stage_5_1__1_percent": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 32,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "none",
        "subset_ratio": 0.01,
        "validation_set_size": "reduced"
    },
    "stage_5_1__50_per_class": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 16,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "none",
        "samples_per_class": 50,
        "validation_set_size": "reduced"
    },
    "stage_5_1__20_per_class": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 8,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "none",
        "samples_per_class": 20,
        "validation_set_size": "reduced"
    },

    # Stage FS-2: wpływ augmentacji przy małych danych
    # Hipoteza: augmentacje powinny pomagać bardziej im mniej danych
        "stage_5_2__10_percent_aug": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 64,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "advanced",
        "subset_ratio": 0.1,
        "validation_set_size": "reduced"
    },
    "stage_5_2__1_percent_aug": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 32,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "advanced",
        "subset_ratio": 0.01,
        "validation_set_size": "reduced"
    },
    "stage_5_2__50_per_class_aug": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 16,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "advanced",
        "samples_per_class": 50,
        "validation_set_size": "reduced"
    },
    "stage_5_2__20_per_class_aug": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 8,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "advanced",
        "samples_per_class": 20,
        "validation_set_size": "reduced"
    },

    # Stage FS-3: scheduler + regularyzacja przy małych danych
    # Przy few-shot overfitting jest głównym problemem
    "stage_5_3__10_percent_reg": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 64,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "basic",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "subset_ratio": 0.3,
        "validation_set_size": "reduced"
    },
    "stage_5_3__10_percent_reg_true": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 64,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "basic",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "subset_ratio": 0.1,
        "validation_set_size": "reduced"
    },
    "stage_5_3__1_percent_reg": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 32,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "basic",
        "subset_ratio": 0.01,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "validation_set_size": "reduced"
    },
    "stage_5_3__50_per_class_reg": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 16,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "basic",
        "samples_per_class": 50,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "validation_set_size": "reduced"
    },
    "stage_5_3__20_per_class_reg": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.001,
        "batch_size": 8,
        "scheduler" : "ReduceLROnPlateau",
        "optimizer": "ADAM",
        "augmentations": "basic",
        "samples_per_class": 20,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "validation_set_size": "reduced"
    },

    # Stage FS-4: lr przy małych danych - mniejszy lr może być stabilniejszy
    "stage_5_4__pre_10_percent_0.0003": {
        "models": ["efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.0003,
        "batch_size": 64,
        "optimizer": "ADAM",
        "augmentations": "advanced",
        "scheduler": "ReduceLROnPlateau",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "subset_ratio": 0.1,
        "validation_set_size": "reduced"
    },
    "stage_5_4__pre_1_percent_0.0003": {
        "models": ["efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.0003,
        "batch_size": 64,
        "optimizer": "ADAM",
        "augmentations": "advanced",
        "scheduler": "ReduceLROnPlateau",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "subset_ratio": 0.01,
        "validation_set_size": "reduced"
    },
    "stage_5_4__pre_50_per_class_0.0003": {
        "models": ["efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.0003,
        "batch_size": 64,
        "optimizer": "ADAM",
        "augmentations": "advanced",
        "scheduler": "ReduceLROnPlateau",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "samples_per_class": 50,
        "validation_set_size": "reduced"
    },
        "stage_5_4__pre_20_per_class_0.0003": {
        "models": ["efficientnet_b0_pretrained"],
        "epochs": 30,
        "lr": 0.0003,
        "batch_size": 64,
        "optimizer": "ADAM",
        "augmentations": "advanced",
        "scheduler": "ReduceLROnPlateau",
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "samples_per_class": 20,
        "validation_set_size": "reduced"
    },
}
