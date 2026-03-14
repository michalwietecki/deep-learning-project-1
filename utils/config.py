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
    
    try: 
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
    except Exception as e:
        pass


    print(f"full reproducibility with seed set at: {seed}")

# Statystyki specyficzne dla CINIC-10
CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]

# experiments configuration
EXPERIMENTS = {
    # stage 1: 3 models, for future comparison
    "stage_1": {
        "models": ["baseline_cnn", "efficientnet_b0", "efficientnet_b0_pretrained"],
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 64,
        "optimizer": "ADAM",
        "augmentations": "none"
    },
}