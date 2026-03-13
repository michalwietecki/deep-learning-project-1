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