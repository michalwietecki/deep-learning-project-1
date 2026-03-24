import torch
import os
import matplotlib.pyplot as plt
from utils.config import TRAINED_MODELS_DIR
import numpy as np

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

def load_all_scenarios(root_dir = TRAINED_MODELS_DIR, plot_acc = False):
    all_results = {}

    for subdir, dirs, files in os.walk(root_dir):
        scenario_name = os.path.basename(subdir)
        
        if not files:
            continue

        all_results[scenario_name] = {}

        for filename in files:
            if filename.endswith(".pth"):
                model_name = filename.replace(".pth", "")
                full_path = os.path.join(subdir, filename)
                
                history, config = load_trained_model(full_path, plot_acc=plot_acc)
                
                all_results[scenario_name][model_name] = {
                    'history': history,
                    'config': config
                }
                
    return all_results


def load_trained_model(filepath, model=None, plot_acc=False):

    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return None

    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    history = checkpoint.get('history')
    config = checkpoint.get('config')
    # scenario_name and model from path
    scenario_name = os.path.basename(os.path.dirname(filepath)) 
    filename = os.path.basename(filepath)
    model_name = os.path.splitext(filename)[0]

    # line plot of training history

    if history is not None and plot_acc:
        plt.figure(figsize=(8, 4))
        
        # plt.subplot(1, 2, 1)
        # plt.plot(history['train_loss'], label='Train Loss')
        # plt.plot(history['val_loss'], label='Val Loss')
        # plt.title('Loss over Epochs')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()

        plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
        plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
        plt.title(f'Accuracy over Epochs - {scenario_name}: {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.xlim(0,11)
        plt.ylim(0,105)
        plt.grid()
        
        plt.tight_layout()
        plt.show()
    
    # load model weights if model architecture is provided
    if model is not None:
        state_dict = checkpoint.get('model_state_dict')
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Model loaded on {device}")

    return history, config