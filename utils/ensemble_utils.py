import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from utils.models import get_model
from utils.evaluation_utils import load_trained_model


def load_models_for_ensemble(checkpoint_paths: dict, device):
    """
        checkpoint_paths: dict {model_name: filepath}, np.:
            {
                "baseline_cnn":               "./trained_models/.../baseline_cnn_420.pth",
                "efficientnet_b0":            "./trained_models/.../efficientnet_b0_420.pth",
                "efficientnet_b0_pretrained": "./trained_models/.../efficientnet_b0_pretrained_420.pth",
            }
        device: torch.device

    """
    loaded = []
    for model_name, path in checkpoint_paths.items():
        _, config = load_trained_model(path)
        if config is None:
            config = {}

        model = get_model(model_name, config)

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']

        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print(f"loaded: {model_name}")
        loaded.append((model_name, model))

    return loaded

@torch.no_grad()
def get_all_logits(models_list, loader, device):
    all_logits = []
    all_targets = None

    for model_name, model in models_list:
        model_logits = []
        targets_list = []

        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            model_logits.append(outputs.cpu())
            if all_targets is None:
                targets_list.append(targets)

        all_logits.append(torch.cat(model_logits, dim=0))

        if all_targets is None:
            all_targets = torch.cat(targets_list, dim=0)

    all_logits = torch.stack(all_logits, dim=0)
    return all_logits, all_targets


def hard_voting(all_logits):
    votes = all_logits.argmax(dim=2)
    predictions, _ = torch.mode(votes, dim=0)
    return predictions


def soft_voting(all_logits):
    probs = F.softmax(all_logits, dim=2)
    avg_probs = probs.mean(dim=0)
    return avg_probs.argmax(dim=1)

def run_ensemble_evaluation(stage_name, checkpoint_paths, val_loader, device):
    print(f"\n{'='*55}")
    print(f"  Ensemble: {stage_name}")
    print(f"{'='*55}")

    models_list = load_models_for_ensemble(checkpoint_paths, device)
    all_logits, all_targets = get_all_logits(models_list, val_loader, device)
    targets_np = all_targets.numpy()

    rows = []

    # singular models
    for i, (model_name, _) in enumerate(models_list):
        preds = all_logits[i].argmax(dim=1)
        acc = (preds == all_targets).float().mean().item() * 100
        f1  = f1_score(targets_np, preds.numpy(), average='macro')
        rows.append({"stage": stage_name, "mode": model_name,
                     "accuracy": round(acc, 2), "f1": round(f1, 4)})

    # Hard voting
    hard_preds = hard_voting(all_logits)
    hard_acc   = (hard_preds == all_targets).float().mean().item() * 100
    hard_f1    = f1_score(targets_np, hard_preds.numpy(), average='macro')
    rows.append({"stage": stage_name, "mode": "hard_voting",
                 "accuracy": round(hard_acc, 2), "f1": round(hard_f1, 4)})

    # Soft voting
    soft_preds = soft_voting(all_logits)
    soft_acc   = (soft_preds == all_targets).float().mean().item() * 100
    soft_f1    = f1_score(targets_np, soft_preds.numpy(), average='macro')
    rows.append({"stage": stage_name, "mode": "soft_voting",
                 "accuracy": round(soft_acc, 2), "f1": round(soft_f1, 4)})

    df = pd.DataFrame(rows)

    single_mask     = ~df["mode"].isin(["hard_voting", "soft_voting"])
    best_single_acc = df.loc[single_mask, "accuracy"].max()
    best_single_f1  = df.loc[single_mask, "f1"].max()
    soft_row        = df[df["mode"] == "soft_voting"].iloc[0]

    print(df.to_string(index=False))
    print(f"\n soft vs best single"
          f"Acc: {soft_row['accuracy'] - best_single_acc:+.2f}pp  |  "
          f"F1:  {soft_row['f1'] - best_single_f1:+.4f}")

    return df
