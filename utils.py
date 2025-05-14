import yaml
import random
import numpy as np
import torch
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def seed_everything(seed: int):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Function to save the confusion matrix as a PNG
def save_confusion_matrix(y_true, y_pred, result_dir, split):

    filepath = os.path.join(result_dir, f"confusion_matrix_{split}.png")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Class 0', 'Class 1']  # Class labels

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the figure to avoid displaying it in interactive environments



