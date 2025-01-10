import yaml
import random
import numpy as np
import torch
import os

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
