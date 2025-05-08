# utils.py
import torch
import numpy as np
import random
import os

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True # Can slow down, but good for reproducibility
        torch.backends.cudnn.benchmark = False   # Ensure deterministic behavior

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
