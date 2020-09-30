import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import util
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pathlib import Path

class Signals():
    def __init__(self, path):
        super(Signals, self).__init__()
        
        signals = torch.load(path)
        
        self.clean = signals[:, 0, :].unsqueeze(dim=1)
        self.noisy = signals[:, 1, :].unsqueeze(dim=1)
        
        self.n_samples = self.clean.shape[0]
        
    
    def __getitem__(self, index):
        return self.clean[index, 0, :], self.noisy[index, 0, :]
    
    def __len__(self):
        return self.n_samples
