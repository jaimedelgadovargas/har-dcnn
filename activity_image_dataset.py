import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import utils

class ActivityImageDataset(Dataset):
  def __init__(self, signals_dir, label_dir):
    signals = []
    for signal_dir in signals_dir:
      signal_data = pd.read_csv(signal_dir, delim_whitespace=True, header=None).values
      signals.append(signal_data)
      
    self.signals = signals
    self.labels = pd.read_csv(label_dir, delim_whitespace=True, header=None).values

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    activity_image = utils.get_activity_image(self.signals, idx)
    label = self.labels[idx]
    activity_image = torch.tensor([activity_image[:36]], dtype=torch.float32)    
    label = torch.tensor(label, dtype=torch.long).flatten()
    label = label[0] - 1

    return activity_image, label
  
  
