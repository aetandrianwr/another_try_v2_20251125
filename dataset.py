"""
Dataset class for Geolife trajectory data
"""

import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


class GeolifeDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return {
            'locations': torch.LongTensor(sample['X']),
            'users': torch.LongTensor(sample['user_X']),
            'weekdays': torch.LongTensor(sample['weekday_X']),
            'start_minutes': torch.FloatTensor(sample['start_min_X']),
            'durations': torch.FloatTensor(sample['dur_X']),
            'time_diffs': torch.LongTensor(sample['diff']),
            'target': torch.LongTensor([sample['Y']])
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    """
    # Find max sequence length in batch
    max_len = max([item['locations'].size(0) for item in batch])
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    locations = torch.zeros(batch_size, max_len, dtype=torch.long)
    users = torch.zeros(batch_size, max_len, dtype=torch.long)
    weekdays = torch.zeros(batch_size, max_len, dtype=torch.long)
    start_minutes = torch.zeros(batch_size, max_len, dtype=torch.float)
    durations = torch.zeros(batch_size, max_len, dtype=torch.float)
    time_diffs = torch.zeros(batch_size, max_len, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['locations'].size(0)
        locations[i, :seq_len] = item['locations']
        users[i, :seq_len] = item['users']
        weekdays[i, :seq_len] = item['weekdays']
        start_minutes[i, :seq_len] = item['start_minutes']
        durations[i, :seq_len] = item['durations']
        time_diffs[i, :seq_len] = item['time_diffs']
        targets[i] = item['target']
        lengths[i] = seq_len
    
    # Create attention mask (1 for real tokens, 0 for padding)
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    return {
        'locations': locations,
        'users': users,
        'weekdays': weekdays,
        'start_minutes': start_minutes,
        'durations': durations,
        'time_diffs': time_diffs,
        'targets': targets,
        'mask': mask,
        'lengths': lengths
    }
