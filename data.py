import torch
from torch.utils.data import Dataset, DataLoader
import random
import pickle

class MINDDataset(Dataset):
    def __init__(self, behaviors_path):
        with open(behaviors_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        pos_id = item['pos_id'] 
        all_negs = item['neg_candidates']
        if len(all_negs) >= 4:
            sampled_negs = random.sample(list(all_negs), 4)
        else:
            sampled_negs = list(all_negs) + [0] * (4 - len(all_negs))
        
        candidates = [pos_id] + sampled_negs

        return {
            'user_idx': torch.tensor(item['user_idx'], dtype=torch.long),
            'history': torch.tensor(item['history'], dtype=torch.long),
            'candidates': torch.tensor(candidates, dtype=torch.long),
            'label': torch.tensor(0, dtype=torch.long) # Positive label index
        }

