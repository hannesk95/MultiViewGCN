import torch
from torch.utils.data import Dataset

class FeatureExtractorDataset(Dataset):
    def __init__(self, data: list, labels: list):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]        
        sample = torch.load(sample, weights_only=False)      
        sample = sample.view(-1).view(1, -1)  # Flatten the sample to a 1D tensor

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample, label