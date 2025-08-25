from torch.utils.data import Dataset
import torchio as tio
import SimpleITK as sitk
import torch


class FinetuningDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # print(sample)
        volume = sitk.ReadImage(sample)
        volume = sitk.GetArrayFromImage(volume)
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        # print(volume.shape)

        # volume = tio.ScalarImage(sample)
        # volume = volume.tensor
        label = self.labels[idx]
        
        return volume, label