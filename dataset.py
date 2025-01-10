import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


class SarcomaDataset(Dataset):
    def __init__(self, method, split, size, pool_data):
        super().__init__()

        self.method = method
        self.split = split

        match method:
            case "deep_learning":
                if pool_data:
                    data = []
                    data.extend(torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_train_dinov2-{size}.pt", weights_only=False))
                    data.extend(torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_test_dinov2-{size}.pt", weights_only=False))
                    labels = np.array([graph.label for graph in data])
                    labels = np.array([0 if item == 1 else 1 for item in labels])

                    match split:
                        case "train":
                            self.data, _, self.labels, _ = train_test_split(data, labels, train_size=0.8, stratify=labels, random_state=42)
                            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                            self.class_weights = torch.tensor(self.class_weights).to(torch.float32)

                        case "val":
                            _, val_test_data, _, val_test_labels = train_test_split(data, labels, train_size=0.8, stratify=labels, random_state=42)
                            self.data, _, self.labels, _ = train_test_split(val_test_data, val_test_labels, train_size=0.5, stratify=val_test_labels, random_state=42)                            

                        case "test":
                            _, val_test_data, _, val_test_labels = train_test_split(data, labels, train_size=0.8, stratify=labels, random_state=42)
                            _, self.data, _, self.labels = train_test_split(val_test_data, val_test_labels, train_size=0.5, stratify=val_test_labels, random_state=42)


                else:
                    match split:
                        case "train":
                            self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_train_dinov2-{size}.pt", weights_only=False)
                            self.labels = np.array([graph.label for graph in self.data])
                            self.labels = np.array([0 if item == 1 else 1 for item in self.labels])

                            self.data, _, self.labels, _ = train_test_split(self.data, self.labels, train_size=0.8, stratify=self.labels, random_state=42)

                            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                            self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
                        
                        case "val":
                            self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_train_dinov2-{size}.pt", weights_only=False)
                            self.labels = np.array([graph.label for graph in self.data])
                            self.labels = np.array([0 if item == 1 else 1 for item in self.labels])
                            
                            _, self.data, _, self.labels = train_test_split(self.data, self.labels, train_size=0.8, stratify=self.labels, random_state=42)
                     

                        case "test":
                            self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_test_dinov2-{size}.pt", weights_only=False)
                            self.labels = np.array([graph.label for graph in self.data])
                            self.labels = np.array([0 if item == 1 else 1 for item in self.labels])
                    
            
            case "radiomics":
                match split:
                    case "train":

                        self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/radiomics_t1_train.pt_dinov2-{size}", weights_only=False)
                        self.labels = np.array([graph.label for graph in self.data])
                        self.labels = np.array([0 if item == 1 else 1 for item in self.labels])
                        self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                        self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to("cuda")
                    
                    case "test":
                        self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/radiomics_t1_test_dinov2-{size}.pt", weights_only=False)
                        self.labels = np.array([graph.label for graph in self.data])
                        self.labels = np.array([0 if item == 1 else 1 for item in self.labels])

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data = self.data[index]
        data.x = data.x.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.long)       

        label = self.labels[index]
        label = torch.tensor(label)

        return data, label

class SarcomaDatasetCV(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if isinstance(sample, str):
            sample = torch.load(sample, weights_only=True)
            # sample = sample.repeat(3, 1, 1)

        return sample, label
    



        