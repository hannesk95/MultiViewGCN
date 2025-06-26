import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split


# class SarcomaDataset(Dataset):
#     def __init__(self, method, split, size, pool_data):
#         super().__init__()

#         self.method = method
#         self.split = split

#         match method:
#             case "deep_learning":
#                 if pool_data:
#                     data = []
#                     data.extend(torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_train_dinov2-{size}.pt", weights_only=False))
#                     data.extend(torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_test_dinov2-{size}.pt", weights_only=False))
#                     labels = np.array([graph.label for graph in data])
#                     labels = np.array([0 if item == 1 else 1 for item in labels])

#                     match split:
#                         case "train":
#                             self.data, _, self.labels, _ = train_test_split(data, labels, train_size=0.8, stratify=labels, random_state=42)
#                             self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
#                             self.class_weights = torch.tensor(self.class_weights).to(torch.float32)

#                         case "val":
#                             _, val_test_data, _, val_test_labels = train_test_split(data, labels, train_size=0.8, stratify=labels, random_state=42)
#                             self.data, _, self.labels, _ = train_test_split(val_test_data, val_test_labels, train_size=0.5, stratify=val_test_labels, random_state=42)                            

#                         case "test":
#                             _, val_test_data, _, val_test_labels = train_test_split(data, labels, train_size=0.8, stratify=labels, random_state=42)
#                             _, self.data, _, self.labels = train_test_split(val_test_data, val_test_labels, train_size=0.5, stratify=val_test_labels, random_state=42)


#                 else:
#                     match split:
#                         case "train":
#                             self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_train_dinov2-{size}.pt", weights_only=False)
#                             self.labels = np.array([graph.label for graph in self.data])
#                             self.labels = np.array([0 if item == 1 else 1 for item in self.labels])

#                             self.data, _, self.labels, _ = train_test_split(self.data, self.labels, train_size=0.8, stratify=self.labels, random_state=42)

#                             self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
#                             self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
                        
#                         case "val":
#                             self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_train_dinov2-{size}.pt", weights_only=False)
#                             self.labels = np.array([graph.label for graph in self.data])
#                             self.labels = np.array([0 if item == 1 else 1 for item in self.labels])
                            
#                             _, self.data, _, self.labels = train_test_split(self.data, self.labels, train_size=0.8, stratify=self.labels, random_state=42)
                     

#                         case "test":
#                             self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/deep_learning_t1_test_dinov2-{size}.pt", weights_only=False)
#                             self.labels = np.array([graph.label for graph in self.data])
#                             self.labels = np.array([0 if item == 1 else 1 for item in self.labels])
                    
            
#             case "radiomics":
#                 match split:
#                     case "train":

#                         self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/radiomics_t1_train.pt_dinov2-{size}", weights_only=False)
#                         self.labels = np.array([graph.label for graph in self.data])
#                         self.labels = np.array([0 if item == 1 else 1 for item in self.labels])
#                         self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
#                         self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to("cuda")
                    
#                     case "test":
#                         self.data = torch.load(f"/home/johannes/Code/MultiViewGCN/data/radiomics_t1_test_dinov2-{size}.pt", weights_only=False)
#                         self.labels = np.array([graph.label for graph in self.data])
#                         self.labels = np.array([0 if item == 1 else 1 for item in self.labels])

    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
        
#         data = self.data[index]
#         data.x = data.x.to(torch.float32)
#         data.edge_index = data.edge_index.to(torch.long)       

#         label = self.labels[index]
#         label = torch.tensor(label)

#         return data, label

# class SarcomaDatasetCV(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.labels[idx]

#         if isinstance(sample, str):
#             sample = torch.load(sample, weights_only=True)
#             # sample = sample.repeat(3, 1, 1)

#         return sample, label
    
class GNNDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]        
        sample = torch.load(sample, weights_only=False)            

        return sample
    
class MLPDataset(Dataset):
    def __init__(self, data: list, perspective: str = "spherical", n_views: int = 24):
        self.data = data
        self.perspective = perspective
        self.n_views = n_views

        assert self.perspective in ["spherical", "axial"], "Perspective must be either 'spherical' or 'axial'."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]        
        sample = torch.load(sample, weights_only=False)  
        
        x = sample.x
        x = x.to(torch.float32)  

        # edge_index = sample.edge_index
        # edge_index = edge_index.to(torch.long)

        if self.perspective == "axial":
            middle_slice = x.shape[0] // 2

            if (self.n_views == 1) or (self.n_views == 3):
                x = x[(middle_slice - (self.n_views//2)):(middle_slice + (self.n_views//2))+1, :]
            
            else:
                x = x[(middle_slice - (self.n_views//2)):(middle_slice + (self.n_views//2)), :]

            if x.shape[0] != self.n_views:
                pad_rows = self.n_views - x.shape[0]
                pad_top = pad_rows // 2
                pad_bottom = pad_rows - pad_top

                top_padding = torch.zeros(pad_top, x.shape[1], device=x.device, dtype=x.dtype)
                bottom_padding = torch.zeros(pad_bottom, x.shape[1], device=x.device, dtype=x.dtype)
                x = torch.cat([top_padding, x, bottom_padding], dim=0)

        y = sample.label
        y = y.to(torch.long)

        return x, y

def get_data(datset_name: str, n_views: int, dino_size: str):

    if datset_name == "sarcoma_t1":        
        # files = glob(f"./data/sarcoma/*/T1/*graph-fibonacci_views{n_views}_dinov2-{dino_size}.pt")
        files = glob(f"./data/sarcoma/*/T1/*graph-fibonacci-edge_attr_views{n_views}_dinov2-{dino_size}.pt")
        data = [torch.load(temp, weights_only=False) for temp in files]
        labels = [graph.label.item() for graph in data]
        labels = [0 if item == 1 else 1 for item in labels]

        for i in range(len(data)):
            data[i].label = labels[i]

        labels = torch.tensor(labels).to(torch.long)
    
    elif datset_name == "sarcoma_t2":        
        files = glob(f"./data/sarcoma/*/T2/*graph-fibonacci_views{n_views}_dinov2-{dino_size}.pt")
        data = [torch.load(temp, weights_only=False) for temp in files]
        labels = [graph.label.item() for graph in data]
        labels = [0 if item == 1 else 1 for item in labels]

        for i in range(len(data)):
            data[i].label = labels[i]
            
        labels = torch.tensor(labels).to(torch.long)
    
    elif datset_name == "headneck":
        files = glob(f"./data/headneck/*/converted_nii/*/*graph-fibonacci*views{n_views}_dinov2-{dino_size}.pt")        
        data = [torch.load(temp, weights_only=False) for temp in files]
        labels = [graph.label.item() for graph in data]
        labels = [0 if item == 1 else 1 for item in labels]
        labels = torch.tensor(labels).to(torch.long)
    
    elif datset_name == "vessel":
        files = glob(f"./data/medmnist3d/vesselmnist3d_64/*graph-fibonacci*views{n_views}_dinov2-{dino_size}.pt")        
        data = [torch.load(temp, weights_only=False) for temp in files]
        labels = [graph.label.item() for graph in data]
        # labels = [0 if item == 1 else 1 for item in labels]
        labels = torch.tensor(labels).to(torch.long)
    
    elif datset_name == "synapse":
        files = glob(f"./data/medmnist3d/synapsemnist3d_64/*graph-fibonacci*views{n_views}_dinov2-{dino_size}.pt")        
        data = [torch.load(temp, weights_only=False) for temp in files]
        labels = [graph.label.item() for graph in data]
        # labels = [0 if item == 1 else 1 for item in labels]
        labels = torch.tensor(labels).to(torch.long)
    
    elif datset_name == "adrenal":
        files = glob(f"./data/medmnist3d/adrenalmnist3d_64/*graph-fibonacci*views{n_views}_dinov2-{dino_size}.pt")        
        data = [torch.load(temp, weights_only=False) for temp in files]
        labels = [graph.label.item() for graph in data]
        # labels = [0 if item == 1 else 1 for item in labels]
        labels = torch.tensor(labels).to(torch.long)

    elif datset_name == "nodule":
        files = glob(f"./data/medmnist3d/nodulemnist3d_64/*graph-fibonacci*views{n_views}_dinov2-{dino_size}.pt")        
        data = [torch.load(temp, weights_only=False) for temp in files]
        labels = [graph.label.item() for graph in data]
        # labels = [0 if item == 1 else 1 for item in labels]
        labels = torch.tensor(labels).to(torch.long)

    return data, labels



def get_data_dicts(task: str) -> tuple[list[str], list[int]]:

    if task == "sarcoma_t1_grading_binary":
        images = glob("./data/sarcoma/*/T1/*.nii.gz")
        images = sorted([img for img in images if not "label" in img])

        masks = glob("./data/sarcoma/*/T1/*.nii.gz")
        masks = sorted([mask for mask in masks if "label" in mask])        
        
        labels = []
        subjects = []
        df = pd.read_csv('./data/sarcoma/patient_metadata.csv')
        for img in images:
            img = img.split("/")[-1].split(".")[0]            
            patient_id = img[:6] if img.startswith("Sar") else img[:4]
            grading = df[df["ID"] == patient_id].Grading.item()
            grading = 0 if grading == 1 else 1            
            labels.append(grading)
            subjects.append(patient_id)

        data_dicts = [{"image": img, "mask": mask, "label": label} for img, mask, label in zip(images, masks, labels)]
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return data_dicts, labels, subjects


def split_dataset(images: list[str], labels: list[int], val_size: float = 0.15, test_size: float = 0.15, seed: int = 42):
    

    # First split off the test set
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=val_size + test_size, stratify=labels, random_state=seed
    )

    # Then split temp into val and test
    relative_test_size = test_size / (val_size + test_size)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=relative_test_size, stratify=temp_labels, random_state=seed
    )

    return train_images, train_labels, val_images, val_labels, test_images, test_labels