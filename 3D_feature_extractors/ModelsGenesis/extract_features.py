import sys
import torch
from glob import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    CropForegroundd, Orientationd, Spacingd, ToTensord, 
    ScaleIntensityd, DivisiblePadd, ThresholdIntensityd
)
import nibabel as nib
import numpy as np
from monai.data import MetaTensor
from tqdm import tqdm
import unet3d
import torch.nn.functional as F
import torch.nn as nn

# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        # self.dense_1 = nn.Linear(512, 1024, bias=True)
        # self.dense_2 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        # self.linear_out = self.dense_1(self.out_glb_avg_pool)
        # final_out = self.dense_2( F.relu(self.linear_out))
        # return final_out
        return self.out_glb_avg_pool       


def save_metatensor_as_nifti(tensor: MetaTensor, filename: str):
    # Step 1: Detach and convert to NumPy
    data_np = tensor.detach().cpu().numpy()

    # Step 2: Remove channel dimension if necessary
    if data_np.shape[0] == 1:  # shape (1, D, H, W) -> (D, H, W)
        data_np = data_np[0]

    # Step 3: Get affine matrix (fallback to identity)
    affine = tensor.meta.get("affine", np.eye(4))

    # Step 4: Save with nibabel
    img = nib.Nifti1Image(data_np, affine)
    nib.save(img, filename)
    print(f"Saved NIfTI image to: {filename}")

def extract_features(dataset: str):

    base_model = unet3d.UNet3D()

    #Load pre-trained weights
    weight_dir = './Genesis_Chest_CT.pt'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict, strict=False)
    model = TargetNet(base_model).cuda() if torch.cuda.is_available() else TargetNet(base_model)

    match dataset:
        case "glioma_t1c_grading_binary":
            volumes = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*T1c_bias.nii.gz"))
            masks = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*tumor_segmentation_merged.nii.gz"))
        case "glioma_flair_grading_binary":
            volumes = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*FLAIR_bias.nii.gz"))
            masks = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*tumor_segmentation_merged.nii.gz"))
        case "sarcoma_t1_grading_binary":
            files = [file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/*/T1/*nii.gz")]
            volumes = sorted([file for file in files if not "label" in file])
            masks = sorted([file for file in files if "label" in file])
        case "sarcoma_t2_grading_binary":
            files = [file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/sarcoma/*/T2/*nii.gz")]
            volumes = sorted([file for file in files if not "label" in file])
            masks = sorted([file for file in files if "label" in file])
        case "breast_mri_grading_binary":
            volumes = sorted([file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/breast/duke_tumor_grading/*0001.nii.gz")])
            masks = sorted([file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/breast/duke_tumor_grading/*segmentation.nii.gz")])
        case "headneck_ct_hpv_binary":
            files = [file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/headneck/converted_nii_merged/*/*.nii.gz")]
            volumes = sorted([file for file in files if not "mask" in file])
            masks = sorted([file for file in files if "mask" in file])  
        case "kidney_ct_grading_binary":
            files = [file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/kidney/converted_nii/*.nii.gz")]
            volumes = sorted([file for file in files if "arterial" in file])
            masks = sorted([file for file in files if "segmentation_tumor" in file])  
        case "liver_ct_riskscore_binary":
            files = [file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/liver/converted_nii/*.nii.gz")]
            volumes = sorted([file for file in files if not "segmentation" in file])
            masks = sorted([file for file in files if "segmentation" in file])  
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")    

    transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),        
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ThresholdIntensityd(keys=["image"], threshold=4000.0, above=True, cval=0.0),  # Threshold to remove background
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        CropForegroundd(keys=["image", "mask"], source_key="mask", select_fn=lambda x: x > 0),    
        DivisiblePadd(keys=["image", "mask"], k=16),  
        ToTensord(keys=["image"]),
    ])


    for volume, mask in tqdm(zip(volumes, masks)):
        data = {
            "image": volume,
            "mask": mask
        }
        
        transformed = transforms(data)
        
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]

        # save_metatensor_as_nifti(image_tensor, f"transformed.nii.gz")
        
        # Ensure the tensors are on the same device as the model
        image_tensor = image_tensor.cuda() if torch.cuda.is_available() else image_tensor
        mask_tensor = mask_tensor.cuda() if torch.cuda.is_available() else image_tensor
        
        # Forward pass through the model
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))  # Add batch dimension if needed
        
        torch.save(output, f"{volume.replace('.nii.gz', '_ModelsGenesis_features.pt')}")
        print(f"Processed {volume} with output shape: {output.shape}")
        

if __name__ == "__main__":

    for dataset in ["sarcoma_t1_grading_binary", "sarcoma_t2_grading_binary", 
                    "glioma_t1c_grading_binary", "glioma_flair_grading_binary", 
                    "breast_mri_grading_binary", "headneck_ct_hpv_binary", 
                    "kidney_ct_grading_binary", "liver_ct_riskscore_binary"]:
        extract_features(dataset=dataset)