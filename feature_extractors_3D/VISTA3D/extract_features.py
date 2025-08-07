import sys
import torch
from glob import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    CropForegroundd, Orientationd, Spacingd, ToTensord, 
    ScaleIntensityd, DivisiblePadd, ThresholdIntensityd, SpatialPadd, CenterSpatialCropd
)
import nibabel as nib
import numpy as np
from monai.data import MetaTensor
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from loguru import logger
import monai

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, weights_path: str):
        """Load model weights from file"""
        pass

    # @abstractmethod
    # def preprocess(self, x):
    #     """Preprocess input data before forward pass"""
    #     pass

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model"""
        pass

class VISTA3DExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = monai.networks.nets.segresnet_ds.SegResEncoder(
            spatial_dims=3,
            in_channels=1,
            init_filters=48,
            blocks_down=[1, 2, 2, 4, 4],
            norm="instance",
            head_module=lambda x: torch.nn.functional.adaptive_avg_pool3d(
                x[-1], 1
            ).flatten(
                start_dim=1
            ),  # Get only the last feature across block levels and average pool it.
        )
        # self.transforms = get_transforms(
        #     orient="RAS",
        #     scale_range=(-1024, 2048),
        #     spatial_size=(48, 48, 48),
        #     spacing=(1, 1, 1),
        # )

    def load(self, weights_path: str = None):
        # Download weights from huggingface if path not provided
        # if weights_path is None:
        #     weights_path = "model_vista3d.pt"
        #     if not os.path.exists(weights_path):
        #         weights_path = wget.download(
        #             "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_vista3d.pt",
        #             bar=wget.bar_adaptive,
        #         )

        weights_path = "/home/johannes/Data/SSD_1.9TB/MultiViewGCN/3D_feature_extractors/VISTA3D/model_vista3d.pt"
        weights = torch.load(weights_path)
        # Modify prefix of weights to match model structure
        weights = {
            k.replace("image_encoder.encoder.", ""): v for k, v in weights.items()
        }
        msg = self.model.load_state_dict(
            weights, strict=False
        )  # Set strict to False as we load only the encoder

        logger.info(msg)
        self.model.eval()

    # def preprocess(self, x):
    #     return self.transforms(x)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)      


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

    model = VISTA3DExtractor().cuda() if torch.cuda.is_available() else VISTA3DExtractor()

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
        case "liver_ct_grading_binary":
            files = [file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/liver/CECT/HCC_CHCC_C2/*.nii.gz")]
            volumes = sorted([file for file in files if not "mask" in file])
            masks = sorted([file for file in files if "mask" in file])  
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")    

    transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),        
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        # ThresholdIntensityd(keys=["image"], threshold=4000.0, above=True, cval=0.0),  # Threshold to remove background
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        CropForegroundd(keys=["image", "mask"], source_key="mask", select_fn=lambda x: x > 0),
        SpatialPadd(keys=["image", "mask"], spatial_size=(128, 128, 128), mode="constant", constant_values=0),
        CenterSpatialCropd(keys=["image", "mask"], roi_size=(128, 128, 128)),    
        # DivisiblePadd(keys=["image", "mask"], k=16),  
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
        
        torch.save(output, f"{volume.replace('.nii.gz', '_VISTA3D_features.pt')}")
        print(f"Processed {volume} with output shape: {output.shape}")
        

if __name__ == "__main__":

    for dataset in ["liver_ct_grading_binary"]:
    # for dataset in ["sarcoma_t1_grading_binary", "sarcoma_t2_grading_binary", 
    #                 "glioma_t1c_grading_binary", "glioma_flair_grading_binary", 
    #                 "breast_mri_grading_binary", "headneck_ct_hpv_binary", 
    #                 "kidney_ct_grading_binary", "liver_ct_riskscore_binary"]:
        
        extract_features(dataset=dataset)