from glob import glob
import os
import pandas as pd
from radiomics import featureextractor
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    CropForegroundd, Orientationd, Spacingd, ToTensord, 
    ScaleIntensityd, DivisiblePadd, ThresholdIntensityd, NormalizeIntensityd
)
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import torch
from sklearn.preprocessing import normalize
import torchio as tio


def extract_features(dataset: str):

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

    # transforms = Compose([
    #     LoadImaged(keys=["image", "mask"]),
    #     EnsureChannelFirstd(keys=["image", "mask"]),        
    #     Orientationd(keys=["image", "mask"], axcodes="RAS"),
    #     Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    #     NormalizeIntensityd(keys=["image"], nonzero=True),
    #     ToTensord(keys=["image", "mask"]),
    # ])


    extractor = featureextractor.RadiomicsFeatureExtractor(label=1)

    for volume, mask in tqdm(zip(volumes, masks)):

        # if os.path.exists(f"{volume.replace('.nii.gz', '_PyRadiomics_features.pt')}"):
        #     print(f"Skipping {volume}, already processed.")
        #     continue

        print(f"\nImage: {volume}")
        print(f"Mask:  {mask}\n")

        

        subject = tio.Subject(
            image=tio.ScalarImage(volume),
            mask=tio.LabelMap(mask)
        )

        subject = tio.ToCanonical()(subject)
        subject = tio.Resample((1.0, 1.0, 1.0))(subject)
        subject = tio.ZNormalization()(subject)

        image_tio = subject['image']
        mask_tio = subject['mask']
        
        mask_tio = tio.Resample(target=image_tio)(mask_tio)

        image_sitk = image_tio.as_sitk()
        mask_sitk = mask_tio.as_sitk() 
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
        mask_sitk = mask_sitk / sitk.GetArrayFromImage(mask_sitk).max()  # Normalize mask to 0-1       

        features = extractor.execute(image_sitk, mask_sitk)

        feature_vector = []
        for key in features.keys():
            if key.startswith("original_"):
                feature_vector.append(features[key].item())

        # feature_vector = np.array(feature_vector).reshape(1, -1)
        # feature_vector = normalize(feature_vector, norm='l2')
        feature_vector = torch.tensor(feature_vector).view(1, -1)

        torch.save(feature_vector, f"{volume.replace('.nii.gz', '_PyRadiomics_features.pt')}")
        print(f"Processed {volume} with output shape: {feature_vector.shape}")    
    

if __name__ == "__main__":
    
    for dataset in ["liver_ct_grading_binary"]:
    # for dataset in ["sarcoma_t1_grading_binary", "sarcoma_t2_grading_binary", 
    #                 "glioma_t1c_grading_binary", "glioma_flair_grading_binary", 
    #                 "breast_mri_grading_binary", "headneck_ct_hpv_binary", 
    #                 "kidney_ct_grading_binary", "liver_ct_riskscore_binary"]:
        
        extract_features(dataset=dataset)