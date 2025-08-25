from DINOv2.model import DINOv2Extractor
from glob import glob
import torchio as tio
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def find_largest_lesion_slice(mask: torch.Tensor, axis: int) -> int:
    assert mask.ndim == 3, "Mask must be a 3D tensor"
    assert axis in [0, 1, 2], "Axis must be 0, 1, or 2"

    # Move the desired axis to the front
    slices = mask.moveaxis(axis, 0)

    # Compute lesion area (non-zero count) per slice
    areas = torch.sum(slices != 0, dim=(1, 2))

    # Get index of maximum area
    max_index = torch.argmax(areas).item()
    return max_index

def extract_axial_features(dataset, model_name, views):

    match dataset:
        case "glioma_t1c_grading_binary":
            volumes = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*T1c_bias.nii.gz"))
            masks = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*tumor_segmentation_merged.nii.gz"))
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
        case "liver_ct_grading_binary":
            files = [file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/liver/CECT/HCC_CHCC_C2/*.nii.gz")]
            volumes = sorted([file for file in files if not "mask" in file])
            masks = sorted([file for file in files if "mask" in file])  
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")

    match model_name:
        case "DINOv2":
            pass
            model = DINOv2Extractor().cuda()
        case _:
            raise ValueError(f"Unknown model: {model}")

    for volume, mask in tqdm(zip(volumes, masks), total=len(volumes), desc=f"Extracting axial {model_name} features from {dataset} with {views} views"):

        if os.path.exists(f"{volume.replace('.nii.gz', f'_{model_name}_{str(views).zfill(2)}views_axial_features.pt')}"):
            print(f"Skipping {volume}, features already extracted.")
            continue

        img = tio.ScalarImage(volume)
        seg = tio.LabelMap(mask)

        seg = tio.Resample(target=img)(seg)

        subject = tio.Subject(
            image=img,
            mask=seg
        )

        subject = tio.ToCanonical()(subject)
        subject = tio.Resample((1.0, 1.0, 1.0))(subject)        
        subject_temp = tio.CropOrPad(mask_name="mask")(subject)
        max_dim = np.max(subject_temp.image.shape)
        subject = tio.CropOrPad(target_shape=(max_dim, max_dim, max_dim), mask_name="mask")(subject)

        img_tensor = subject.image.tensor[0]
        seg_tensor = subject.mask.tensor[0]
        index_axial = find_largest_lesion_slice(seg_tensor, axis=2)

        img_slices = []
        match views:
            case 1:
                img_slices.append(img_tensor[:, :, index_axial])
            case 3:
                img_slices.append(img_tensor[:, :, index_axial-1])
                img_slices.append(img_tensor[:, :, index_axial])
                img_slices.append(img_tensor[:, :, index_axial+1])
            case _:

                assert views in [8, 16, 24]

                upper_index = index_axial + (views // 2)
                lower_index = index_axial - (views // 2)

                for i in range(lower_index, upper_index):

                    if i < 0:
                        i = 0
                    if i >= img_tensor.shape[2]:
                        i = img_tensor.shape[2] - 1

                    img_slices.append(img_tensor[:, :, i])

        encodings = []
        for idx, img_slice in enumerate(img_slices):
            img_slice = img_slice.numpy()
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            img_slice = (img_slice * 255).astype(np.uint8)
            img_pil = transforms.ToPILImage()(img_slice)
            img_pil.save("axial_output_image.png", format="PNG")
            encodings.append(model(img_pil))
        
        features = torch.concat(encodings, dim=0)

        torch.save(features, f"{volume.replace('.nii.gz', f'_{model_name}_{str(views).zfill(2)}views_axial_features.pt')}")
        # print(f"Processed {volume} with features shape: {features.shape}")
    
    os.remove("axial_output_image.png")

if __name__ == "__main__":

    for dataset in ["glioma_t1c_grading_binary", "sarcoma_t2_grading_binary", "breast_mri_grading_binary", "headneck_ct_hpv_binary", "kidney_ct_grading_binary", "liver_ct_grading_binary"]:
        for model in ["DINOv2"]:
            for views in [1, 3, 8, 16, 24]:
                extract_axial_features(dataset, model, views)