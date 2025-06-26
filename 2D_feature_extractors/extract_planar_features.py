from DINOv2.model import DINOv2Extractor
from glob import glob
import torchio as tio
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt

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

def get_foreground_extent(mask: torch.Tensor, margin: float = 1.1):
    if mask.ndim != 3:
        raise ValueError("Expected 3D tensor of shape (D, H, W)")

    # Get foreground indices
    foreground_voxels = (mask > 0).nonzero(as_tuple=False)

    if foreground_voxels.numel() == 0:
        raise ValueError("Mask contains no foreground voxels.")

    z_coords, y_coords, x_coords = foreground_voxels[:, 0], foreground_voxels[:, 1], foreground_voxels[:, 2]

    x_dim = x_coords.max().item() - x_coords.min().item() + 1
    y_dim = y_coords.max().item() - y_coords.min().item() + 1
    z_dim = z_coords.max().item() - z_coords.min().item() + 1

    x_dim = int(x_dim * margin)
    y_dim = int(y_dim * margin)
    z_dim = int(z_dim * margin)

    z_dim = np.max([z_dim, 30])

    return (x_dim, y_dim, z_dim)

def extract_planar_features(dataset, model_name, views):

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

    match model_name:
        case "DINOv2":
            model = DINOv2Extractor().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        case _:
            raise ValueError(f"Unknown model: {model}")
        
    for volume, mask in zip(volumes, masks):

        # if os.path.exists(f"{volume.replace('.nii.gz', f'_{model_name}_{str(views).zfill(2)}views_planar_features.pt')}"):
        #     print(f"Skipping {volume}, features already extracted.")
        #     continue

        img = tio.ScalarImage(volume)
        seg = tio.LabelMap(mask)

        seg = tio.Resample(target=img)(seg)

        subject = tio.Subject(
            image=img,
            mask=seg
        )

        subject = tio.ToCanonical()(subject)
        subject = tio.Resample((1.0, 1.0, 1.0))(subject)

        x_dim, y_dim, z_dim = get_foreground_extent(mask=subject.mask.tensor[0])

        subject = tio.CropOrPad(target_shape=(x_dim, y_dim, z_dim), mask_name="mask")(subject)

        max_dim = np.max(subject.image.shape)
        subject = tio.CropOrPad((max_dim, max_dim, max_dim))(subject)

        img_tensor = subject.image.tensor[0]
        seg_tensor = subject.mask.tensor[0]

        index_saggital = find_largest_lesion_slice(seg_tensor, axis=0)
        index_coronal = find_largest_lesion_slice(seg_tensor, axis=1)
        index_axial = find_largest_lesion_slice(seg_tensor, axis=2)

        img_slices = []
        match views:
            case 1:
                img_slices.append(img_tensor[:, :, index_axial])
            case 3:
                img_slices.append(img_tensor[index_saggital, :, :])
                img_slices.append(img_tensor[:, index_coronal, :])
                img_slices.append(img_tensor[:, :, index_axial])
            case _:
                assert views in [8, 12, 16, 20, 24], "Unsupported number of views"

                upper_index = index_axial + (views // 2)
                lower_index = index_axial - (views // 2)

                for i in range(lower_index, upper_index):
                    img_slices.append(img_tensor[:, :, i])

        encodings = []
        for idx, img_slice in enumerate(img_slices):
            img_slice = img_slice.numpy()
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            img_slice = (img_slice * 255).astype(np.uint8)
            img_pil = transforms.ToPILImage()(img_slice)
            img_pil.save("planar_output_image.png", format="PNG")
            encodings.append(model(img_pil))
        
        features = torch.concat(encodings, dim=0)

        torch.save(features, f"{volume.replace('.nii.gz', f'_{model_name}_{str(views).zfill(2)}views_planar_features.pt')}")
        print(f"Processed {volume} with features shape: {features.shape}")


if __name__ == "__main__":
    
    for dataset in ["sarcoma_t1_grading_binary", "sarcoma_t2_grading_binary", 
                    "glioma_t1c_grading_binary", "glioma_flair_grading_binary", 
                    "breast_mri_grading_binary", "headneck_ct_hpv_binary", 
                    "kidney_ct_grading_binary", "liver_ct_riskscore_binary"]:
        for model in ["DINOv2"]:
            for views in [1, 3, 8, 12, 16, 20, 24]:
                extract_planar_features(dataset, model, views)

