import os
import time
import sys
import torch
import argparse
from pathlib import Path
from glob import glob
import torchio as tio
from tqdm import tqdm
import numpy as np
from typing import List
import open3d as o3d
from scipy.ndimage import affine_transform
from scipy.ndimage import label, center_of_mass
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

def rescale_image(img: np.ndarray) -> np.ndarray:
    img = img - (np.min(img))
    img = img / (np.max(img))
    img = img * 255
    img = np.uint8(img)

    return img

def get_biggest_lesion_slice(volume, mask, axis=2, cmap='gray'):
    """
    Get a 2D slice of the largest lesion in a 3D volume based on the segmentation mask.
    
    Parameters:
        volume (numpy.ndarray): The 3D volume (e.g., CT or MRI scan).
        mask (numpy.ndarray): The 3D segmentation mask corresponding to the volume.
        axis (int): The axis along which to extract the slice (default is 2 for the last dimension).
        cmap (str): Colormap for visualization (default is 'gray').
    """
    if volume.shape != mask.shape:
        raise ValueError("Volume and mask must have the same shape.")
    if volume.ndim != 3 or mask.ndim != 3:
        raise ValueError("Both volume and mask must be 3D arrays.")
    
    # Label connected components in the segmentation mask
    labeled_mask, num_features = label(mask)
    
    if num_features == 0:
        raise ValueError("No lesions found in the segmentation mask.")
    
    # Identify the largest lesion
    region_sizes = np.array([(labeled_mask == i).sum() for i in range(1, num_features + 1)])
    largest_region_label = np.argmax(region_sizes) + 1
    
    # Compute the centroid of the largest lesion
    centroid = center_of_mass(mask, labeled_mask, largest_region_label)
    
    # Extract the index of the slice along the specified axis closest to the centroid
    slice_idx = int(round(centroid[axis]))
    
    # Extract the 2D slice from both the volume and the mask
    if axis == 0:
        slice_volume = volume[slice_idx, :, :]
        slice_mask = mask[slice_idx, :, :]
    elif axis == 1:
        slice_volume = volume[:, slice_idx, :]
        slice_mask = mask[:, slice_idx, :]
    elif axis == 2:
        slice_volume = volume[:, :, slice_idx]
        slice_mask = mask[:, :, slice_idx]
    else:
        raise ValueError("Invalid axis. Must be 0, 1, or 2.")
    
    return slice_volume, slice_mask

def get_max_dim_mask(mask):
    """
    Get the bounding box of a 3D binary segmentation mask.

    Parameters:
        mask (np.ndarray): A 3D binary mask (values are 0 or 1).
    
    Returns:
        tuple: Bounding box ((z_min, z_max), (y_min, y_max), (x_min, x_max)).
    """
    assert mask.ndim == 3, "The input mask must be a 3D array."
    assert np.issubdtype(mask.dtype, np.integer), "The mask must be an integer array."
    
    # Find nonzero indices
    nonzero_indices = np.nonzero(mask)
    
    # Compute the bounding box
    z_min, z_max = nonzero_indices[0].min(), nonzero_indices[0].max()
    y_min, y_max = nonzero_indices[1].min(), nonzero_indices[1].max()
    x_min, x_max = nonzero_indices[2].min(), nonzero_indices[2].max()

    max_dim = np.max([np.abs(z_max - z_min), np.abs(y_max - y_min), np.abs(x_min - x_max)])
    
    return max_dim

def rotate_array_around_center(array, rotation_matrix, order):
    """
    Rotate a 3D numpy array around its center using a given rotation matrix.
    
    Parameters:
    - array: 3D numpy array to rotate.
    - rotation_matrix: 3x3 numpy array representing the rotation matrix.
    
    Returns:
    - rotated_array: 3D numpy array after rotation.
    """
    # Validate inputs
    assert len(array.shape) == 3, "Input array must be 3D."
    assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3."

    # Compute the center of the array
    center = np.array(array.shape) / 2.0

    # Create the affine transformation matrix
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rotation_matrix

    # Translate to origin, apply rotation, and translate back
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -center

    translation_back = np.eye(4)
    translation_back[:3, 3] = center

    # Combine transformations: T_back * R * T_origin
    combined_transform = translation_back @ affine_matrix @ translation_to_origin

    # Apply the transformation using scipy.ndimage.affine_transform
    rotated_array = affine_transform(
        array,
        matrix=combined_transform[:3, :3],
        offset=combined_transform[:3, 3],
        order=order,  # Linear interpolation
        mode="constant",  # Padding mode for out-of-bounds regions
        cval=0  # Value for out-of-bounds regions
    )

    return rotated_array

def calculate_rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Calculate rotation matrix to rotate vertices1 into vertices2

    Args:
        vertices1 (np.ndarray): Origin vertices
        vertices2 (np.ndarray): Target vertices

    Returns:
        np.ndarray: Rotation matrix to rotate vertices1 to vertices2
    """
    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Compute the cross product (axis of rotation)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # If the vectors are already the same, no rotation is needed
    if axis_norm == 0:
        # print("Vectors are the same!")
        return np.eye(3)

    # Normalize the axis of rotation
    axis = axis / axis_norm

    # Compute the angle between the vectors
    cos_theta = np.dot(v1, v2)
    sin_theta = axis_norm  # This is the norm of the cross product

    # Compute the skew-symmetric matrix K
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    # Rodrigues' rotation formula
    R = np.eye(3) + np.sin(np.arccos(cos_theta)) * K + (1 - cos_theta) * np.dot(K, K)

    return R

def create_rotation_matrices_and_graph_toplogy(n_views: int) -> List[np.ndarray]:

    rot_matrix_list = []
    n_views2resolution = {6: 2, 14: 3, 26: 4, 42: 5}

    match n_views:
        case 1:
            rot_matrix_list.append(np.eye(3))
            edge_index = torch.tensor([[0], 
                                       [0]])

        case 3:
            rot_matrix_list.append(np.eye(3))
            rot_matrix_list.append(np.array([[1,  0,  0],
                                             [0,  0, -1],
                                             [0,  1,  0]]))
            rot_matrix_list.append(np.array([[ 0,  0,  1],
                                             [ 0,  1,  0],
                                             [-1,  0,  0]]))
            edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], 
                                       [1, 2, 0, 2, 0, 1]])

        case 6 | 14 | 26 | 42:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=n_views2resolution[n_views])
            sphere.translate(-sphere.get_center())
            vertices = np.asarray(sphere.vertices)    
            faces = torch.from_numpy(np.asarray(sphere.triangles))        
            
            for i in range(n_views):
                rot_matrix = calculate_rotation_matrix(v1=vertices[0], v2=vertices[i])
                rot_matrix_list.append(rot_matrix)   

            # Convert faces to edges (COO format)            
            edge_index = torch.cat([faces[:, [0, 1]],
                                    faces[:, [1, 2]],
                                    faces[:, [2, 0]]], dim=0).t().contiguous()

    return rot_matrix_list, edge_index

def show_biggest_lesion_slice(volume, mask, axis=2, cmap='gray', save_plot=False, patient_id="", pad2square=False, crop2square=True):
    """
    Display a 2D slice of the largest lesion in a 3D volume based on the segmentation mask.
    
    Parameters:
        volume (numpy.ndarray): The 3D volume (e.g., CT or MRI scan).
        mask (numpy.ndarray): The 3D segmentation mask corresponding to the volume.
        axis (int): The axis along which to extract the slice (default is 2 for the last dimension).
        cmap (str): Colormap for visualization (default is 'gray').
    """
    if volume.shape != mask.shape:
        raise ValueError("Volume and mask must have the same shape.")
    if volume.ndim != 3 or mask.ndim != 3:
        raise ValueError("Both volume and mask must be 3D arrays.")
    
    # Label connected components in the segmentation mask
    labeled_mask, num_features = label(mask)
    
    if num_features == 0:
        raise ValueError("No lesions found in the segmentation mask.")
    
    # Identify the largest lesion
    region_sizes = np.array([(labeled_mask == i).sum() for i in range(1, num_features + 1)])
    largest_region_label = np.argmax(region_sizes) + 1
    
    # Compute the centroid of the largest lesion
    centroid = center_of_mass(mask, labeled_mask, largest_region_label)
    
    # Extract the index of the slice along the specified axis closest to the centroid
    slice_idx = int(round(centroid[axis]))
    
    # Extract the 2D slice from both the volume and the mask
    if axis == 0:
        slice_volume = volume[slice_idx, :, :]
        slice_mask = mask[slice_idx, :, :]
    elif axis == 1:
        slice_volume = volume[:, slice_idx, :]
        slice_mask = mask[:, slice_idx, :]
    elif axis == 2:
        slice_volume = volume[:, :, slice_idx]
        slice_mask = mask[:, :, slice_idx]
    else:
        raise ValueError("Invalid axis. Must be 0, 1, or 2.")
    
    if pad2square:
        slice_volume = pad_to_square(slice_volume, padding_value=0)
        slice_mask = pad_to_square(slice_mask, padding_value=0)
    
    if crop2square:
        slice_volume = crop_to_square(slice_volume, slice_mask)
        slice_mask = crop_to_square(slice_mask, slice_mask)

    if save_plot:
        # Display the volume and mask slice side by side
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(slice_volume, cmap=cmap, origin='lower')
        ax[0].set_title(f"Volume Slice (Index {slice_idx} | Shape {slice_volume.shape}) | {patient_id}")
        ax[0].axis('off')
        
        ax[1].imshow(slice_mask, cmap='hot', origin='lower')
        ax[1].set_title(f"Segmentation Mask (Largest Lesion)")
        ax[1].axis('off')
        
        plt.savefig("temp.png")
        plt.close()

    return slice_volume, slice_mask

def pad_to_square(array, padding_value=0):
    """
    Pads a 2D array to make it square by adding padding with the specified value.

    Parameters:
        array (np.ndarray): The input 2D array.
        padding_value: The value used for padding.

    Returns:
        np.ndarray: The padded square array.
    """
    rows, cols = array.shape
    size = max(rows, cols)  # Determine the size for the square matrix

    # Calculate padding for rows and columns
    pad_rows = size - rows
    pad_cols = size - cols

    # Pad the array using np.pad
    padded_array = np.pad(
        array,
        ((0, pad_rows), (0, pad_cols)),  # (top/bottom padding, left/right padding)
        mode='constant',
        constant_values=padding_value
    )
    return padded_array

def crop_to_square(array, mask):
    """
    Crops a 2D array to a square region around the ROI defined by a binary mask.

    Parameters:
        array (np.ndarray): The input 2D array.
        mask (np.ndarray): The binary mask with the ROI (same shape as the array).

    Returns:
        np.ndarray: The cropped square array.
    """
    # Ensure the mask is binary
    mask = mask > 0

    # Find the bounding box of the ROI
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Calculate width and height of the bounding box
    height = row_max - row_min + 1
    width = col_max - col_min + 1

    # Determine the size of the square
    square_size = max(height, width)

    # Center the square around the bounding box
    center_row = (row_min + row_max) // 2
    center_col = (col_min + col_max) // 2

    # Calculate new boundaries for the square
    half_size = square_size // 2
    square_row_min = max(0, center_row - half_size)
    square_row_max = min(array.shape[0], center_row + half_size)
    square_col_min = max(0, center_col - half_size)
    square_col_max = min(array.shape[1], center_col + half_size)

    # Adjust boundaries to ensure the square size
    if square_row_max - square_row_min < square_size:
        if square_row_min == 0:
            square_row_max = min(array.shape[0], square_row_min + square_size)
        else:
            square_row_min = max(0, square_row_max - square_size)
    if square_col_max - square_col_min < square_size:
        if square_col_min == 0:
            square_col_max = min(array.shape[1], square_col_min + square_size)
        else:
            square_col_min = max(0, square_col_max - square_size)

    # Crop the square region
    cropped_array = array[square_row_min:square_row_max, square_col_min:square_col_max]

    return cropped_array


def main(args: argparse.Namespace):

    print(f"[INFO] Creating Dataset with {args.n_views} views")

    # input_list = glob(os.path.join(args.input_folder, "*.nii.gz"))
    # input_list = glob("/home/johannes/Code/MultiViewGCN/data/deep_learning/*/*/*.nii.gz") # all: t1 t2 train test
    input_list = glob("/home/johannes/Code/MultiViewGCN/data/deep_learning/*/*/*.nii.gz") # sarcoma
    input_list = glob("/home/johannes/Code/MultiViewGCN/data/head_and_neck/*/converted_nii/*/*.nii.gz") # head and neck
    assert len(input_list) != 0, "No input volumes available, check path!"

    img_list = sorted([item for item in input_list if not "mask" in item])
    seg_list = sorted([item for item in input_list if "mask" in item])
    assert len(img_list) == len(seg_list)
    print(f"[INFO] Number of images and masks found: {len(img_list)}")

    rot_matrices, graph_edge_index = create_rotation_matrices_and_graph_toplogy(n_views=args.n_views)

    for i in tqdm(range(len(img_list)), desc="Preprocess Data: "):

        patient_id = [img_list[i].split("/")[-1]]
        # patient_id = [temp[:6] if temp.startswith("Sar") else temp[:4] for temp in patient_id][0] # sarcoma
        patient_id = patient_id[0].split("_")[0]

        # try:

        # print(img_list[i])
        # print(seg_list[i])
    
        # Load volume and mask as torchio subject
        subject = tio.Subject(
            img = tio.ScalarImage(img_list[i]),
            seg = tio.LabelMap(seg_list[i])
        )

        # Bring subject into RAS+ orientation
        subject_aligned = tio.ToCanonical()(subject)
        subject_aligned = tio.Resample("img")(subject)

        # Resample subject into target spacing
        subject_resampled = tio.Resample(target=1, image_interpolation=args.interpolation)(subject_aligned)
        # subject_resampled.img.save("img.nii.gz")
        # subject_resampled.seg.save("seg.nii.gz")
        # max_dim = max(subject_resampled.img.shape)
        # target_shape = int(np.sqrt(max_dim*max_dim + max_dim*max_dim))
        # subject_isotropic = tio.CropOrPad(target_shape=target_shape)(subject_resampled)

        non_zero_indices = np.array(np.nonzero(subject_resampled.seg.numpy()[0]))
        min_coords = non_zero_indices.min(axis=1)
        max_coords = non_zero_indices.max(axis=1)
        dimensions = max_coords - min_coords + 1
        max_dimension = dimensions.max()
        target_size_with_margin = int(max_dimension*1.5)

        subject_isotropic = tio.CropOrPad(target_shape=target_size_with_margin, mask_name="seg")(subject_resampled)
        subject_isotropic = tio.CropOrPad(target_shape=int(target_size_with_margin*1.5))(subject_isotropic)

        # Convert subject into numpy array representations
        img_arr = subject_isotropic.img.numpy()[0]
        seg_arr = subject_isotropic.seg.numpy()[0]
        # img_arr = subject_resampled.img.numpy()[0]
        # seg_arr = subject_resampled.seg.numpy()[0]

        # Bring input volume & mask into 3D Slicer representation
        # img_arr = np.rot90(img_arr, k=1, axes=(0, 1))
        # img_arr = np.rot90(img_arr, k=1, axes=(0, 1))
        # img_arr = np.rot90(img_arr, k=1, axes=(0, 1))
        # img_arr = np.fliplr(img_arr)
        # seg_arr = np.rot90(seg_arr, k=1, axes=(0, 1))
        # seg_arr = np.rot90(seg_arr, k=1, axes=(0, 1))
        # seg_arr = np.rot90(seg_arr, k=1, axes=(0, 1))
        # seg_arr = np.fliplr(seg_arr)

        img_arr_list = []
        seg_arr_list = []
        # max_dims = []
        for rot_matrix in rot_matrices:
            img_arr_rotated = rotate_array_around_center(img_arr, rot_matrix, 1)
            seg_arr_rotated = rotate_array_around_center(seg_arr, rot_matrix, 0)
            # max_dim = get_max_dim_mask(seg_arr_rotated)

            img_arr_list.append(img_arr_rotated)
            seg_arr_list.append(seg_arr_rotated)
            # max_dims.append(max_dim)        
        
        # max_dim = np.max(max_dims)

        img_crops = []
        seg_crops = []
        outputs_list = []

        processor = AutoImageProcessor.from_pretrained(args.dinov2_model)
        model = AutoModel.from_pretrained(args.dinov2_model).to(args.device)

        for j in range(len(img_arr_list)):

            subject_new = tio.Subject(
                img = tio.ScalarImage(path=None, tensor=np.expand_dims(img_arr_list[j], axis=0)),
                seg = tio.LabelMap(path=None, tensor=np.expand_dims(seg_arr_list[j], axis=0))
            )

            # cropped images
            # subject_cropped = tio.CropOrPad(target_shape=max_dim, mask_name="seg")(subject_new)
            # subject_cropped = tio.CropOrPad(mask_name="seg")(subject_new)
            # img_crop = subject_cropped.img.numpy()[0]
            # seg_crop = subject_cropped.seg.numpy()[0]
            
            # full images
            img_crop = subject_new.img.numpy()[0]
            seg_crop = subject_new.seg.numpy()[0]

            img_crops.append(img_crop)
            seg_crops.append(seg_crop)

            # img_slice, seg_slice = get_biggest_lesion_slice(volume=img_crop, mask=seg_crop)
            img_slice, seg_slice = show_biggest_lesion_slice(volume=img_crop, mask=seg_crop, axis=2, save_plot=True, patient_id=patient_id, pad2square=False, crop2square=True)
            
            # time.sleep(1)                
            
            img_slice = np.expand_dims(img_slice, axis=-1)
            img_slice = np.repeat(img_slice, 3, axis=-1)
            img_slice = rescale_image(img=img_slice)
            img_slice_pil = Image.fromarray(img_slice)
            inputs = processor(images=img_slice_pil, return_tensors="pt")
            inputs = inputs.pixel_values.to(args.device)
            outputs = model(inputs)
            outputs = model.layernorm(outputs.pooler_output).detach().cpu()
            outputs_list.append(outputs)

        features = torch.concatenate(outputs_list, dim=0)        
        edge_index = to_undirected(graph_edge_index)
        
        # patient_id = [img_list[i].split("/")[-1]]
        # patient_id = [temp[:6] if temp.startswith("Sar") else temp[:4] for temp in patient_id][0]
        
        df = pd.read_csv(args.label_csv)
        # label = df[df["ID"] == patient_id].Grading.item() # sarcoma
        label = df[df["id"] == patient_id].hpv.item() # head and neck
        label = 0 if label == "negative" else 1

        data = Data(x=features, edge_index=edge_index, label=torch.tensor(label))
        model_name = args.dinov2_model.split("/")[-1]
        save_path = seg_list[i].replace("label", "graph-new").replace(".nii.gz", "") + f"_views{args.n_views}_{model_name}.pt"
        torch.save(data, save_path)
        
        # except KeyboardInterrupt:
        #     sys.exit()

        # except Exception as e:
        #     print(f"Failed file: {img_list[i]}")
        #     print(f"An error occurred: {e}")


if __name__ == "__main__":

    # start_time = time.time()

    data_path = "/home/johannes/Code/MultiViewGCN/data/deep_learning/train/T1"
    # label_path = "/home/johannes/Code/MultiViewGCN/data/patient_metadata.csv" # sarcoma
    label_path = "/home/johannes/Code/MultiViewGCN/data/head_and_neck/patient_metadata.csv" # head and neck
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default=data_path, help="Path to folder where volumes and corresponding masks are present.", type=Path)
    parser.add_argument("--n_views", default=3, choices=[1, 3, 6, 14, 26, 42], help="Number of view points to slice input volume.", type=int)
    parser.add_argument("--target_spacing", default=1, help="Target voxel spacing of input data.", type=int)
    parser.add_argument("--interpolation", default="linear", choices=["linear", "bspline"], help="Interpolation type used for resampling.", type=str)
    parser.add_argument("--dinov2_model", default='facebook/dinov2-small', help="Pretrained DINOv2 model architecture.", type=str,
                        choices=['facebook/dinov2-small', 'facebook/dinov2-base', 'facebook/dinov2-large', 'facebook/dinov2-giant'])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Processing device.", type=str)
    parser.add_argument("--label_csv", default=label_path, help="Path to csv file which contains labels.", type=Path)
    args = parser.parse_args()

    for views in [1]:
        args.n_views = views
        main(args)

    # end_time = time.time()
    # print(f"Execution Time: {end_time - start_time:.2f} seconds")           
