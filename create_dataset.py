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
import pyvista as pv
import trimesh
import torch.nn.functional as F
import os

def pad_to_shape(array, target_shape):
    """
    Symmetrically zero-pad a 2D NumPy array to a desired shape.

    Parameters:
    - array: 2D NumPy array to pad
    - target_shape: tuple of (target_rows, target_cols)

    Returns:
    - padded_array: zero-padded array with shape == target_shape
    """
    assert len(array.shape) == 2, "Input must be 2D"
    assert all(ts >= s for ts, s in zip(target_shape, array.shape)), "Target shape must be >= original shape"

    pad_rows = target_shape[0] - array.shape[0]
    pad_cols = target_shape[1] - array.shape[1]

    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    padded_array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return padded_array

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

def rotate_3d_tensor_around_center(tensor, rotation_matrix, order=1, device='cuda'):
    """
    Rotate a 3D torch tensor around its center using a given rotation matrix.

    Parameters:
    - tensor: 3D torch tensor to rotate.
    - rotation_matrix: 3x3 torch tensor representing the rotation matrix.
    - order: Interpolation order (0: nearest, 1: linear).  Defaults to 1.

    Returns:
    - rotated_tensor: 3D torch tensor after rotation.
    """
    # Validate inputs
    if tensor.ndim != 3:
        raise ValueError("Input tensor must be 3D.")
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if order not in [0, 1]:
        raise ValueError("Order must be 0 (nearest) or 1 (linear).")

    # Compute the center of the tensor
    center = torch.tensor(tensor.shape, dtype=torch.float32) / 2.0

    # Create the affine transformation matrix
    affine_matrix = torch.eye(4)
    affine_matrix[:3, :3] = rotation_matrix

    # Translate to origin, apply rotation, and translate back
    translation_to_origin = torch.eye(4)
    translation_to_origin[:3, 3] = -center

    translation_back = torch.eye(4)
    translation_back[:3, 3] = center

    # Combine transformations: T_back * R * T_origin
    combined_transform = translation_back @ affine_matrix @ translation_to_origin

    # Create a meshgrid of coordinates for the original volume
    d_coords = torch.arange(tensor.shape[0], dtype=torch.float32)
    h_coords = torch.arange(tensor.shape[1], dtype=torch.float32)
    w_coords = torch.arange(tensor.shape[2], dtype=torch.float32)
    grid = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
    coords = torch.stack(grid, dim=-1)  # Shape: (D, H, W, 3)

    # Reshape to (D*H*W, 3) for matrix multiplication
    original_coords_flat = coords.reshape(-1, 3)

    # Add a homogeneous coordinate (1) to each point
    ones = torch.ones(original_coords_flat.shape[0], 1)
    original_coords_homogeneous = torch.cat((original_coords_flat, ones), dim=1)  # (D*H*W, 4)

    # Apply the inverse transformation to get source coordinates
    # We use the inverse because grid_sample samples *from* the input
    # at locations given by the output (transformed) coordinates.
    transformed_coords_homogeneous = original_coords_homogeneous @ torch.inverse(combined_transform).T

    # Extract the spatial coordinates (x, y, z)
    transformed_coords = transformed_coords_homogeneous[:, :3]

    # Normalize to the range [-1, 1] for grid_sample
    normalized_coords_d = 2 * transformed_coords[:, 0] / (tensor.shape[0] - 1) - 1
    normalized_coords_h = 2 * transformed_coords[:, 1] / (tensor.shape[1] - 1) - 1
    normalized_coords_w = 2 * transformed_coords[:, 2] / (tensor.shape[2] - 1) - 1

    # Create the sampling grid for grid_sample
    sampling_grid = torch.stack((normalized_coords_w, normalized_coords_h, normalized_coords_d), dim=-1)
    sampling_grid = sampling_grid.reshape(1, tensor.shape[0], tensor.shape[1], tensor.shape[2], 3)

    # Use grid_sample to perform the rotation
    mode = 'bilinear' if order == 1 else 'nearest'
    rotated_tensor = F.grid_sample(
        tensor.unsqueeze(0).unsqueeze(0),  # Add batch dimension for grid_sample
        sampling_grid.to(device),
        mode=mode,
        padding_mode='zeros',
        align_corners=True
    ).squeeze(0) # Remove batch dimension

    return rotated_tensor.squeeze(0).cpu().numpy()  # Remove channel dimension

# def rotate_3d_tensor(volume, rotation_matrix, mode='bilinear'):
#     """
#     Rotate a 3D volume around its center using a 3x3 rotation matrix.
    
#     Args:
#         volume (torch.Tensor): 3D tensor of shape (D, H, W), must be float32 and on GPU.
#         rotation_matrix (torch.Tensor): 3x3 rotation matrix, must be float32 and on GPU.
        
#     Returns:
#         torch.Tensor: Rotated volume of the same shape.
#     """

#     if not isinstance(volume, torch.Tensor):
#         volume = torch.tensor(volume, dtype=torch.float32).cuda()

#     if not isinstance(rotation_matrix, torch.Tensor):
#         rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32).cuda()

#     device = volume.device
#     D, H, W = volume.shape
#     volume = volume.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]
    
#     # Compute center of the volume
#     center = torch.tensor([D/2, H/2, W/2], device=device)

#     # Build the full 3x4 affine matrix with translation to rotate about center
#     # Affine transform is applied in normalized coordinates [-1, 1]
#     affine = torch.eye(4, device=device)
#     affine[:3, :3] = rotation_matrix

#     # Translate center to origin -> apply rotation -> translate back
#     T1 = torch.eye(4, device=device)
#     T1[:3, 3] = -center

#     T2 = torch.eye(4, device=device)
#     T2[:3, 3] = center

#     transform = T2 @ affine @ T1
#     affine_3x4 = transform[:3, :]  # final affine matrix for grid generation

#     # Convert affine to normalized coordinates
#     norm_affine = affine_3x4.clone()
#     norm_affine[:, 0] /= (D - 1) / 2
#     norm_affine[:, 1] /= (H - 1) / 2
#     norm_affine[:, 2] /= (W - 1) / 2

#     # Create affine grid and sample
#     grid = F.affine_grid(norm_affine.unsqueeze(0), volume.shape, align_corners=True)
#     rotated = F.grid_sample(volume, grid, mode=mode, padding_mode="border", align_corners=True)

#     return rotated.squeeze(0).squeeze(0).cpu().numpy()  # Shape: [D, H, W]

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

def show_biggest_lesion_slice(volume, mask, axis=2, cmap='gray', save_plot=False, patient_id="", pad2square=False, crop2square=True, medmnist=False):
    """
    Display a 2D slice of the largest lesion in a 3D volume based on the segmentation mask.
    
    Parameters:
        volume (numpy.ndarray): The 3D volume (e.g., CT or MRI scan).
        mask (numpy.ndarray): The 3D segmentation mask corresponding to the volume.
        axis (int): The axis along which to extract the slice (default is 2 for the last dimension).
        cmap (str): Colormap for visualization (default is 'gray').
    """

    if not medmnist:

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
    
    else:
        middle_slice = volume.shape[-1]//2
        slice_volume = volume[:, :, middle_slice]
        slice_mask = mask[:, :, middle_slice]

    if save_plot:
        # Display the volume and mask slice side by side
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(slice_volume, cmap=cmap, origin='lower')
        ax[0].set_title(f"Volume Slice | Shape {slice_volume.shape} | Patient {patient_id}")
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
    square_size = int(square_size * 1.1)

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

def create_fibonacci_sphere(n_vertices: int, save_sphere: bool = False):

    if n_vertices == 1:
        vertices = np.random.randn(1, 3)
        faces = []

    elif n_vertices == 3:
        vertices = np.random.randn(3, 3)
        faces = []

    else:    
        assert n_vertices >= 8, "Please make sure number of views is greater than 8 or equal to 1 or 3!"
        indices = np.arange(0, n_vertices, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_vertices)
        theta = np.pi * (1 + 5**0.5) * indices

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points = np.vstack((x, y, z)).T

        point_cloud = pv.PolyData(points)
        mesh = point_cloud.delaunay_3d()
        surf = mesh.extract_surface()
        vertices = surf.points
        faces = surf.faces.reshape(-1, 4)[:, 1:]

        if save_sphere:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces) 
            mesh.export("mesh.ply")


    return vertices, faces

def create_rotation_matrices(vertices: np.ndarray):

    rot_matrices = []
    if vertices.shape[0] == 1:
        pass
    
    elif vertices.shape[0] == 3:
        rot_matrices.append(np.array([[ 0.,  0., -1.],
                                      [ 0.,  1.,  0.],
                                      [ 1.,  0.,  0.]]))
        
        rot_matrices.append(np.array([[ 1.,  0.,  0.],
                                      [ 0.,  0., -1.],
                                      [ 0.,  1.,  0.]]))        
                                    
    else:
        assert vertices.shape[0] >= 8, "Please make sure number of views is greater than 8 or equal to 1 or 3!"
    
        origin = np.array(vertices[0])

        
        for i in range(1, len(vertices)):
            rot_matrices.append(calculate_rotation_matrix(origin, np.array(vertices[i])))
        
    return rot_matrices

def create_graph_topology(vertices: np.ndarray, faces: np.ndarray):

    if vertices.shape[0] == 1:
        adjacency_matrix = np.ones(1)
        edge_index = torch.tensor([[0], 
                                   [0]])
        edge_index = to_undirected(edge_index=edge_index)

    elif vertices.shape[0] == 3:
        adjacency_matrix = np.ones((3, 3))
        np.fill_diagonal(adjacency_matrix, 0)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], 
                                   [1, 2, 0, 2, 0, 1]])
        edge_index = to_undirected(edge_index=edge_index)

    else:    
        assert vertices.shape[0] >= 8, "Please make sure number of views is greater than 8 or equal to 1 or 3!"   
        
        # Convert faces to dense adjacency matrix
        num_vertices = vertices.shape[0]
        adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=np.float32)

        for face in faces:
            i, j, k = face
            # Add edges between vertices of each triangle
            adjacency_matrix[i, j] = 1  # Edge between i and j
            adjacency_matrix[j, i] = 1  # Symmetric edge
            adjacency_matrix[j, k] = 1  # Edge between j and k
            adjacency_matrix[k, j] = 1  # Symmetric edge
            adjacency_matrix[k, i] = 1  # Edge between k and i
            adjacency_matrix[i, k] = 1  # Symmetric edge

        # Convert faces to edges (COO format) 
        faces_tensor = torch.from_numpy(np.copy(faces))           
        edge_index = torch.cat([faces_tensor[:, [0, 1]],
                                faces_tensor[:, [1, 2]],
                                faces_tensor[:, [2, 0]]], dim=0).t().contiguous()    
        edge_index = to_undirected(edge_index=edge_index)

    return adjacency_matrix, edge_index

def encode_largest_lesion_slice(volume: np.array, mask: np.array, image_processor, backbone, patient_id, dataset_name, axis=2):

    if dataset_name in ["sarcoma_t1", "sarcoma_t2", "headneck"]:
        img_slice, seg_slice = show_biggest_lesion_slice(volume=volume, mask=mask, axis=axis, save_plot=True, patient_id=patient_id, pad2square=False, crop2square=True, medmnist=False)
    
    else:
        img_slice, seg_slice = show_biggest_lesion_slice(volume=volume, mask=mask, axis=axis, save_plot=True, patient_id=patient_id, pad2square=False, crop2square=True, medmnist=True)

    img_slice = np.expand_dims(img_slice, axis=-1)
    img_slice = np.repeat(img_slice, 3, axis=-1)
    img_slice = rescale_image(img=img_slice)
    img_slice_pil = Image.fromarray(img_slice)
    inputs = image_processor(images=img_slice_pil, return_tensors="pt")
    inputs = inputs.pixel_values.to("cuda")
    outputs = backbone(inputs)
    outputs = backbone.layernorm(outputs.pooler_output).detach().cpu()
    
    return outputs, img_slice

def extract_foreground_slices(volume, mask, processor, model, save_to_file=False):
    """
    Extracts all 2D slices along the last dimension of a 3D volume
    where the corresponding binary mask has at least one foreground pixel.

    Parameters:
    - volume: 3D NumPy array (e.g., shape (H, W, D))
    - mask:   3D NumPy array, same shape as volume

    Returns:
    - slices: List of 2D arrays from volume
    - indices: List of slice indices that were extracted
    """
    assert volume.shape == mask.shape, "Volume and mask must have the same shape"
    
    slices = []
    indices = []

    for i in range(volume.shape[2]):  # iterate over last axis (e.g., D)
        if np.any(mask[:, :, i] > 0):
            slices.append(volume[:, :, i])
            indices.append(i)

    outputs_list = []
    for idx, slice in enumerate(slices):
        img_slice_temp = np.expand_dims(slice, axis=-1)

        img_slice = np.concatenate([img_slice_temp, img_slice_temp, img_slice_temp], axis=-1)
        img_slice = rescale_image(img=img_slice)
        img_slice_pil = Image.fromarray(img_slice)

        if save_to_file:
            img_slice_pil.save(f"/home/johannes/Data/SSD_1.9TB/MultiViewGCN/PIL_image_slices/slice_{str(idx).zfill(3)}.png")

        inputs = processor(images=img_slice_pil, return_tensors="pt")
        inputs = inputs.pixel_values.to("cuda")
        outputs = model(inputs)
        outputs = model.layernorm(outputs.pooler_output).detach().cpu()

        outputs_list.append(outputs)  

    features = torch.concatenate(outputs_list, dim=0)   

    return features

# @profile
def main(args: argparse.Namespace):

    dataset_name = str(args.dataset)
    model_name = args.dinov2_model.split("/")[-1]

    print(f"\n[INFO] Creating {args.dataset} dataset with {args.n_views} views using {args.dinov2_model} model")
    
    if str(args.dataset) == "sarcoma_t1":
        input_list = glob("./data/sarcoma/*/T1/*.nii.gz")
        img_list = sorted([item for item in input_list if not "label" in item])
        seg_list = sorted([item for item in input_list if "label" in item])
        args.label_csv = "./data/sarcoma/patient_metadata.csv"
    elif str(args.dataset) == "sarcoma_t2":
        input_list = glob("./data/sarcoma/*/T2/*.nii.gz")
        img_list = sorted([item for item in input_list if not "label" in item])
        seg_list = sorted([item for item in input_list if "label" in item])
        args.label_csv = "./data/sarcoma/patient_metadata.csv"
    elif str(args.dataset) == "headneck":
        input_list = glob("./data/headneck/converted_nii_merged/*/*.nii.gz")
        img_list = sorted([item for item in input_list if not "mask" in item]) 
        seg_list = sorted([item for item in input_list if "mask" in item]) 
        args.label_csv = "./data/headneck/patient_metadata_filtered.csv"
    elif str(args.dataset) == "adrenal":
        input_list = glob("./data/medmnist3d/adrenalmnist3d_64/adrenal*image*.npy")
        img_list = sorted(input_list) 
        seg_list = img_list
    elif str(args.dataset) == "nodule":
        input_list = glob("./data/medmnist3d/nodulemnist3d_64/nodule*image*.npy")
        img_list = sorted(input_list) 
        seg_list = img_list
    elif str(args.dataset) == "synapse":
        input_list = glob("./data/medmnist3d/synapsemnist3d_64/synapse*image*.npy")
        img_list = sorted(input_list) 
        seg_list = img_list
    elif str(args.dataset) == "vessel":
        input_list = glob("./data/medmnist3d/vesselmnist3d_64/vessel*image*.npy")
        img_list = sorted(input_list) 
        seg_list = img_list
    elif str(args.dataset) == "glioma_t1":
        img_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*T1_bias.nii.gz"))
        seg_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*segmentation.nii.gz"))
        args.label_csv = "/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/UCSF-PDGM-metadata_v2.csv"
    elif str(args.dataset) == "glioma_t1c":
        img_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*T1c_bias.nii.gz"))
        seg_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*segmentation.nii.gz"))
        args.label_csv = "/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/UCSF-PDGM-metadata_v2.csv"
    elif str(args.dataset) == "glioma_t2":
        img_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*T2_bias.nii.gz"))
        seg_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*segmentation.nii.gz"))
        args.label_csv = "/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/UCSF-PDGM-metadata_v2.csv"
    elif str(args.dataset) == "glioma_flair":
        img_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*FLAIR_bias.nii.gz"))
        seg_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*segmentation.nii.gz"))
        args.label_csv = "/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/UCSF-PDGM-metadata_v2.csv"
    elif str(args.dataset) == "breast":
        img_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/breast/duke_tumor_grading/DUKE_*_0000.nii.gz"))
        seg_list = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/breast/duke_tumor_grading/DUKE_*segmentation.nii.gz"))
        args.label_csv = "/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/breast/clinical_and_imaging_info.xlsx"
    else:
        raise NotImplementedError(f"Given dataset name {args.dataset} is not implemented!")
        
    assert len(img_list) != 0, "No input volumes available, check path!"   
    assert len(img_list) == len(seg_list), "Number of images and segmentation masks are not the same!"
    print(f"[INFO] Number of images and masks found: {len(img_list)}")

    if args.n_views != "all_axial":
        if args.n_views > 3:

            vertices, faces = create_fibonacci_sphere(n_vertices=args.n_views, save_sphere=False)
            rot_matrices = create_rotation_matrices(vertices=vertices)
            adjacency_matrix, graph_edge_index = create_graph_topology(vertices=vertices, faces=faces)
    
        else:
            rot_matrices = []
            edge_index = None
            edge_attr = None
    
    else:
            rot_matrices = []
            edge_index = None
            edge_attr = None

    processor = AutoImageProcessor.from_pretrained(args.dinov2_model, use_fast=False)
    model = AutoModel.from_pretrained(args.dinov2_model).to("cuda")

    for i in tqdm(range(len(img_list)), desc="Preprocess Data: "):

        if str(args.dataset) in ["sarcoma_t1", "sarcoma_t2", "headneck", "glioma_t1", "glioma_t1c", "glioma_t2", "glioma_flair", "breast"]:
            
            if str(args.dataset) == "sarcoma_t1":
                save_path = seg_list[i].replace("label", "graph-fibonacci-edge_attr").replace(".nii.gz", "") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = [img_list[i].split("/")[-1]]
                patient_id = [temp[:6] if temp.startswith("Sar") else temp[:4] for temp in patient_id][0]
            
            elif str(args.dataset) == "sarcoma_t2":
                save_path = seg_list[i].replace("label", "graph-fibonacci-edge_attr").replace(".nii.gz", "") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = [img_list[i].split("/")[-1]]
                patient_id = [temp[:6] if temp.startswith("Sar") else temp[:4] for temp in patient_id][0]
            
            elif str(args.dataset) == "headneck":
                save_path = seg_list[i].replace("mask", "graph-fibonacci-edge_attr").replace(".nii.gz", "") + f"_views{args.n_views}_{model_name}.pt" 
                patient_id = [img_list[i].split("/")[-1]]
                patient_id = patient_id[0].split("_")[0]   

            elif str(args.dataset) == "glioma_t1":
                save_path = img_list[i].replace(".nii.gz", "_graph-fibonacci-edge_attr") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = os.path.basename(img_list[i]).split("_")[0]
                patient_id = patient_id.split("-")[0] + "-" + patient_id.split("-")[1] + "-" + patient_id.split("-")[2][1:]
            
            elif str(args.dataset) == "glioma_t1c":
                save_path = img_list[i].replace(".nii.gz", "_graph-fibonacci-edge_attr") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = os.path.basename(img_list[i]).split("_")[0]     
                patient_id = patient_id.split("-")[0] + "-" + patient_id.split("-")[1] + "-" + patient_id.split("-")[2][1:]         
            
            elif str(args.dataset) == "glioma_t2":
                save_path = img_list[i].replace(".nii.gz", "_graph-fibonacci-edge_attr") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = os.path.basename(img_list[i]).split("_")[0]  
                patient_id = patient_id.split("-")[0] + "-" + patient_id.split("-")[1] + "-" + patient_id.split("-")[2][1:]            
            
            elif str(args.dataset) == "glioma_flair":
                save_path = img_list[i].replace(".nii.gz", "_graph-fibonacci-edge_attr") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = os.path.basename(img_list[i]).split("_")[0]  
                patient_id = patient_id.split("-")[0] + "-" + patient_id.split("-")[1] + "-" + patient_id.split("-")[2][1:]      
            
            elif str(args.dataset) == "breast":
                save_path = img_list[i].replace(".nii.gz", "_graph-fibonacci-edge_attr") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = os.path.basename(img_list[i]).replace(".nii.gz", "")[:8]
                   
            subject = tio.Subject(
                img = tio.ScalarImage(img_list[i]),
                seg = tio.LabelMap(seg_list[i])
            )
  
            subject_aligned = tio.ToCanonical()(subject)
            subject_resampled = tio.Resample("img")(subject_aligned)
            subject_iso_resampled = tio.Resample(target=args.target_spacing, image_interpolation=args.interpolation)(subject_resampled)

            non_zero_indices = np.array(np.nonzero(subject_iso_resampled.seg.numpy()[0]))
            min_coords = non_zero_indices.min(axis=1)
            max_coords = non_zero_indices.max(axis=1)
            dimensions = max_coords - min_coords + 1
            max_dimension = dimensions.max()          

            if max_dimension > 300:
                print(f"\n [INFO] Very big ROI ({max_dimension}): {seg_list[i]}")                

            # target_size_iso_crop = int(max_dimension*1.5)
            target_size_iso_crop = max_dimension
            # target_size_iso_pad = int(target_size_iso_crop*1.5)
            target_size_iso_pad = int(target_size_iso_crop*np.sqrt(2))

            subject_iso_crop = tio.CropOrPad(target_shape=target_size_iso_crop, mask_name="seg")(subject_iso_resampled)
            subject_iso_pad = tio.CropOrPad(target_shape=target_size_iso_pad)(subject_iso_crop)
        
        else: 
            save_path = img_list[i].replace("image", "graph-fibonacci-edge_attr").replace(".npy", "") + f"_views{args.n_views}_{model_name}.pt" 
            patient_id = [img_list[i].split("/")[-1]]
            patient_id = patient_id[0].split("_")[-1].replace(".npy", "")
            img_arr = np.load(img_list[i])
            seg_arr = np.load(seg_list[i])

            subject = tio.Subject(
                img = tio.ScalarImage(path=None, tensor=np.expand_dims(img_arr, axis=0)),
                seg = tio.LabelMap(path=None, tensor=np.expand_dims(seg_arr, axis=0))
            )

            subject_iso_pad = tio.CropOrPad(target_shape=int(64*1.5))(subject)

        # if os.path.exists(save_path):
        #     print(f"[INFO] File already exists: {save_path}")
        #     continue

        img_arr = subject_iso_pad.img.numpy()[0]
        seg_arr = subject_iso_pad.seg.numpy()[0]

        if args.n_views == "all_axial":
            features = extract_foreground_slices(volume=img_arr, mask=seg_arr, processor=processor, model=model, save_to_file=True)

        else:

            outputs_list = []
            encoding, _ = encode_largest_lesion_slice(volume=img_arr, mask=seg_arr, 
                                                image_processor=processor, backbone=model, 
                                                patient_id=patient_id, dataset_name=dataset_name)
            outputs_list.append(encoding)

            # if args.n_views == 3:
            #     encoding, _ = encode_largest_lesion_slice(volume=img_arr, mask=seg_arr, 
            #                                            image_processor=processor, backbone=model, 
            #                                            patient_id=patient_id, dataset_name=dataset_name, axis=0)
            #     outputs_list.append(encoding)

            #     encoding, _ = encode_largest_lesion_slice(volume=img_arr, mask=seg_arr, 
            #                                            image_processor=processor, backbone=model, 
            #                                            patient_id=patient_id, dataset_name=dataset_name, axis=1)
            #     outputs_list.append(encoding)
            
            if args.n_views == 3:
                encoding, img_slice = encode_largest_lesion_slice(volume=img_arr, mask=seg_arr, 
                                                    image_processor=processor, backbone=model, 
                                                    patient_id=patient_id, dataset_name=dataset_name, axis=0)     

                img_slice_0 = img_slice[:, :, 0]
                img_slice_0_shape = img_slice_0.shape[0]      

                encoding, img_slice = encode_largest_lesion_slice(volume=img_arr, mask=seg_arr, 
                                                    image_processor=processor, backbone=model, 
                                                    patient_id=patient_id, dataset_name=dataset_name, axis=1)         
                img_slice_1 = img_slice[:, :, 0]  
                img_slice_1_shape = img_slice_1.shape[0] 
                
                encoding, img_slice = encode_largest_lesion_slice(volume=img_arr, mask=seg_arr, 
                                                    image_processor=processor, backbone=model, 
                                                    patient_id=patient_id, dataset_name=dataset_name, axis=2)
                img_slice_2 = img_slice[:, :, 0]  
                img_slice_2_shape = img_slice_2.shape[0]  

                target_shape = np.max([img_slice_0_shape, img_slice_1_shape, img_slice_2_shape])

                img_slice_0_pad = pad_to_shape(img_slice_0, target_shape=(target_shape, target_shape))
                img_slice_1_pad = pad_to_shape(img_slice_1, target_shape=(target_shape, target_shape))
                img_slice_2_pad = pad_to_shape(img_slice_2, target_shape=(target_shape, target_shape))
                img_slice_0 = np.expand_dims(img_slice_0_pad, axis=-1)
                img_slice_1 = np.expand_dims(img_slice_1_pad, axis=-1)
                img_slice_2 = np.expand_dims(img_slice_2_pad, axis=-1)

                img_slice = np.concatenate([img_slice_0, img_slice_1, img_slice_2], axis=-1)
                img_slice_pil = Image.fromarray(img_slice)
                inputs = processor(images=img_slice_pil, return_tensors="pt")
                inputs = inputs.pixel_values.to("cuda")
                outputs = model(inputs)
                outputs = model.layernorm(outputs.pooler_output).detach().cpu()

                outputs_list = []
                outputs_list.append(outputs)            


            for rot_matrix in tqdm(rot_matrices):
                # img_arr_rotated = rotate_array_around_center(img_arr, rot_matrix, 1)
                img_arr_rotated = rotate_3d_tensor_around_center(torch.tensor(img_arr, dtype=torch.float32).cuda(), torch.tensor(rot_matrix, dtype=torch.float32).cuda(), order=1, device='cuda')
                # seg_arr_rotated = rotate_array_around_center(seg_arr, rot_matrix, 0)
                seg_arr_rotated = rotate_3d_tensor_around_center(torch.tensor(seg_arr, dtype=torch.float32).cuda(), torch.tensor(rot_matrix, dtype=torch.float32).cuda(), order=0, device='cuda')

                
                encoding, _ = encode_largest_lesion_slice(volume=img_arr_rotated, mask=seg_arr_rotated, 
                                                    image_processor=processor, backbone=model, 
                                                    patient_id=patient_id, dataset_name=dataset_name)
                
                outputs_list.append(encoding)      

            features = torch.concatenate(outputs_list, dim=0)   

            if args.n_views > 3:     
                edge_index = to_undirected(graph_edge_index)

                rot_matrices_temp = [np.eye(3)] + rot_matrices

                edge_attr = []
                for i in range(edge_index.shape[-1]):
                    edge = edge_index[:, i]
                    source = edge[0]
                    target = edge[1]
                    source_rot_matrix = rot_matrices_temp[source]
                    target_rot_matrix = rot_matrices_temp[target]
                    transition_matrix = target_rot_matrix @ source_rot_matrix.T
                    edge_attr.append(torch.from_numpy(transition_matrix.flatten()))
                edge_attr = torch.stack(edge_attr, dim=0)
        
        if dataset_name in ["sarcoma_t1", "sarcoma_t2", "headneck", "glioma_t1", "glioma_t1c", "glioma_t2", "glioma_flair"]:
            df = pd.read_csv(args.label_csv)
            
            if dataset_name == "sarcoma_t1":
                label = df[df["ID"] == patient_id].Grading.item() 
                label = 0 if label == 1 else 1
            elif dataset_name == "sarcoma_t2":
                label = df[df["ID"] == patient_id].Grading.item() 
                label = 0 if label == 1 else 1
            elif dataset_name == "glioma_t1":
                label = df[df["ID"] == patient_id]["WHO CNS Grade"].item()
                label = 0 if label < 4 else 1
            elif dataset_name == "glioma_t1c":
                label = df[df["ID"] == patient_id]["WHO CNS Grade"].item()
                label = 0 if label < 4 else 1
            elif dataset_name == "glioma_t2":
                label = df[df["ID"] == patient_id]["WHO CNS Grade"].item()
                label = 0 if label < 4 else 1
            elif dataset_name == "glioma_flair":
                label = df[df["ID"] == patient_id]["WHO CNS Grade"].item()
                label = 0 if label < 4 else 1
            elif dataset_name == "headneck":
                label = df[df["id"] == patient_id].hpv.item() 
                label = 0 if label == "negative" else 1
            else:
                raise NotImplementedError(f"Dataset {dataset_name} is not implemented for label extraction!")
        
        elif dataset_name in ["breast"]:
            df = pd.read_excel(args.label_csv)
            label = df[df["patient_id"] == patient_id]["nottingham_grade"].item()
            label = 1 if label == "high" else 0
        
        else:
            label = np.load(img_list[i].replace("image", "label")).item()

        # data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr, adj_matrix=adjacency_matrix, label=torch.tensor(label))        
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr, label=torch.tensor(label))        
        torch.save(data, save_path)
     
if __name__ == "__main__":    

    for dataset in ["sarcoma_t1", "sarcoma_t2", "glioma_t1c", "glioma_flair", "breast"]:
        # for views in [1, 3, 8, 12, 16, 20, 24]:
        for views in ["all_axial"]:
            
            parser = argparse.ArgumentParser()
            parser.add_argument("--dataset", default="sarcoma_t1", help="Name of dataset to be processed.", type=Path,
                                choices=["sarcoma_t1", "sarcoma_t2", "headneck", "glioma_t1c", "glioma_flair", "breast"], )
            parser.add_argument("--n_views", default=3, choices=[1, 3, 8, 12, 16, 20, 24], help="Number of view points to slice input volume.", type=int)
            parser.add_argument("--target_spacing", default=1, help="Target voxel spacing of input data.", type=int)
            parser.add_argument("--interpolation", default="linear", choices=["linear", "bspline"], help="Interpolation type used for resampling.", type=str)
            parser.add_argument("--dinov2_model", default='facebook/dinov2-small', help="Pretrained DINOv2 model architecture.", type=str,
                                choices=['facebook/dinov2-small', 'facebook/dinov2-base', 'facebook/dinov2-large', 'facebook/dinov2-giant'])
            parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Processing device.", type=str)
            args = parser.parse_args()

            args.dataset = dataset
            args.n_views = views
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            main(args)  
