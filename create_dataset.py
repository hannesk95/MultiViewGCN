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
    
    origin = np.array(vertices[0])

    rot_matrices = []
    for i in range(1, len(vertices)):
        rot_matrices.append(calculate_rotation_matrix(origin, np.array(vertices[i])))
    
    return rot_matrices

def create_graph_topology(vertices: np.ndarray, faces: np.ndarray):   
        
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

def encode_largest_lesion_slice(volume: np.array, mask: np.array, image_processor, backbone, patient_id, dataset_name):

    if dataset_name in ["sarcoma_t1", "sarcoma_t2", "headneck"]:
        img_slice, seg_slice = show_biggest_lesion_slice(volume=volume, mask=mask, axis=2, save_plot=True, patient_id=patient_id, pad2square=False, crop2square=True, medmnist=False)
    
    else:
        img_slice, seg_slice = show_biggest_lesion_slice(volume=volume, mask=mask, axis=2, save_plot=True, patient_id=patient_id, pad2square=False, crop2square=True, medmnist=True)
    
    img_slice = np.expand_dims(img_slice, axis=-1)
    img_slice = np.repeat(img_slice, 3, axis=-1)
    img_slice = rescale_image(img=img_slice)
    img_slice_pil = Image.fromarray(img_slice)
    inputs = image_processor(images=img_slice_pil, return_tensors="pt")
    inputs = inputs.pixel_values.to(args.device)
    outputs = backbone(inputs)
    outputs = backbone.layernorm(outputs.pooler_output).detach().cpu()
    
    return outputs

def main(args: argparse.Namespace):

    dataset_name = str(args.dataset)
    model_name = args.dinov2_model.split("/")[-1]

    print(f"\n[INFO] Creating {args.dataset} dataset with {args.n_views} views using {args.dinov2_model} model")
    
    if str(args.dataset) == "sarcoma_t1":
        input_list = glob("/home/johannes/Code/MultiViewGCN/data/sarcoma/*/T1/*.nii.gz")
        img_list = sorted([item for item in input_list if not "label" in item])
        seg_list = sorted([item for item in input_list if "label" in item])
        args.label_csv = "/home/johannes/Code/MultiViewGCN/data/sarcoma/patient_metadata.csv"
    elif str(args.dataset) == "sarcoma_t2":
        input_list = glob("/home/johannes/Code/MultiViewGCN/data/sarcoma/*/T2/*.nii.gz")
        img_list = sorted([item for item in input_list if not "label" in item])
        seg_list = sorted([item for item in input_list if "label" in item])
        args.label_csv = "/home/johannes/Code/MultiViewGCN/data/sarcoma/patient_metadata.csv"
    elif str(args.dataset) == "headneck":
        input_list = glob("/home/johannes/Code/MultiViewGCN/data/headneck/*/converted_nii/*/*.nii.gz")
        img_list = sorted([item for item in input_list if not "mask" in item]) 
        seg_list = sorted([item for item in input_list if "mask" in item]) 
        args.label_csv = "/home/johannes/Code/MultiViewGCN/data/headneck/patient_metadata.csv"
    elif str(args.dataset) == "adrenal":
        input_list = glob("/home/johannes/Code/MultiViewGCN/data/medmnist3d/adrenalmnist3d_64/adrenal*image*.npy")
        img_list = sorted(input_list) 
        seg_list = img_list
    elif str(args.dataset) == "nodule":
        input_list = glob("/home/johannes/Code/MultiViewGCN/data/medmnist3d/nodulemnist3d_64/nodule*image*.npy")
        img_list = sorted(input_list) 
        seg_list = img_list
    elif str(args.dataset) == "synapse":
        input_list = glob("/home/johannes/Code/MultiViewGCN/data/medmnist3d/synapsemnist3d_64/synapse*image*.npy")
        img_list = sorted(input_list) 
        seg_list = img_list
    elif str(args.dataset) == "vessel":
        input_list = glob("/home/johannes/Code/MultiViewGCN/data/medmnist3d/vesselmnist3d_64/vessel*image*.npy")
        img_list = sorted(input_list) 
        seg_list = img_list
    else:
        raise NotImplementedError(f"Given dataset name {args.dataset} is not implemented!")
        
    assert len(input_list) != 0, "No input volumes available, check path!"   
    assert len(img_list) == len(seg_list), "Number of images and segmentation masks are not the same!"
    print(f"[INFO] Number of images and masks found: {len(img_list)}")

    vertices, faces = create_fibonacci_sphere(n_vertices=args.n_views, save_sphere=False)
    rot_matrices = create_rotation_matrices(vertices=vertices)
    adjacency_matrix, graph_edge_index = create_graph_topology(vertices=vertices, faces=faces)

    processor = AutoImageProcessor.from_pretrained(args.dinov2_model)
    model = AutoModel.from_pretrained(args.dinov2_model).to(args.device)


    for i in tqdm(range(len(img_list)), desc="Preprocess Data: "):

        if str(args.dataset) in ["sarcoma_t1", "sarcoma_t2", "headneck"]:
            
            if str(args.dataset) == "sarcoma_t1":
                save_path = seg_list[i].replace("label", "graph-fibonacci").replace(".nii.gz", "") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = [img_list[i].split("/")[-1]]
                patient_id = [temp[:6] if temp.startswith("Sar") else temp[:4] for temp in patient_id][0]
            elif str(args.dataset) == "sarcoma_t2":
                save_path = seg_list[i].replace("label", "graph-fibonacci").replace(".nii.gz", "") + f"_views{args.n_views}_{model_name}.pt"
                patient_id = [img_list[i].split("/")[-1]]
                patient_id = [temp[:6] if temp.startswith("Sar") else temp[:4] for temp in patient_id][0]
            elif str(args.dataset) == "headneck":
                save_path = seg_list[i].replace("mask", "graph-fibonacci").replace(".nii.gz", "") + f"_views{args.n_views}_{model_name}.pt" 
                patient_id = [img_list[i].split("/")[-1]]
                patient_id = patient_id[0].split("_")[0]        
        
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

            target_size_iso_crop = int(max_dimension*1.5)
            target_size_iso_pad = int(target_size_iso_crop*1.5)

            subject_iso_crop = tio.CropOrPad(target_shape=target_size_iso_crop, mask_name="seg")(subject_iso_resampled)
            subject_iso_pad = tio.CropOrPad(target_shape=target_size_iso_pad)(subject_iso_crop)
       
            img_arr = subject_iso_pad.img.numpy()[0]  
            seg_arr = subject_iso_pad.seg.numpy()[0]
        
        else: 
            patient_id = [img_list[i].split("/")[-1]]
            patient_id = patient_id[0].split("_")[-1].replace(".npy", "")
            img_arr = np.load(img_list[i])
            seg_arr = np.load(seg_list[i])

            subject = tio.Subject(
                img = tio.ScalarImage(path=None, tensor=np.expand_dims(img_arr, axis=0)),
                seg = tio.LabelMap(path=None, tensor=np.expand_dims(seg_arr, axis=0))
            )

            subject_iso_pad = tio.CropOrPad(target_shape=int(64*1.5))(subject)

            img_arr = subject_iso_pad.img.numpy()[0]
            seg_arr = subject_iso_pad.seg.numpy()[0]
    

        
        outputs_list = []
        encoding = encode_largest_lesion_slice(volume=img_arr, mask=seg_arr, 
                                               image_processor=processor, backbone=model, 
                                               patient_id=patient_id, dataset_name=dataset_name)
        outputs_list.append(encoding)
        for rot_matrix in rot_matrices:
            img_arr_rotated = rotate_array_around_center(img_arr, rot_matrix, 1)
            seg_arr_rotated = rotate_array_around_center(seg_arr, rot_matrix, 0)  
            
            encoding = encode_largest_lesion_slice(volume=img_arr_rotated, mask=seg_arr_rotated, 
                                                   image_processor=processor, backbone=model, 
                                                   patient_id=patient_id, dataset_name=dataset_name)
            
            outputs_list.append(encoding)      

        features = torch.concatenate(outputs_list, dim=0)        
        edge_index = to_undirected(graph_edge_index)
        
        if dataset_name in ["sarcoma_t1", "sarcoma_t2", "headneck"]:
            df = pd.read_csv(args.label_csv)
            if dataset_name == "sarcoma_t1":
                label = df[df["ID"] == patient_id].Grading.item() 
            elif dataset_name == "sarcoma_t2":
                label = df[df["ID"] == patient_id].Grading.item() 
            else:
                label = df[df["id"] == patient_id].hpv.item() 
                label = 0 if label == "negative" else 1
        else:
            label = np.load(img_list[i].replace("image", "label")).item()

        data = Data(x=features, edge_index=edge_index, adj_matrix = adjacency_matrix, label=torch.tensor(label))        
        torch.save(data, save_path)
     
if __name__ == "__main__":    

    for dataset in ["sarcoma_t1"]:
        for views in [8, 12, 16, 20]:
            
            parser = argparse.ArgumentParser()
            parser.add_argument("--dataset", default="sarcoma_t1", help="Name of dataset to be processed.", type=Path,
                                choices=["sarcoma_t1", "sarcoma_t2", "headneck", "nodule", "adrenal", "synapse", "vessel"], )
            parser.add_argument("--n_views", default=3, choices=[1, 3, 6, 14, 26, 42], help="Number of view points to slice input volume.", type=int)
            parser.add_argument("--target_spacing", default=1, help="Target voxel spacing of input data.", type=int)
            parser.add_argument("--interpolation", default="linear", choices=["linear", "bspline"], help="Interpolation type used for resampling.", type=str)
            parser.add_argument("--dinov2_model", default='facebook/dinov2-small', help="Pretrained DINOv2 model architecture.", type=str,
                                choices=['facebook/dinov2-small', 'facebook/dinov2-base', 'facebook/dinov2-large', 'facebook/dinov2-giant'])
            parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Processing device.", type=str)
            args = parser.parse_args()

            args.dataset = dataset
            args.n_views = views
            main(args)  
