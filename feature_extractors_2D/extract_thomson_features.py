from DINOv2.model import DINOv2Extractor
from glob import glob
import torchio as tio
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import trimesh
import pyvista as pv
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
import time

def calculate_rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
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

def create_thomson_sphere(n_vertices: int, save_sphere: bool = False):
    
    def normalize(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def random_points_on_sphere(n):
        """Uniform random points on unit sphere."""
        vec = np.random.randn(n, 3)
        return normalize(vec)

    def coulomb_repulsion(points, fixed_mask, lr=0.01, steps=2000):
        """
        Optimize free points on a sphere by minimizing Coulomb energy.
        points: (N,3) array on sphere
        fixed_mask: boolean array, True for fixed points
        """
        n = len(points)
        pts = points.copy()

        for step in range(steps):
            forces = np.zeros_like(pts)

            for i in range(n):
                for j in range(i+1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    f = diff / (dist**3 + 1e-9)  # Coulomb force
                    forces[i] += f
                    forces[j] -= f

            # Update only free points
            pts[~fixed_mask] += lr * forces[~fixed_mask]
            pts[~fixed_mask] = normalize(pts[~fixed_mask])

        return pts
    
    N = n_vertices

    if not os.path.exists(f"thomson_graph_{N}_views.pt"):
        print(f"Creating Thomson sphere with {N} views...")

        # Fixed points
        fixed_points = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        fixed_mask = np.array([True, True, True] + [False]*(N-3))

        # Initialize
        free_points = random_points_on_sphere(N-3)
        points_init = np.vstack([fixed_points, free_points])

        # Optimize
        points_opt = coulomb_repulsion(points_init, fixed_mask, lr=0.01, steps=10000)

        point_cloud = pv.PolyData(points_opt)
        mesh = point_cloud.delaunay_3d()
        surf = mesh.extract_surface()
        vertices = surf.points
        faces = surf.faces.reshape(-1, 4)[:, 1:]

        vertices_faces = {"vertices": vertices, "faces": faces}
        torch.save(vertices_faces, f"thomson_graph_{N}_views.pt")
    
    else:
        print(f"Loading Thomson sphere with {N} views...")
        vertices_faces = torch.load(f"thomson_graph_{N}_views.pt", weights_only=False)
        vertices = vertices_faces["vertices"]
        faces = vertices_faces["faces"]

    if save_sphere:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces) 
        mesh.export(f"thomson_mesh_{N}_views.ply")

    return vertices, faces

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

    return edge_index

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

    x_coords, y_coords, z_coords = foreground_voxels[:, 0], foreground_voxels[:, 1], foreground_voxels[:, 2]

    x_dim = x_coords.max().item() - x_coords.min().item() + 1
    y_dim = y_coords.max().item() - y_coords.min().item() + 1
    z_dim = z_coords.max().item() - z_coords.min().item() + 1

    x_dim = int(x_dim * margin)
    y_dim = int(y_dim * margin)
    z_dim = int(z_dim * margin)

    return (x_dim, y_dim, z_dim)

def rotate_3d_tensor_around_center(tensor: torch.Tensor, rotation_matrix: torch.Tensor, order: int = 1, device: str = 'cuda'):
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

    return rotated_tensor.squeeze(0).cpu()  # Remove channel dimension

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

def extract_thomson_features(dataset, model_name, views):

    match dataset:
        case "glioma_t1c_grading_binary":
            volumes = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*T1c_bias.nii.gz"))
            masks = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/*tumor_segmentation_merged.nii.gz"))
        case "glioma_t1c_grading_binary_custom_zspacing":
            volumes = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_T1c_custom_z_spacing/*T1c_bias_zspacing6.nii.gz"))
            masks = sorted(glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_T1c_custom_z_spacing/*tumor_segmentation_merged_zspacing6.nii.gz"))
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
        case "liver_ct_grading_binary_custom_zspacing":
            files = [file for file in glob("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/liver/CECT/HCC_CHCC_C2_custom_z_spacing/*zspacing6.nii.gz")]
            volumes = sorted([file for file in files if not "mask" in file])
            masks = sorted([file for file in files if "mask" in file])  
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")

    match model_name:
        case "DINOv2":
            model = DINOv2Extractor().cuda()
        case _:
            raise ValueError(f"Unknown model: {model}")
        
    vertices, faces = create_thomson_sphere(n_vertices=views, save_sphere=True)
    rot_matrices = create_rotation_matrices(vertices=vertices)
    graph_edge_index = create_graph_topology(vertices=vertices, faces=faces)

    for volume, mask in tqdm(zip(volumes, masks), total=len(volumes), desc=f"Extracting Thomson sphere {model_name} features from {dataset} with {views} views"):

        img = tio.ScalarImage(volume)
        seg = tio.LabelMap(mask)

        seg = tio.Resample(target=img)(seg)

        subject = tio.Subject(
            image=img,
            mask=seg
        )

        subject = tio.ToCanonical()(subject)
        subject = tio.Resample((1.0, 1.0, 1.0))(subject)

        # Crop cube around lesion with margin
        subject_temp = tio.CropOrPad(mask_name="mask")(subject)
        max_dim = np.max(subject_temp.image.shape)
        target_dim = int(max_dim * np.sqrt(2))
        subject = tio.CropOrPad(target_shape=(target_dim, target_dim, target_dim), mask_name="mask")(subject)

        img_slices = []
        seg_slices = []

        img_tensor = subject.image.tensor[0]
        seg_tensor = subject.mask.tensor[0]

        idx_slice = find_largest_lesion_slice(seg_tensor, axis=2)
        img_slices.append(img_tensor[:, :, idx_slice])
        seg_slices.append(seg_tensor[:, :, idx_slice])

        for rot_matrix in rot_matrices:
            img_tensor_rotated = rotate_3d_tensor_around_center(img_tensor.to(torch.float32).cuda(), torch.tensor(rot_matrix, dtype=torch.float32).cuda(), order=1, device='cuda')
            seg_tensor_rotated = rotate_3d_tensor_around_center(seg_tensor.to(torch.float32).cuda(), torch.tensor(rot_matrix, dtype=torch.float32).cuda(), order=0, device='cuda')
            idx_slice = find_largest_lesion_slice(seg_tensor_rotated, axis=2)
            img_slices.append(img_tensor_rotated[:, :, idx_slice])
            seg_slices.append(seg_tensor_rotated[:, :, idx_slice])

        encodings = []
        for img_slice, seg_slice in zip(img_slices, seg_slices):
            img_slice = img_slice.numpy()
            img_slice = crop_to_square(img_slice, mask=seg_slice.numpy())
            # img_slice = pad_to_square(img_slice, padding_value=0)
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            img_slice = (img_slice * 255).astype(np.uint8)
            img_pil = transforms.ToPILImage()(img_slice)
            img_pil.save("thomson_output_image.png", format="PNG")
            encodings.append(model(img_pil))
        
        features = torch.concat(encodings, dim=0)

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

        data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)  

        torch.save(data, f"{volume.replace('.nii.gz', f'_{model_name}_{str(views).zfill(2)}views_thomson_features.pt')}")
        # print(f"Processed {volume} with features shape: {features.shape}")

if __name__ == "__main__":
    
    for dataset in ["glioma_t1c_grading_binary", "sarcoma_t2_grading_binary", "breast_mri_grading_binary", "headneck_ct_hpv_binary", "kidney_ct_grading_binary", "liver_ct_grading_binary"]:
        for model in ["DINOv2"]:
            for views in [8, 16, 24]:
                extract_thomson_features(dataset, model, views)
