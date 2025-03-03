"""
Mesh model loader for FBX and other 3D model formats.
Converts mesh models to point clouds.
"""
import os
import numpy as np
import random
from trimesh import load
import trimesh
from tqdm import tqdm

def load_mesh_file(file_path):
    """
    Load a mesh file using trimesh and return the mesh object.
    
    Args:
        file_path: Path to the mesh file
        
    Returns:
        Trimesh mesh object or None if loading failed
    """
    try:
        mesh = load(file_path)
        return mesh
    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
        return None

def mesh_to_point_cloud(mesh, num_points=1024, method="surface", noise=0.0):
    """
    Convert a mesh to a point cloud.
    
    Args:
        mesh: Trimesh mesh object
        num_points: Number of points to sample
        method: Sampling method - "surface" or "volume"
        noise: Amount of Gaussian noise to add (as a fraction of the model size)
        
    Returns:
        Numpy array of shape (num_points, 3) containing point cloud
    """
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, extract all meshes
        mesh_points = []
        for m in mesh.geometry.values():
            if hasattr(m, 'vertices') and len(m.vertices) > 0:
                mesh_points.append(m.vertices)
                
        if not mesh_points:
            raise ValueError("No valid meshes found in the scene")
            
        # Combine all vertices
        vertices = np.vstack(mesh_points)
    else:
        # Use the vertices directly
        vertices = mesh.vertices
    
    # If there are not enough vertices, use the sampling methods
    if len(vertices) < num_points:
        if method == "surface" and hasattr(mesh, "sample"):
            try:
                # Sample points from the surface
                points = mesh.sample(num_points)
            except:
                # Fall back to vertices if sampling fails
                indices = np.random.choice(len(vertices), num_points, replace=True)
                points = vertices[indices]
        else:
            # Use vertices with replacement
            indices = np.random.choice(len(vertices), num_points, replace=True)
            points = vertices[indices]
    else:
        # Random sample of vertices without replacement
        indices = np.random.choice(len(vertices), num_points, replace=False)
        points = vertices[indices]
    
    # Add noise if requested
    if noise > 0:
        # Calculate scale based on the model's bounding box
        if hasattr(mesh, 'bounding_box'):
            bbox = mesh.bounding_box.extents
            scale = np.max(bbox) * noise
        else:
            # Estimate scale from point cloud
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            scale = np.max(maxs - mins) * noise
            
        # Add Gaussian noise
        points += np.random.normal(0, scale, points.shape)
    
    # Normalize coordinates to fit within a unit cube
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    scale = np.max(maxs - mins)
    center = (mins + maxs) / 2
    
    # Center and scale
    normalized_points = (points - center) / scale
    
    return normalized_points

def load_directory_models(directory, num_points=1024, method="surface", 
                         supported_extensions=(".obj", ".fbx", ".stl", ".ply", ".glb", ".gltf"),
                         noise=0.01):
    """
    Load all supported 3D models from a directory and convert them to point clouds.
    
    Args:
        directory: Directory containing mesh files
        num_points: Number of points per point cloud
        method: Sampling method - "surface" or "volume"
        supported_extensions: Tuple of supported file extensions
        noise: Amount of Gaussian noise to add
        
    Returns:
        Dictionary of {filename: point_cloud}
    """
    point_clouds = {}
    
    # Scan directory for supported files
    model_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_extensions):
                model_files.append(os.path.join(root, file))
    
    print(f"Found {len(model_files)} supported 3D models in {directory}")
    
    # Process each file
    for file_path in tqdm(model_files, desc="Converting meshes to point clouds"):
        try:
            # Load the mesh
            mesh = load_mesh_file(file_path)
            
            if mesh is not None:
                # Convert to point cloud
                point_cloud = mesh_to_point_cloud(mesh, num_points, method, noise)
                
                # Use filename as key
                filename = os.path.basename(file_path)
                point_clouds[filename] = point_cloud
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return point_clouds

def save_point_clouds(point_clouds, output_dir="point_clouds"):
    """
    Save a dictionary of point clouds to numpy files.
    
    Args:
        point_clouds: Dictionary of {filename: point_cloud}
        output_dir: Directory to save point clouds
        
    Returns:
        Dictionary of {original_filename: saved_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    
    for filename, point_cloud in tqdm(point_clouds.items(), desc="Saving point clouds"):
        # Create output filename
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.npy")
        
        # Save as numpy array
        np.save(output_path, point_cloud)
        saved_paths[filename] = output_path
    
    return saved_paths

# Simple visualization function using matplotlib
def visualize_mesh_point_cloud(point_cloud, title="Mesh Point Cloud"):
    """
    Visualize a point cloud using matplotlib.
    
    Args:
        point_cloud: Numpy array of shape (num_points, 3)
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set aspect ratio
    max_range = np.max([
        np.max(point_cloud[:, 0]) - np.min(point_cloud[:, 0]),
        np.max(point_cloud[:, 1]) - np.min(point_cloud[:, 1]),
        np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])
    ])
    
    mid_x = (np.max(point_cloud[:, 0]) + np.min(point_cloud[:, 0])) * 0.5
    mid_y = (np.max(point_cloud[:, 1]) + np.min(point_cloud[:, 1])) * 0.5
    mid_z = (np.max(point_cloud[:, 2]) + np.min(point_cloud[:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()