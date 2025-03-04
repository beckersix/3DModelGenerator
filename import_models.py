#!/usr/bin/env python
"""
Import 3D models utility script for the 3D Model Generator.

This script helps users import 3D models from a source directory to the reference_models directory,
converting them if needed and organizing them properly for training.
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
import trimesh
import numpy as np
from tqdm import tqdm

# Set up constants
DEFAULT_REFERENCE_DIR = "reference_models"
SUPPORTED_FORMATS = [".obj", ".stl", ".fbx", ".ply", ".gltf", ".glb"]
DEFAULT_NUM_POINTS = 2048
DEFAULT_NOISE = 0.05

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Import 3D models for training")
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="Source directory containing 3D models")
    parser.add_argument("--dest", "-d", type=str, default=DEFAULT_REFERENCE_DIR,
                        help=f"Destination directory for reference models (default: {DEFAULT_REFERENCE_DIR})")
    parser.add_argument("--convert", "-c", action="store_true",
                        help="Convert models to point clouds immediately")
    parser.add_argument("--points", "-p", type=int, default=DEFAULT_NUM_POINTS,
                        help=f"Number of points per model if converting (default: {DEFAULT_NUM_POINTS})")
    parser.add_argument("--noise", "-n", type=float, default=DEFAULT_NOISE,
                        help=f"Noise level for point sampling if converting (default: {DEFAULT_NOISE})")
    parser.add_argument("--organize", "-o", action="store_true",
                        help="Organize models by type based on filename")
    parser.add_argument("--overwrite", "-w", action="store_true",
                        help="Overwrite existing files")
    
    return parser.parse_args()

def find_models(source_dir):
    """Find all 3D model files in the source directory."""
    model_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in SUPPORTED_FORMATS:
                model_files.append(os.path.join(root, file))
    
    return model_files

def ensure_dir(directory):
    """Ensure the directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    return directory

def guess_model_type(filename):
    """Guess the model type based on the filename."""
    filename = filename.lower()
    
    # Common categories
    categories = {
        "chair": ["chair", "seat", "stool"],
        "table": ["table", "desk", "stand"],
        "car": ["car", "vehicle", "auto"],
        "aircraft": ["plane", "aircraft", "jet", "airplane"],
        "building": ["house", "building", "structure"],
        "furniture": ["furniture", "cabinet", "shelf", "bookcase", "dresser"],
        "animal": ["animal", "dog", "cat", "bird", "horse"],
        "human": ["human", "person", "man", "woman", "figure"],
        "plant": ["plant", "tree", "flower", "bush"],
        "object": ["object", "thing", "item"]
    }
    
    base_name = os.path.splitext(os.path.basename(filename))[0].lower()
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in base_name:
                return category
    
    return "misc"

def load_and_sample_mesh(mesh_path, num_points=2048, noise=0.05):
    """Load a mesh and sample points from its surface."""
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        
        # Sample points from the surface
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        
        # Center the points
        centroid = np.mean(points, axis=0)
        points -= centroid
        
        # Scale to fit in a unit sphere
        max_distance = np.max(np.linalg.norm(points, axis=1))
        if max_distance > 0:
            points /= max_distance
        
        # Add noise
        if noise > 0:
            points += np.random.normal(0, noise, points.shape)
        
        return points
    except Exception as e:
        print(f"Error processing {mesh_path}: {str(e)}")
        return None

def import_models(args):
    """Import 3D models from source to destination."""
    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.dest)
    
    print(f"Searching for 3D models in {source_dir}")
    model_files = find_models(source_dir)
    print(f"Found {len(model_files)} model files")
    
    if not model_files:
        print("No model files found. Exiting.")
        return
    
    # Ensure destination directory exists
    ensure_dir(dest_dir)
    
    # Create metadata to track imports
    metadata_path = os.path.join(dest_dir, "import_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            metadata = {}
    
    # Import all models
    successfully_imported = 0
    skipped = 0
    errors = 0
    
    for model_file in tqdm(model_files, desc="Importing models"):
        filename = os.path.basename(model_file)
        dest_subdir = dest_dir
        
        # Organize by type if requested
        if args.organize:
            model_type = guess_model_type(filename)
            dest_subdir = os.path.join(dest_dir, model_type)
            ensure_dir(dest_subdir)
        
        dest_file = os.path.join(dest_subdir, filename)
        
        # Check if file already exists
        if os.path.exists(dest_file) and not args.overwrite:
            print(f"Skipping {filename} (already exists)")
            skipped += 1
            continue
        
        try:
            # Copy the file to the destination
            shutil.copy2(model_file, dest_file)
            
            # Record in metadata
            rel_path = os.path.relpath(dest_file, dest_dir)
            metadata[rel_path] = {
                "original_path": model_file,
                "type": guess_model_type(filename),
                "imported_at": "AUTO",  # Will be replaced with the current timestamp
                "description": f"Model imported from {os.path.basename(model_file)}"
            }
            
            # Convert to point cloud if requested
            if args.convert:
                points = load_and_sample_mesh(dest_file, 
                                            num_points=args.points, 
                                            noise=args.noise)
                if points is not None:
                    # Save as numpy file
                    points_file = os.path.splitext(dest_file)[0] + ".npy"
                    np.save(points_file, points)
                    # Update metadata
                    rel_points_path = os.path.relpath(points_file, dest_dir)
                    metadata[rel_path]["point_cloud"] = rel_points_path
                    metadata[rel_path]["num_points"] = args.points
                    metadata[rel_path]["noise"] = args.noise
            
            successfully_imported += 1
            
        except Exception as e:
            print(f"Error importing {filename}: {str(e)}")
            errors += 1
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"""
Import summary:
    Successfully imported: {successfully_imported}
    Skipped (already exist): {skipped}
    Errors: {errors}
    Total processed: {len(model_files)}
    
Models imported to: {dest_dir}
Metadata saved to: {metadata_path}
""")
    
    if args.convert:
        print(f"Point clouds generated with {args.points} points and {args.noise} noise level")
    
    print("Next steps:")
    print("1. Check the imported models in the reference_models directory")
    print("2. Run the training script to train the model on your new data:")
    print("   python main-app.py --mode train")

if __name__ == "__main__":
    args = parse_arguments()
    import_models(args)
