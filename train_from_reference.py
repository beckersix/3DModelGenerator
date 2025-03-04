#!/usr/bin/env python
"""
Training script for the 3D Model Generator that processes reference models
and trains the adaptive generator model.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt

# Import project modules
from training import train_adaptive_generator
from point_cloud_processor import process_point_cloud, extract_metadata
from shape_ai import ShapeAI
from shape_generation import visualize_point_cloud

# Set up constants
DEFAULT_REFERENCE_DIR = "reference_models"
DEFAULT_MODEL_DIR = "models"
DEFAULT_DATA_DIR = "training_data"
DEFAULT_NUM_POINTS = 2048
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.0005

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train 3D Model Generator from reference models")
    parser.add_argument("--ref_dir", type=str, default=DEFAULT_REFERENCE_DIR,
                        help=f"Directory containing reference 3D models (default: {DEFAULT_REFERENCE_DIR})")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Directory to save trained models (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Directory to save processed training data (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--points", type=int, default=DEFAULT_NUM_POINTS,
                        help=f"Number of points per model (default: {DEFAULT_NUM_POINTS})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training if available")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize point clouds during processing")
    parser.add_argument("--force_preprocess", action="store_true",
                        help="Force preprocessing even if data already exists")
    
    return parser.parse_args()

def ensure_dir(directory):
    """Ensure the directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    return directory

def find_model_files(directory):
    """Find all 3D model files and numpy point clouds in the directory."""
    model_files = []
    npy_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('.npy'):
                npy_files.append(file_path)
            elif any(file.lower().endswith(ext) for ext in 
                    ['.obj', '.stl', '.fbx', '.ply', '.gltf', '.glb']):
                model_files.append(file_path)
    
    return model_files, npy_files

def load_and_process_models(model_files, npy_files, args):
    """Load and process 3D models and/or numpy point clouds."""
    processed_data = []
    processed_metadata = []
    
    # Process existing numpy files first
    for npy_file in tqdm(npy_files, desc="Processing existing point clouds"):
        try:
            # Load the point cloud
            point_cloud = np.load(npy_file)
            
            # Ensure we have the right number of points
            if point_cloud.shape[0] != args.points:
                # Resample if necessary
                print(f"Resampling {os.path.basename(npy_file)} to {args.points} points")
                point_cloud = process_point_cloud(point_cloud, target_size=args.points)
            
            # Extract metadata
            file_name = os.path.basename(npy_file)
            model_type = os.path.basename(os.path.dirname(npy_file)) or "unknown"
            description = f"Point cloud from {file_name} ({model_type})"
            
            processed_data.append(point_cloud)
            processed_metadata.append({
                "file_path": npy_file,
                "type": model_type,
                "description": description
            })
            
            if args.visualize:
                visualize_point_cloud(point_cloud, title=f"Point Cloud: {file_name}")
                
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
    
    # Process 3D model files
    for model_file in tqdm(model_files, desc="Processing 3D models"):
        try:
            # Load the mesh
            mesh = trimesh.load(model_file)
            
            # Sample points from the surface
            point_cloud, _ = trimesh.sample.sample_surface(mesh, args.points)
            
            # Normalize
            centroid = np.mean(point_cloud, axis=0)
            point_cloud -= centroid
            
            max_distance = np.max(np.linalg.norm(point_cloud, axis=1))
            if max_distance > 0:
                point_cloud /= max_distance
            
            # Extract metadata
            file_name = os.path.basename(model_file)
            model_type = os.path.basename(os.path.dirname(model_file)) or "unknown"
            description = f"Point cloud from {file_name} ({model_type})"
            
            processed_data.append(point_cloud)
            processed_metadata.append({
                "file_path": model_file,
                "type": model_type,
                "description": description
            })
            
            if args.visualize:
                visualize_point_cloud(point_cloud, title=f"Point Cloud: {file_name}")
                
        except Exception as e:
            print(f"Error processing {model_file}: {str(e)}")
    
    return processed_data, processed_metadata

def save_training_data(processed_data, processed_metadata, data_dir):
    """Save processed point clouds and metadata for training."""
    ensure_dir(data_dir)
    
    # Save each point cloud
    file_paths = []
    for i, point_cloud in enumerate(processed_data):
        file_path = os.path.join(data_dir, f"pointcloud_{i}.npy")
        np.save(file_path, point_cloud)
        file_paths.append(file_path)
    
    # Create full metadata with file paths
    metadata = []
    for i, meta in enumerate(processed_metadata):
        metadata.append({
            **meta,
            "training_file": file_paths[i]
        })
    
    # Save metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return file_paths, metadata_path

def load_or_process_data(args):
    """Load existing processed data or process new data."""
    data_dir = args.data_dir
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    # Check if data already exists and we're not forcing reprocessing
    if os.path.exists(metadata_path) and not args.force_preprocess:
        print(f"Using existing training data from {data_dir}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load point clouds
        point_clouds = []
        descriptions = []
        shape_labels = []
        
        for item in tqdm(metadata, desc="Loading training data"):
            try:
                file_path = item.get("training_file")
                if file_path and os.path.exists(file_path):
                    point_cloud = np.load(file_path)
                    point_clouds.append(point_cloud)
                    descriptions.append(item.get("description", ""))
                    shape_labels.append(item.get("type", "unknown"))
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        print(f"Loaded {len(point_clouds)} point clouds from existing data")
        return point_clouds, descriptions, shape_labels
    
    # Process new data
    print("Processing reference models to create training data")
    ref_dir = args.ref_dir
    model_files, npy_files = find_model_files(ref_dir)
    
    if not model_files and not npy_files:
        print(f"No 3D models or point cloud files found in {ref_dir}")
        print("Please use import_models.py to add reference models first.")
        sys.exit(1)
    
    print(f"Found {len(model_files)} 3D models and {len(npy_files)} point cloud files")
    
    # Process files
    processed_data, processed_metadata = load_and_process_models(model_files, npy_files, args)
    
    if not processed_data:
        print("No data could be processed. Exiting.")
        sys.exit(1)
    
    # Save processed data
    print(f"Saving {len(processed_data)} processed point clouds to {data_dir}")
    save_training_data(processed_data, processed_metadata, data_dir)
    
    # Extract information for training
    descriptions = [item.get("description", "") for item in processed_metadata]
    shape_labels = [item.get("type", "unknown") for item in processed_metadata]
    
    return processed_data, descriptions, shape_labels

def train_models(point_clouds, descriptions, shape_labels, args):
    """Train the adaptive generator model."""
    print("Starting model training...")
    
    # Determine device
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    # Train the adaptive generator
    trainer = train_adaptive_generator(
        point_clouds=point_clouds,
        descriptions=descriptions,
        shape_labels=shape_labels,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        model_dir=args.model_dir,
        device=device
    )
    
    # Save the models
    trainer.save_models(args.model_dir)
    
    # Create shape_ai instance
    shape_ai = ShapeAI(
        classifier=trainer.classifier, 
        generator=trainer.generator,
        tokenizer=trainer.tokenizer,
        shape_types=trainer.shape_types,
        device=device
    )
    
    # Save shape_ai
    shape_ai.save(args.model_dir)
    
    print(f"Models saved to {args.model_dir}")
    
    return trainer, shape_ai

def main(args):
    """Main function to train from reference models."""
    # Ensure all directories exist
    ensure_dir(args.ref_dir)
    ensure_dir(args.model_dir)
    ensure_dir(args.data_dir)
    
    # Check if we have any reference models
    if not os.path.exists(args.ref_dir) or not os.listdir(args.ref_dir):
        print(f"No reference models found in {args.ref_dir}")
        print("Please use import_models.py to add reference models first, or specify a different directory.")
        return
    
    # Load or process training data
    point_clouds, descriptions, shape_labels = load_or_process_data(args)
    
    print(f"Prepared {len(point_clouds)} point clouds for training")
    
    # Train the models
    trainer, shape_ai = train_models(point_clouds, descriptions, shape_labels, args)
    
    print("\nTraining completed successfully!")
    print("\nNext steps:")
    print("1. Use the trained models to generate new shapes:")
    print("   python main-app.py")
    print("2. To continue training with more data, add more reference models and run this script again")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
