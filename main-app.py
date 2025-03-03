#!/usr/bin/env python3
"""
Main entry point for 3D Shape AI application.
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from command_interpreter import CommandInterpreter


# Import modules
from models import PointCloudNetwork, PointCloudGenerator, EnhancedPointCloudGenerator
from data_utils import (
    PointCloudDataset, SimpleTokenizer, 
    load_training_data,generate_enhanced_training_data
)
from shape_generation import (
    generate_point_cloud, generate_custom_shape,
    visualize_point_cloud
)
from training import train_classifier, train_enhanced_generator
from shape_ai import ShapeAI

# Fix for OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    """Main function to run the 3D Shape AI application."""
    print("=== 3D Shape AI ===")
    
    # Default to CPU if GPU is not available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
   # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="3D Shape AI")
    parser.add_argument('--mode', type=str, default='interactive', choices=['train', 'test', 'interactive'],
                      help='Mode to run in')
    parser.add_argument('--data_dir', type=str, default='training_data',
                      help='Directory for training data')
    parser.add_argument('--model_dir', type=str, default=None,
                      help='Directory for saved models')
    parser.add_argument('--num_samples', type=int, default=1000,
                      help='Number of samples to generate for training')
    
    args = parser.parse_args()

    # Set default model directory if not specified
    if args.model_dir is None:
        args.model_dir = get_default_model_directory()
        print(f"Using default model directory: {args.model_dir}")
    
    # Create the ShapeAI instance
    shape_ai = None
    
    # Handle different modes
    if args.mode == 'train':
        print("Generating training data...")
        # Check if data directory exists, otherwise generate new data
        data_dir = get_default_output_directory(args.data_dir)
        if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, "metadata.json")):
            # Use enhanced training data generator
            point_clouds, shape_labels, descriptions = generate_enhanced_training_data(
                args.num_samples, output_dir=data_dir)
        else:
            print(f"Loading existing training data from {data_dir}")
            point_clouds, shape_labels, descriptions = load_training_data(data_dir)
        
        # Build tokenizer for text descriptions
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(descriptions)
        
        # Split data into train and validation sets
        train_clouds, val_clouds, train_labels, val_labels, train_descs, val_descs = train_test_split(
            point_clouds, shape_labels, descriptions, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders for classifier
        train_dataset = PointCloudDataset(train_clouds, train_labels, num_points=1024)
        val_dataset = PointCloudDataset(val_clouds, val_labels, num_points=1024)
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Initialize and train classifier
        print("Training classifier...")
        classifier = PointCloudNetwork(num_classes=len(set(shape_labels)))
        classifier, classifier_history = train_classifier(
            classifier, train_dataloader, val_dataloader, num_epochs=30, device=device)
        
        # Plot training history
        os.makedirs(args.model_dir, exist_ok=True)
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(classifier_history['train_loss'], label='Train Loss')
        plt.plot(classifier_history['val_loss'], label='Val Loss')
        plt.title('Classification Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(122)
        plt.plot(classifier_history['train_acc'], label='Train Accuracy')
        plt.plot(classifier_history['val_acc'], label='Val Accuracy')
        plt.title('Classification Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.model_dir, 'classifier_history.png'))
        plt.show()
        
        # Initialize and train enhanced generator
        print("Training enhanced generator...")
        # Use the enhanced generator model
        generator = EnhancedPointCloudGenerator(num_points=1024)
        generator, generator_history = train_enhanced_generator(
            generator, tokenizer, train_clouds, train_descs, train_labels, 
            num_epochs=50, device=device)
        
        # Plot generator training history
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(generator_history['chamfer_loss'], label='Chamfer Loss')
        plt.title('Generator Shape Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Chamfer Distance')
        plt.legend()
        
        if generator_history['classification_loss'] is not None:
            plt.subplot(122)
            plt.plot(generator_history['classification_loss'], label='Classification Loss')
            plt.title('Generator Classification Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.model_dir, 'generator_history.png'))
        plt.show()
        
        # Create and save the combined model
        shape_types = list(set([descriptions[i].split()[1] for i in range(len(descriptions))]))
        shape_ai = ShapeAI(classifier, generator, tokenizer, shape_types, device)
        
        # Save models
        os.makedirs(args.model_dir, exist_ok=True)
        shape_ai.save(args.model_dir)
        print(f"Models saved to {args.model_dir}")
        
        # Save model paths to an easily accessible location
        with open("model_path.txt", "w") as f:
            f.write(os.path.abspath(args.model_dir))
        print(f"Model path saved to model_path.txt for easy access in interactive mode")
        
    elif args.mode == 'test':
        # Load existing models
        if os.path.exists(args.model_dir):
            shape_ai = ShapeAI.load(args.model_dir, device)
            print("Models loaded successfully")
            
            # Test with some example inputs
            print("\nTesting classifier...")
            for shape_type in shape_ai.shape_types:
                print(f"Generating a {shape_type}...")
                test_point_cloud = generate_point_cloud(shape_type)
                predicted_shape, probs = shape_ai.classify(test_point_cloud)
                print(f"Predicted shape: {predicted_shape}")
                
                # Visualize
                visualize_point_cloud(test_point_cloud, f"{shape_type} (Predicted: {predicted_shape})")
            
            print("\nTesting generator...")
            for shape_type in shape_ai.shape_types:
                description = f"a {shape_type}"
                print(f"Generating from description: '{description}'")
                generated_cloud = shape_ai.generate(description)
                visualize_point_cloud(generated_cloud, f"Generated from: '{description}'")
        else:
            print(f"Error: Model directory {args.model_dir} not found. Please train models first.")
    
    # Interactive mode
    elif args.mode == 'interactive':
        # Try to load existing models if available
        model_loaded = False
        
        # First check if model_path.txt exists
        if os.path.exists("model_path.txt"):
            try:
                with open("model_path.txt", "r") as f:
                    model_dir = f.read().strip()
                if os.path.exists(model_dir):
                    print(f"Loading models from {model_dir}")
                    shape_ai = ShapeAI.load(model_dir, device)
                    print("Models loaded successfully")
                    model_loaded = True
            except Exception as e:
                print(f"Error loading models from path in model_path.txt: {e}")
        
        # If not loaded yet, try default path
        if not model_loaded and os.path.exists(args.model_dir):
            try:
                shape_ai = ShapeAI.load(args.model_dir, device)
                print("Models loaded successfully")
                model_loaded = True
            except Exception as e:
                print(f"Error loading models from default path: {e}")
                shape_ai = None
        
        if not model_loaded:
            shape_ai = None
            print("No models found. Some features will be unavailable until you train models.")
                
        # Run interactive interface
        run_interactive_interface(shape_ai)

def run_interactive_interface(shape_ai):
    """Run the interactive interface for the 3D Shape AI application."""
    while True:
        print("\n=== 3D Shape AI - Interactive Mode ===")
        print("1. Generate a standard shape")
        print("2. Generate a novel shape")
        print("3. Generate a shape from description (requires trained model)")
        print("4. Classify a point cloud (requires trained model)")
        print("5. Load 3D mesh models (FBX/OBJ/STL/etc.) and convert to point clouds")
        print("6. Natural language interface")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            # Generate a standard shape
            print("\nAvailable standard shapes:")
            print("1. Cube")
            print("2. Rectangular Prism")
            print("3. Sphere")
            print("4. Cylinder")
            print("5. Pyramid")
            print("6. Torus")
            
            shape_choice = input("\nEnter shape number (1-6) or name: ")
            
            # Convert numeric choice to shape name
            if shape_choice.isdigit():
                choice_num = int(shape_choice)
                if choice_num == 1:
                    shape_name = "cube"
                elif choice_num == 2:
                    shape_name = "rectangular_prism"
                elif choice_num == 3:
                    shape_name = "sphere"
                elif choice_num == 4:
                    shape_name = "cylinder"
                elif choice_num == 5:
                    shape_name = "pyramid"
                elif choice_num == 6:
                    shape_name = "torus"
                else:
                    print("Invalid choice. Please enter a number between 1 and 6.")
                    continue
            else:
                shape_name = shape_choice.lower()
            
            # Get parameters
            try:
                num_points = int(input("Enter number of points (default: 1024): ") or 1024)
                noise = float(input("Enter noise level (0.0-0.1, default: 0.05): ") or 0.05)
                
                # Shape-specific parameters
                kwargs = {'noise': noise}
                
                if shape_name in ['cube']:
                    size = float(input("Enter cube size (default: 1.0): ") or 1.0)
                    kwargs['size'] = size
                
                elif shape_name in ['rectangular_prism', 'rect_prism']:
                    size_x = float(input("Enter X size (default: 1.0): ") or 1.0)
                    size_y = float(input("Enter Y size (default: 2.0): ") or 2.0)
                    size_z = float(input("Enter Z size (default: 0.5): ") or 0.5)
                    kwargs.update({'size_x': size_x, 'size_y': size_y, 'size_z': size_z})
                
                elif shape_name in ['sphere']:
                    radius = float(input("Enter radius (default: 1.0): ") or 1.0)
                    kwargs['radius'] = radius
                
                elif shape_name in ['cylinder']:
                    radius = float(input("Enter radius (default: 1.0): ") or 1.0)
                    height = float(input("Enter height (default: 2.0): ") or 2.0)
                    kwargs.update({'radius': radius, 'height': height})
                
                elif shape_name in ['pyramid']:
                    base_size = float(input("Enter base size (default: 1.0): ") or 1.0)
                    height = float(input("Enter height (default: 1.5): ") or 1.5)
                    kwargs.update({'base_size': base_size, 'height': height})
                
                elif shape_name in ['torus']:
                    major_radius = float(input("Enter major radius (default: 1.0): ") or 1.0)
                    minor_radius = float(input("Enter minor radius (default: 0.3): ") or 0.3)
                    kwargs.update({'major_radius': major_radius, 'minor_radius': minor_radius})
                
                # Generate the point cloud
                point_cloud = generate_point_cloud(shape_name, num_points, **kwargs)
                
                # Visualize
                visualize_point_cloud(point_cloud, f"{shape_name.capitalize()} Point Cloud ({num_points} points)")
                
                # Option to save
                save_choice = input("Do you want to save this point cloud? (y/n, default: n): ").lower() or 'n'
                if save_choice == 'y':
                    filename = input("Enter filename (default: point_cloud.npy): ") or "point_cloud.npy"
                    np.save(filename, point_cloud)
                    print(f"Point cloud saved as {filename}")
            
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            # Generate a novel shape
            print("\nAvailable novel shapes:")
            print("1. Cone")
            print("2. Ellipsoid")
            print("3. Capsule")
            print("4. Star")
            print("5. Helix")
            
            shape_choice = input("\nEnter shape number (1-5) or name: ")
            
            # Convert numeric choice to shape name
            if shape_choice.isdigit():
                choice_num = int(shape_choice)
                if choice_num == 1:
                    shape_name = "cone"
                elif choice_num == 2:
                    shape_name = "ellipsoid"
                elif choice_num == 3:
                    shape_name = "capsule"
                elif choice_num == 4:
                    shape_name = "star"
                elif choice_num == 5:
                    shape_name = "helix"
                else:
                    print("Invalid choice. Please enter a number between 1 and 5.")
                    continue
            else:
                shape_name = shape_choice.lower()
            
            # Get parameters
            try:
                num_points = int(input("Enter number of points (default: 1024): ") or 1024)
                noise = float(input("Enter noise level (0.0-0.1, default: 0.05): ") or 0.05)
                
                # Shape-specific parameters
                kwargs = {'noise': noise}
                
                if shape_name == 'cone':
                    radius = float(input("Enter base radius (default: 1.0): ") or 1.0)
                    height = float(input("Enter height (default: 2.0): ") or 2.0)
                    kwargs.update({'radius': radius, 'height': height})
                
                elif shape_name == 'ellipsoid':
                    radius_x = float(input("Enter X radius (default: 1.0): ") or 1.0)
                    radius_y = float(input("Enter Y radius (default: 0.7): ") or 0.7)
                    radius_z = float(input("Enter Z radius (default: 0.5): ") or 0.5)
                    kwargs.update({'radius_x': radius_x, 'radius_y': radius_y, 'radius_z': radius_z})
                
                elif shape_name == 'capsule':
                    radius = float(input("Enter radius (default: 0.5): ") or 0.5)
                    height = float(input("Enter height (default: 2.0): ") or 2.0)
                    kwargs.update({'radius': radius, 'height': height})
                
                elif shape_name == 'star':
                    outer_radius = float(input("Enter outer radius (default: 1.0): ") or 1.0)
                    inner_radius = float(input("Enter inner radius (default: 0.4): ") or 0.4)
                    points = int(input("Enter number of points (default: 5): ") or 5)
                    height = float(input("Enter height (default: 0.3): ") or 0.3)
                    kwargs.update({
                        'outer_radius': outer_radius, 
                        'inner_radius': inner_radius,
                        'points': points,
                        'height': height
                    })
                
                elif shape_name == 'helix':
                    radius = float(input("Enter radius (default: 1.0): ") or 1.0)
                    pitch = float(input("Enter pitch (default: 0.3): ") or 0.3)
                    turns = float(input("Enter number of turns (default: 5): ") or 5)
                    thickness = float(input("Enter thickness (default: 0.2): ") or 0.2)
                    kwargs.update({
                        'radius': radius, 
                        'pitch': pitch,
                        'turns': turns,
                        'thickness': thickness
                    })
                
                # Generate the point cloud
                point_cloud = generate_custom_shape(shape_name, num_points, **kwargs)
                
                # Visualize
                visualize_point_cloud(point_cloud, f"{shape_name.capitalize()} Point Cloud ({num_points} points)")
                
                # Option to save
                save_choice = input("Do you want to save this point cloud? (y/n, default: n): ").lower() or 'n'
                if save_choice == 'y':
                    filename = input("Enter filename (default: point_cloud.npy): ") or "point_cloud.npy"
                    np.save(filename, point_cloud)
                    print(f"Point cloud saved as {filename}")
            
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            # Generate from description (requires trained model)
            if shape_ai is None or shape_ai.generator is None:
                print("Error: No trained generator model available.")
                print("Please train a model first or make sure the model directory contains a generator model.")
                continue
            
            description = input("Enter a description (e.g. 'a large cube', 'a tall cylinder'): ")
            
            try:
                # Generate point cloud from description
                point_cloud = shape_ai.generate(description)
                
                # Visualize
                visualize_point_cloud(point_cloud, f"Generated from: '{description}'")
                
                # Option to save
                save_choice = input("Do you want to save this point cloud? (y/n, default: n): ").lower() or 'n'
                if save_choice == 'y':
                    filename = input("Enter filename (default: generated_shape.npy): ") or "generated_shape.npy"
                    np.save(filename, point_cloud)
                    print(f"Point cloud saved as {filename}")
            
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            # Classify a point cloud (requires trained model)
            if shape_ai is None or shape_ai.classifier is None:
                print("Error: No trained classifier model available.")
                print("Please train a model first or make sure the model directory contains a classifier model.")
                continue
            
            print("\nSelect point cloud to classify:")
            print("1. Generate a new point cloud")
            print("2. Load from file")
            
            classify_choice = input("Enter choice (1-2): ")
            
            try:
                if classify_choice == '1':
                    # Generate a new point cloud for classification
                    print("\nAvailable shapes:")
                    for i, shape in enumerate(["cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"]):
                        print(f"{i+1}. {shape.capitalize()}")
                    
                    shape_idx = int(input("Enter shape number (1-6): ")) - 1
                    if shape_idx < 0 or shape_idx >= 6:
                        print("Invalid shape number.")
                        continue
                    
                    shape_name = ["cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"][shape_idx]
                    point_cloud = generate_point_cloud(shape_name)
                
                elif classify_choice == '2':
                    # Load from file
                    filename = input("Enter filename: ")
                    if not os.path.exists(filename):
                        print(f"Error: File {filename} not found.")
                        continue
                    
                    point_cloud = np.load(filename)
                
                else:
                    print("Invalid choice.")
                    continue
                
                # Classify the point cloud
                predicted_shape, probs = shape_ai.classify(point_cloud)
                
                # Show results
                print(f"\nPredicted shape: {predicted_shape}")
                print("\nClass probabilities:")
                for i, shape in enumerate(shape_ai.shape_types):
                    print(f"{shape.capitalize()}: {probs[i]*100:.2f}%")
                
                # Visualize
                visualize_point_cloud(point_cloud, f"Point Cloud (Predicted: {predicted_shape})")
            
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            # New option: Load 3D mesh models
            try:
                # Import the FBX loader module
                from fbx_loader import (
                    load_directory_models, 
                    save_point_clouds, 
                    visualize_mesh_point_cloud
                )
                
                print("\n=== 3D Mesh Model Loader ===")
                print("This feature converts 3D mesh models to point clouds")
                print("Supported formats: OBJ, FBX, STL, PLY, GLB, GLTF")
                
                # Get directory path
                directory = input("Enter path to directory containing 3D models: ")
                if not os.path.exists(directory):
                    print(f"Error: Directory {directory} does not exist.")
                    continue
                
                # Ask for point cloud options
                num_points = int(input("Enter number of points per model (default: 1024): ") or 1024)
                
                # Ask for sampling method
                print("\nSampling Method:")
                print("1. Surface sampling (recommended)")
                print("2. Vertex sampling")
                method = "surface" if input("Enter choice (1-2, default: 1): ") in ["", "1"] else "volume"
                
                # Ask for noise level
                noise = float(input("Enter noise level (0.0-0.1, default: 0.01): ") or 0.01)
                
                # Ask for output directory
                output_dir = input("Enter output directory for point clouds (default: 'converted_models'): ") or "converted_models"
                
                # Load and convert models
                print(f"\nLoading and converting models from {directory}...")
                point_clouds = load_directory_models(directory, num_points, method, noise=noise)
                
                if not point_clouds:
                    print("No valid models were found or converted.")
                    continue
                
                print(f"\nSuccessfully converted {len(point_clouds)} models to point clouds.")
                
                # Preview option
                preview = input("Would you like to preview a random converted model? (y/n, default: y): ").lower() or 'y'
                if preview == 'y' and point_clouds:
                    # Select a random model
                    random_key = random.choice(list(point_clouds.keys()))
                    print(f"Previewing: {random_key}")
                    visualize_mesh_point_cloud(point_clouds[random_key], f"Converted: {random_key}")
                
                # Save option
                save = input("Would you like to save all converted point clouds? (y/n, default: y): ").lower() or 'y'
                if save == 'y':
                    saved_paths = save_point_clouds(point_clouds, output_dir)
                    print(f"Saved {len(saved_paths)} point clouds to {output_dir}")
                    
                    # Offer to use for classification
                    if shape_ai is not None and shape_ai.classifier is not None:
                        classify = input("Would you like to classify these models? (y/n, default: y): ").lower() or 'y'
                        if classify == 'y':
                            for filename, point_cloud in point_clouds.items():
                                try:
                                    predicted_shape, probs = shape_ai.classify(point_cloud)
                                    print(f"\nFile: {filename}")
                                    print(f"Predicted shape: {predicted_shape}")
                                    print("Top 3 probabilities:")
                                    sorted_probs = sorted([(s, p) for s, p in zip(shape_ai.shape_types, probs)], 
                                                        key=lambda x: x[1], reverse=True)
                                    for shape, prob in sorted_probs[:3]:
                                        print(f"  {shape}: {prob*100:.2f}%")
                                except Exception as e:
                                    print(f"Error classifying {filename}: {e}")
                
            except ImportError as e:
                print(f"Error: Could not load 3D mesh converter module. You may need to install trimesh:")
                print("pip install trimesh matplotlib numpy")
                print(f"Error details: {e}")
            except Exception as e:
                print(f"Error processing 3D models: {e}")
            
        elif choice == '6':
            # Natural language interface
            run_nlp_interface(shape_ai)
            
        elif choice == '7':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

def run_nlp_interface(shape_ai):
    """Run the natural language interface for the 3D Shape AI."""
    interpreter = CommandInterpreter()
    
    print("\n=== Natural Language Interface ===")
    print("Talk to me in plain English about what 3D shapes you want.")
    print("Examples:")
    print("- 'Create a large red cube'")
    print("- 'Make a tall cylinder'")
    print("- 'I want a smooth sphere'")
    print("- 'Convert 3D models in my models folder'")
    print("- 'Generate a donut shape'")
    print("- 'Help'")
    print("\nType 'exit' or 'quit' to return to the main menu.")
    
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Skip empty input
        if not user_input.strip():
            continue
        
        # Interpret the command
        command = interpreter.interpret_command(user_input)
        
        # Generate response
        response = interpreter.generate_response(command)
        print(response)
        
        # Process the command
        if command["command"] == "quit":
            return  # Return to main menu
        
        elif command["command"] == "help":
            continue  # Response already shown
            
        elif command["command"] == "create":
            # Create a specific shape
            try:
                from shape_generation import generate_point_cloud, generate_custom_shape, visualize_point_cloud
                
                shape_type = command["shape_type"]
                params = command["parameters"]
                
                print(f"Creating {shape_type} with parameters: {params}")
                
                # Use the appropriate generation function
                if shape_type in ["cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"]:
                    point_cloud = generate_point_cloud(shape_type, **params)
                else:
                    point_cloud = generate_custom_shape(shape_type, **params)
                
                # Visualize
                visualize_point_cloud(point_cloud, f"{shape_type.capitalize()} Point Cloud")
                
                # Option to save
                save_choice = input("Do you want to save this point cloud? (y/n, default: n): ").lower() or 'n'
                if save_choice == 'y':
                    filename = input("Enter filename (default: point_cloud.npy): ") or "point_cloud.npy"
                    np.save(filename, point_cloud)
                    print(f"Point cloud saved as {filename}")
                
            except Exception as e:
                print(f"Error creating shape: {e}")
        
        elif command["command"] == "generate":
            # Generate from text description
            if shape_ai is None or shape_ai.generator is None:
                print("Error: No trained generator model available.")
                print("Please train a model first or make sure the model directory contains a generator model.")
                continue
            
            description = command["description"]
            try:
                # Generate point cloud from description
                point_cloud = shape_ai.generate(description)
                
                # Visualize
                from shape_generation import visualize_point_cloud
                visualize_point_cloud(point_cloud, f"Generated from: '{description}'")
                
                # Option to save
                save_choice = input("Do you want to save this point cloud? (y/n, default: n): ").lower() or 'n'
                if save_choice == 'y':
                    filename = input("Enter filename (default: generated_shape.npy): ") or "generated_shape.npy"
                    np.save(filename, point_cloud)
                    print(f"Point cloud saved as {filename}")
            
            except Exception as e:
                print(f"Error generating shape: {e}")
        
        elif command["command"] == "classify":
            # Classify a point cloud
            if shape_ai is None or shape_ai.classifier is None:
                print("Error: No trained classifier model available.")
                print("Please train a model first or make sure the model directory contains a classifier model.")
                continue
            
            print("\nSelect point cloud to classify:")
            print("1. Generate a new point cloud")
            print("2. Load from file")
            
            classify_choice = input("Enter choice (1-2): ")
            
            try:
                if classify_choice == '1':
                    # Generate a new point cloud for classification
                    print("\nAvailable shapes:")
                    for i, shape in enumerate(["cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"]):
                        print(f"{i+1}. {shape.capitalize()}")
                    
                    shape_idx = int(input("Enter shape number (1-6): ")) - 1
                    if shape_idx < 0 or shape_idx >= 6:
                        print("Invalid shape number.")
                        continue
                    
                    shape_name = ["cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"][shape_idx]
                    point_cloud = generate_point_cloud(shape_name)
                
                elif classify_choice == '2':
                    # Load from file
                    filename = input("Enter filename: ")
                    if not os.path.exists(filename):
                        print(f"Error: File {filename} not found.")
                        continue
                    
                    point_cloud = np.load(filename)
                
                else:
                    print("Invalid choice.")
                    continue
                
                # Classify the point cloud
                predicted_shape, probs = shape_ai.classify(point_cloud)
                
                # Show results
                print(f"\nPredicted shape: {predicted_shape}")
                print("\nClass probabilities:")
                for i, shape in enumerate(shape_ai.shape_types):
                    print(f"{shape.capitalize()}: {probs[i]*100:.2f}%")
                
                # Visualize
                visualize_point_cloud(point_cloud, f"Point Cloud (Predicted: {predicted_shape})")
            
            except Exception as e:
                print(f"Error classifying shape: {e}")
        
        elif command["command"] == "convert":
            # Convert 3D models
            try:
                # Import the FBX loader module
                from fbx_loader import (
                    load_directory_models, 
                    save_point_clouds, 
                    visualize_mesh_point_cloud
                )
                
                print("\n=== 3D Mesh Model Converter ===")
                
                # Get directory path from command or prompt
                directory = command.get("path_info", "")
                if not directory:
                    directory = input("Enter path to directory containing 3D models: ")
                
                if not os.path.exists(directory):
                    print(f"Error: Directory {directory} does not exist.")
                    continue
                
                # Ask for point cloud options
                num_points = int(input("Enter number of points per model (default: 1024): ") or 1024)
                
                # Ask for sampling method
                print("\nSampling Method:")
                print("1. Surface sampling (recommended)")
                print("2. Vertex sampling")
                method = "surface" if input("Enter choice (1-2, default: 1): ") in ["", "1"] else "volume"
                
                # Ask for noise level
                noise = float(input("Enter noise level (0.0-0.1, default: 0.01): ") or 0.01)
                
                # Ask for output directory
                output_dir = input(f"Enter output directory for point clouds (default: 'converted_models'): ") or get_default_output_directory("converted_models")
                
                # Load and convert models
                print(f"\nLoading and converting models from {directory}...")
                point_clouds = load_directory_models(directory, num_points, method, noise=noise)
                
                if not point_clouds:
                    print("No valid models were found or converted.")
                    continue
                
                print(f"\nSuccessfully converted {len(point_clouds)} models to point clouds.")
                
                # Preview option
                preview = input("Would you like to preview a random converted model? (y/n, default: y): ").lower() or 'y'
                if preview == 'y' and point_clouds:
                    # Select a random model
                    random_key = random.choice(list(point_clouds.keys()))
                    print(f"Previewing: {random_key}")
                    visualize_mesh_point_cloud(point_clouds[random_key], f"Converted: {random_key}")
                
                # Save option
                save = input("Would you like to save all converted point clouds? (y/n, default: y): ").lower() or 'y'
                if save == 'y':
                    saved_paths = save_point_clouds(point_clouds, output_dir)
                    print(f"Saved {len(saved_paths)} point clouds to {output_dir}")
                    
                    # Offer to use for classification
                    if shape_ai is not None and shape_ai.classifier is not None:
                        classify = input("Would you like to classify these models? (y/n, default: y): ").lower() or 'y'
                        if classify == 'y':
                            for filename, point_cloud in point_clouds.items():
                                try:
                                    predicted_shape, probs = shape_ai.classify(point_cloud)
                                    print(f"\nFile: {filename}")
                                    print(f"Predicted shape: {predicted_shape}")
                                    print("Top 3 probabilities:")
                                    sorted_probs = sorted([(s, p) for s, p in zip(shape_ai.shape_types, probs)], 
                                                        key=lambda x: x[1], reverse=True)
                                    for shape, prob in sorted_probs[:3]:
                                        print(f"  {shape}: {prob*100:.2f}%")
                                except Exception as e:
                                    print(f"Error classifying {filename}: {e}")
            
            except ImportError as e:
                print(f"Error: Could not load 3D mesh converter module. You may need to install trimesh:")
                print("pip install trimesh matplotlib numpy")
                print(f"Error details: {e}")
            except Exception as e:
                print(f"Error processing 3D models: {e}")

def get_default_model_directory():
    """
    Get the default model directory path and ensure it exists.
    
    Returns:
        Path to the default model directory
    """
    # Create a models directory within the project folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, "models")
    
    # Create the directory if it doesn't exist
    os.makedirs(default_model_dir, exist_ok=True)
    
    return default_model_dir

def get_default_output_directory(name="converted_models"):
    """
    Get the default output directory path and ensure it exists.
    
    Args:
        name: Name of the output directory
        
    Returns:
        Path to the default output directory
    """
    # Create an output directory within the project folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, name)
    
    # Create the directory if it doesn't exist
    os.makedirs(default_output_dir, exist_ok=True)
    
    return default_output_dir

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the main function
    main()