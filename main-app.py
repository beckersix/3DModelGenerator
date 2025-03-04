#!/usr/bin/env python3
"""
Main application for 3D Shape AI.
"""
import os
import sys
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

from models import PointCloudNetwork, EnhancedPointCloudGenerator
from data_utils import SimpleTokenizer, ShapeDataset, load_training_data, generate_enhanced_training_data
from command_interpreter import CommandInterpreter
from shape_ai import ShapeAI
from shape_generation import visualize_point_cloud
from training import train_classifier, train_enhanced_generator, train_generator, train_adaptive_generator

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def simple_train_test_split(data_list, labels_list, descriptions_list=None, test_size=0.2, random_state=None):
    """
    Simple implementation of train_test_split function to avoid sklearn dependency.
    
    Args:
        data_list: List of data points
        labels_list: List of labels
        descriptions_list: Optional list of descriptions
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Optional random seed for reproducibility
        
    Returns:
        train_data, test_data, train_labels, test_labels, train_descriptions, test_descriptions
    """
    if random_state is not None:
        random.seed(random_state)
        
    # Create indices and shuffle them
    indices = list(range(len(data_list)))
    random.shuffle(indices)
    
    # Calculate split point
    split_idx = int(len(indices) * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Split data and labels
    train_data = [data_list[i] for i in train_indices]
    test_data = [data_list[i] for i in test_indices]
    
    train_labels = [labels_list[i] for i in train_indices]
    test_labels = [labels_list[i] for i in test_indices]
    
    # Split descriptions if provided
    if descriptions_list is not None:
        train_descriptions = [descriptions_list[i] for i in train_indices]
        test_descriptions = [descriptions_list[i] for i in test_indices]
        return train_data, test_data, train_labels, test_labels, train_descriptions, test_descriptions
    
    return train_data, test_data, train_labels, test_labels

def main(args):
    """Main function to run the 3D Shape AI application."""
    print("=== 3D Shape AI ===")
    
    # Default to CPU if GPU is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse command line arguments
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
        train_clouds, val_clouds, train_labels, val_labels, train_descs, val_descs = simple_train_test_split(
            point_clouds, shape_labels, descriptions, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders for classifier
        train_dataset = ShapeDataset(train_clouds, train_labels, num_points=1024)
        val_dataset = ShapeDataset(val_clouds, val_labels, num_points=1024)
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Initialize and train classifier
        print("Training classifier...")
        num_classes = 6  # Fixed number of shape types
        print(f"Creating classifier with {num_classes} classes")
        classifier = PointCloudNetwork(num_classes=num_classes)
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
        shape_ai.save(args.model_dir)
        print(f"Models saved to {args.model_dir}")
        
        # Save model paths to an easily accessible location
        with open("model_path.txt", "w") as f:
            f.write(os.path.abspath(args.model_dir))
        print(f"Model path saved to model_path.txt for easy access in interactive mode")
        
    elif args.mode == 'test':
        # Load existing models
        if os.path.exists(args.model_dir):
            try:
                # Create classifier with fixed number of classes
                num_classes = 6  # Fixed number of shape types
                print(f"Creating classifier with {num_classes} classes")
                classifier = PointCloudNetwork(num_classes=num_classes)
                classifier.load_state_dict(torch.load(os.path.join(args.model_dir, 'classifier.pth'), map_location=device))
                print("Loaded classifier model successfully")
                
                # Create generator
                generator = EnhancedPointCloudGenerator(num_points=1024)
                generator.load_state_dict(torch.load(os.path.join(args.model_dir, 'generator.pth'), map_location=device))
                print("Loaded generator model successfully")
                
                # Create tokenizer
                tokenizer = SimpleTokenizer()
                tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
                if os.path.exists(tokenizer_path):
                    tokenizer.load(tokenizer_path)
                    print("Loaded tokenizer successfully")
                
                # Load shape types
                shape_types = ["cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"]
                
                # Create ShapeAI instance
                shape_ai = ShapeAI(classifier, generator, tokenizer, shape_types, device)
                print("Models loaded successfully")
            except Exception as e:
                print(f"Error loading models: {e}")
            
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
    
    elif args.mode == 'interactive':
        # Load models
        if os.path.exists(args.model_dir):
            try:
                # Create classifier with fixed number of classes
                num_classes = 6  # Fixed number of shape types
                print(f"Creating classifier with {num_classes} classes")
                classifier = PointCloudNetwork(num_classes=num_classes)
                classifier.load_state_dict(torch.load(os.path.join(args.model_dir, 'classifier.pth'), map_location=device))
                print("Loaded classifier model successfully")
                
                # Create generator
                generator = EnhancedPointCloudGenerator(num_points=1024)
                generator.load_state_dict(torch.load(os.path.join(args.model_dir, 'generator.pth'), map_location=device))
                print("Loaded generator model successfully")
                
                # Create tokenizer and command interpreter
                tokenizer = SimpleTokenizer()
                tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
                if os.path.exists(tokenizer_path):
                    tokenizer.load(tokenizer_path)
                    print("Loaded tokenizer successfully")
                
                # Load shape types
                shape_types = ["cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"]
                
                # Create ShapeAI and CommandInterpreter instances
                shape_ai = ShapeAI(classifier, generator, tokenizer, shape_types, device)
                command_interpreter = CommandInterpreter()
                
                print("\nWelcome to 3D Shape Generator!")
                print("Tell me what kind of shape you want to create.")
                print("\nExample commands:")
                print("- Simple shape names: 'cube', 'sphere', 'cylinder'")
                print("- Shape with description: 'create a large sphere'")
                print("- Detailed parameters: 'generate a tall cylinder with radius=0.8'")
                print("- Complex descriptions: 'make a hollow cube with size=1.5'")
                print("\nType 'help' for more information or 'quit' to exit.")
                
                while True:
                    try:
                        command = input("\nEnter your command: ").strip()
                        
                        if not command:
                            continue
                            
                        if command.lower() in ['quit', 'exit', 'bye']:
                            print("Goodbye!")
                            break
                        
                        if command.lower() in ['help', 'commands', '?']:
                            print("\nAvailable commands:")
                            print("- Basic shapes: cube, sphere, cylinder, pyramid, torus, cone")
                            print("- Create shapes: 'create a [shape]', 'generate a [shape]', 'make a [shape]'")
                            print("- Parameters: Add size, radius, height, etc. (e.g., 'cube size=2.0')")
                            print("- Properties: Add descriptors like large, small, tall, wide, etc.")
                            print("- Help: 'help', 'commands', '?'")
                            print("- Quit: 'quit', 'exit', 'bye'")
                            continue
                            
                        # Interpret command and generate shape
                        points, shape_type, properties = command_interpreter.interpret_and_generate(command, shape_ai)
                        
                        if points is not None:
                            # Display properties in a readable format
                            props_display = {}
                            for k, v in properties.items():
                                if isinstance(v, float):
                                    props_display[k] = round(v, 2)
                                else:
                                    props_display[k] = v
                                    
                            print(f"\nGenerated a {shape_type} shape with properties: {props_display}")
                            
                            # Convert to numpy if it's a tensor
                            points_np = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
                            
                            # Visualize the generated point cloud
                            visualize_point_cloud(points_np, title=f"Generated {shape_type}")
                        else:
                            print("Sorry, I couldn't understand that command. Please try again.")
                            print("Type 'help' to see available commands.")
                            
                    except KeyboardInterrupt:
                        print("\nGoodbye!")
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error loading models: {e}")
                return
        else:
            print(f"Model directory {args.model_dir} does not exist")
            return

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Shape AI')
    parser.add_argument('--mode', type=str, default='interactive', 
                      choices=['train', 'test', 'interactive'],
                      help='Mode to run the program in')
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory to save/load models')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory containing training data')
    parser.add_argument('--num-samples', type=int, default=1000,
                      help='Number of samples to generate for training')
    args = parser.parse_args()
    
    main(args)