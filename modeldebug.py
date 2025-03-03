"""
Debug script to check model files and structure.
Run this script to see details about saved models.
"""
import os
import sys
import json
import torch

def inspect_model_directory(model_dir):
    """Check the contents of the model directory and report findings."""
    print(f"Checking model directory: {model_dir}")
    print("-" * 50)
    
    if not os.path.exists(model_dir):
        print(f"ERROR: Directory {model_dir} does not exist!")
        return
    
    # List directory contents
    files = os.listdir(model_dir)
    print(f"Directory contents: {files}")
    
    # Check for essential files
    essential_files = ["classifier.pth", "generator.pth", "tokenizer.json", "shape_types.json"]
    for file in essential_files:
        if file in files:
            file_path = os.path.join(model_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file}: {file_size} bytes")
        else:
            print(f"  ✗ {file}: MISSING")
    
    # Check shape_types.json
    if "shape_types.json" in files:
        try:
            with open(os.path.join(model_dir, "shape_types.json"), "r") as f:
                shape_types = json.load(f)
            print(f"Shape types: {shape_types}")
        except Exception as e:
            print(f"Error reading shape_types.json: {e}")
    
    # Check tokenizer.json
    if "tokenizer.json" in files:
        try:
            with open(os.path.join(model_dir, "tokenizer.json"), "r") as f:
                tokenizer = json.load(f)
            vocab_size = len(tokenizer.get("word_to_idx", {}))
            print(f"Tokenizer vocabulary size: {vocab_size}")
        except Exception as e:
            print(f"Error reading tokenizer.json: {e}")
    
    # Try to load classifier model
    if "classifier.pth" in files:
        try:
            classifier_state = torch.load(os.path.join(model_dir, "classifier.pth"), map_location="cpu")
            print(f"Classifier model keys: {list(classifier_state.keys())}")
        except Exception as e:
            print(f"Error loading classifier.pth: {e}")
    
    # Try to load generator model
    if "generator.pth" in files:
        try:
            generator_state = torch.load(os.path.join(model_dir, "generator.pth"), map_location="cpu")
            print(f"Generator model keys: {list(generator_state.keys())}")
        except Exception as e:
            print(f"Error loading generator.pth: {e}")
    
    print("-" * 50)

def main():
    """Main function to check model directories."""
    # Check default model directory
    default_model_dir = "model"
    inspect_model_directory(default_model_dir)
    
    # Check model path from model_path.txt if it exists
    if os.path.exists("model_path.txt"):
        try:
            with open("model_path.txt", "r") as f:
                model_dir = f.read().strip()
            print("\nAlso checking model directory from model_path.txt:")
            inspect_model_directory(model_dir)
        except Exception as e:
            print(f"Error reading model_path.txt: {e}")
    
    # Allow specifying a directory as a command-line argument
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        print(f"\nAlso checking specified directory: {model_dir}")
        inspect_model_directory(model_dir)

if __name__ == "__main__":
    main()