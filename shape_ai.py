"""
ShapeAI class that combines classification and generation capabilities.
"""
import os
import re
import json
import torch
import torch.nn.functional as F
import numpy as np
import inspect

from models import PointCloudNetwork, PointCloudGenerator, EnhancedPointCloudGenerator
from data_utils import SimpleTokenizer
from shape_generation import generate_point_cloud

class ShapeAI:
    def __init__(self, classifier=None, generator=None, tokenizer=None, shape_types=None, device='cuda'):
        """Class that combines point cloud classification and generation
        
        Args:
            classifier: PointCloudNetwork model
            generator: PointCloudGenerator model
            tokenizer: SimpleTokenizer for text processing
            shape_types: List of shape type names
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.classifier = classifier.to(self.device) if classifier else None
        self.generator = generator.to(self.device) if generator else None
        self.tokenizer = tokenizer
        self.shape_types = shape_types or ["cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"]
        
        # Precompiled regex patterns for parameter extraction
        self.param_patterns = {
            "size": r'size\s*=\s*(\d+\.?\d*)',
            "width": r'width\s*=\s*(\d+\.?\d*)',
            "height": r'height\s*=\s*(\d+\.?\d*)',
            "depth": r'depth\s*=\s*(\d+\.?\d*)',
            "radius": r'radius\s*=\s*(\d+\.?\d*)',
            "major_radius": r'major[_\s]radius\s*=\s*(\d+\.?\d*)',
            "minor_radius": r'minor[_\s]radius\s*=\s*(\d+\.?\d*)',
            "base_size": r'base[_\s]size\s*=\s*(\d+\.?\d*)',
            "noise": r'noise\s*=\s*(\d+\.?\d*)',
        }
    
    def classify(self, point_cloud):
        """Classify a single point cloud
        
        Args:
            point_cloud: Numpy array of shape (num_points, 3)
            
        Returns:
            Predicted shape type and probabilities
        """
        if self.classifier is None:
            raise ValueError("Classifier model is not loaded")
        
        # Preprocess point cloud
        point_cloud_tensor = torch.FloatTensor(point_cloud).unsqueeze(0).to(self.device)
        
        # Get prediction
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(point_cloud_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            
            # Get predicted shape type
            predicted_shape = self.shape_types[predicted_class]
            
            # Get class probabilities
            class_probs = probabilities[0].cpu().numpy()
            
        return predicted_shape, class_probs
    
    def generate(self, description):
        """Generate a point cloud from a text description
        
        Args:
            description: Text description of the desired shape
            
        Returns:
            Generated point cloud as numpy array
        """
        if self.generator is None:
            raise ValueError("Generator model is not loaded")
        
        # Parse the description to see if it contains explicit shape parameters
        shape_params = self._parse_shape_params(description)
        
        # If we have a direct shape request with parameters, use the standard generation functions
        if shape_params is not None:
            shape_name, params = shape_params
            point_cloud = generate_point_cloud(shape_name, **params)
            return point_cloud
        
        # Otherwise, use the trained generator model
        # Encode the description
        encoded_text = torch.tensor(self.tokenizer.encode(description)).unsqueeze(0).to(self.device)
        
        # Generate point cloud
        self.generator.eval()
        with torch.no_grad():
            # Check if we're using the enhanced model by checking for multiple returns
            try:
                # Try with original generator first
                generated = self.generator(encoded_text)
                # If this is a single return value (the point cloud)
                if not isinstance(generated, tuple):
                    point_cloud = generated[0].cpu().numpy()
                else:
                    # It returned multiple values (point cloud and shape logits)
                    point_cloud = generated[0][0].cpu().numpy()
            except TypeError as e:
                if "attention_mask" in str(e):
                    # Enhanced generator requires attention mask
                    attention_mask = torch.ones_like(encoded_text).to(self.device)
                    result = self.generator(encoded_text, attention_mask=attention_mask)
                    point_cloud = result[0][0].cpu().numpy()
                else:
                    # Some other error
                    raise e
        
        return point_cloud
    
    def generate_shape(self, description):
        """
        Generate a point cloud based on a text description.
        
        Args:
            description: Text description of the desired shape
            
        Returns:
            Tensor containing the generated point cloud
        """
        if not self.generator or not self.tokenizer:
            raise RuntimeError("Generator model or tokenizer not initialized")
            
        # Convert description to tensor
        tokens = self.tokenizer.encode(description.lower())
        token_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Generate point cloud
        self.generator.eval()
        with torch.no_grad():
            try:
                # Generate points using the model
                result = self.generator(token_tensor)
                
                # Handle different return types (handle both tensor and tuple returns)
                if isinstance(result, tuple):
                    points = result[0]  # Extract the point cloud from the tuple
                else:
                    points = result
                    
                # Check if points is still a tuple (nested tuple case)
                if isinstance(points, tuple):
                    points = points[0]
                
                # Check if the points are empty
                if points is None or (hasattr(points, 'numel') and points.numel() == 0):
                    # Extract shape type from description
                    shape_type = None
                    for shape in self.shape_types:
                        if shape in description.lower():
                            shape_type = shape
                            break
                    
                    if shape_type:
                        # Generate basic shape
                        points = generate_point_cloud(shape_type, num_points=1024)
                        points = torch.tensor(points).float().unsqueeze(0).to(self.device)
                    else:
                        raise ValueError(f"Could not determine shape type from description: {description}")
                
                return points
                
            except Exception as e:
                print(f"Error in shape generation: {e}")
                raise
    
    def _parse_shape_params(self, text):
        """Parse shape parameters from text
        
        Args:
            text: Text description
            
        Returns:
            Tuple of (shape_name, params_dict) or None if no recognized pattern
        """
        text = text.lower()
        
        # Check if this is a direct shape request
        for shape in self.shape_types:
            if shape in text:
                # Extract parameters using regex
                params = {"num_points": 1024}  # Default number of points
                
                # Extract numeric parameters
                for param_name, pattern in self.param_patterns.items():
                    match = re.search(pattern, text)
                    if match:
                        params[param_name] = float(match.group(1))
                
                return shape, params
        
        # This is not a direct shape request with parameters
        return None
    
    def save(self, directory):
        """Save models and tokenizer
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save classifier if available
        if self.classifier:
            try:
                torch.save(self.classifier.state_dict(), os.path.join(directory, "classifier.pth"))
                print(f"Classifier saved to {os.path.join(directory, 'classifier.pth')}")
            except Exception as e:
                print(f"Error saving classifier: {e}")
        
        # Save generator if available
        if self.generator:
            try:
                torch.save(self.generator.state_dict(), os.path.join(directory, "generator.pth"))
                print(f"Generator saved to {os.path.join(directory, 'generator.pth')}")
            except Exception as e:
                print(f"Error saving generator: {e}")
        
        # Save tokenizer if available
        if self.tokenizer:
            try:
                self.tokenizer.save(os.path.join(directory, "tokenizer.json"))
                print(f"Tokenizer saved to {os.path.join(directory, 'tokenizer.json')}")
            except Exception as e:
                print(f"Error saving tokenizer: {e}")
        
        # Save shape types
        try:
            with open(os.path.join(directory, "shape_types.json"), "w") as f:
                json.dump(self.shape_types, f)
            print(f"Shape types saved to {os.path.join(directory, 'shape_types.json')}")
        except Exception as e:
            print(f"Error saving shape types: {e}")
    
    @classmethod
    def load(cls, directory, device='cuda'):
        """Load models and tokenizer
        
        Args:
            directory: Directory to load from
            device: Device to run on ('cuda' or 'cpu')
            
        Returns:
            ShapeAI instance
        """
        print(f"Loading models from directory: {directory}")
        
        # Load shape types
        shape_types_path = os.path.join(directory, "shape_types.json")
        if not os.path.exists(shape_types_path):
            raise FileNotFoundError(f"Shape types file not found at {shape_types_path}")
            
        with open(shape_types_path, "r") as f:
            shape_types = json.load(f)
        print(f"Loaded shape types: {shape_types}")
        
        # Initialize tokenizer
        tokenizer = SimpleTokenizer()
        tokenizer_path = os.path.join(directory, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer.load(tokenizer_path)
            print("Loaded tokenizer successfully")
        else:
            print(f"Warning: Tokenizer file not found at {tokenizer_path}")
        
        # Examine model parameters
        classifier_path = os.path.join(directory, "classifier.pth")
        generator_path = os.path.join(directory, "generator.pth")
        
        classifier_params = None
        if os.path.exists(classifier_path):
            classifier_params = examine_model_params(classifier_path)
        
        generator_params = None
        if os.path.exists(generator_path):
            generator_params = examine_model_params(generator_path)
        
        # Initialize classifier
        classifier = None
        if os.path.exists(classifier_path):
            try:
                # Use detected parameters if available
                if classifier_params and "num_classes" in classifier_params:
                    num_classes = classifier_params["num_classes"]
                else:
                    num_classes = len(shape_types)
                    
                print(f"Creating classifier with {num_classes} classes")
                classifier = PointCloudNetwork(num_classes=num_classes)
                classifier.load_state_dict(torch.load(classifier_path, map_location=device))
                print("Loaded classifier model successfully")
            except Exception as e:
                print(f"Error loading classifier model: {e}")
        else:
            print(f"Warning: Classifier model not found at {classifier_path}")
        
        # Initialize generator
        generator = None
        if os.path.exists(generator_path):
            try:
                # Use detected parameters if available
                if generator_params and "num_points" in generator_params:
                    num_points = generator_params["num_points"]
                    print(f"Creating generator with {num_points} output points")
                    generator = PointCloudGenerator(num_points=num_points)
                else:
                    generator = PointCloudGenerator()
                    
                generator.load_state_dict(torch.load(generator_path, map_location=device))
                print("Loaded generator model successfully")
            except Exception as e:
                print(f"Error loading generator model: {e}")
        else:
            print(f"Warning: Generator model not found at {generator_path}")
        
        return cls(classifier, generator, tokenizer, shape_types, device)

def examine_model_params(model_path):
    """
    Examine a saved model to determine its parameters.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dict of parameters or None if examination fails
    """
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        params = {}
        
        # For classifier
        if "classifier.8.weight" in state_dict:
            num_classes = state_dict["classifier.8.weight"].shape[0]
            params["num_classes"] = num_classes
            print(f"Detected {num_classes} classes in classifier model")
        
        # For generator
        if "fc_generator.3.weight" in state_dict:
            output_size = state_dict["fc_generator.3.weight"].shape[0]
            num_points = output_size // 3
            params["num_points"] = num_points
            print(f"Detected {num_points} points in generator model")
            
        return params
    except Exception as e:
        print(f"Error examining model: {e}")
        return None
