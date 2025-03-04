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
        if not description:
            raise ValueError("Description cannot be empty")
            
        try:
            # First try to use the neural model for generation
            if self.generator and self.tokenizer:
                # Generate using the AI model
                try:
                    points = self.generate_shape(description)
                    # Check if returned tensor and convert to numpy
                    if isinstance(points, torch.Tensor):
                        return points.squeeze(0).cpu().numpy()
                    return points
                except Exception as e:
                    print(f"AI model generation failed, falling back to rule-based: {e}")
            
            # Fall back to rule-based generation
            shape_info = self._parse_shape_params(description)
            if shape_info:
                shape_type, params = shape_info
                return generate_point_cloud(shape_type, **params)
            else:
                # If no shape recognized, try to guess based on description
                for shape in self.shape_types:
                    if shape.lower() in description.lower():
                        print(f"Generating default {shape} (no specific parameters found)")
                        return generate_point_cloud(shape)
                
                # No recognizable shape in description, generate a cube as fallback
                print("No specific shape recognized. Generating a default cube.")
                return generate_point_cloud("cube")
                
        except Exception as e:
            print(f"Error in shape generation: {e}")
            # Last resort fallback
            print("Falling back to default cube")
            return generate_point_cloud("cube")
    
    def generate_shape(self, description):
        """
        Generate a point cloud from a text description using neural model.
        
        Args:
            description: Text description of the desired shape
            
        Returns:
            Generated point cloud as numpy array
        """
        if self.generator is None:
            raise ValueError("Generator model not initialized")
            
        try:
            # Tokenize the description
            encoded_text = self.tokenizer.encode(description)
            encoded_tensor = torch.tensor(encoded_text).unsqueeze(0).to(self.device)
            
            # Generate point cloud using the model
            self.generator.eval()
            with torch.no_grad():
                try:
                    # First try the standard forward method
                    output = self.generator(encoded_tensor)
                    if isinstance(output, tuple):
                        # Model returns (point_cloud, shape_logits)
                        point_cloud = output[0]
                    else:
                        # Model returns just point_cloud
                        point_cloud = output
                        
                except Exception as e:
                    if "attention_mask" in str(e):
                        # Some models require attention mask
                        attention_mask = torch.ones_like(encoded_tensor).to(self.device)
                        output = self.generator(encoded_tensor, attention_mask=attention_mask)
                        if isinstance(output, tuple):
                            point_cloud = output[0]
                        else:
                            point_cloud = output
                    else:
                        raise e
                        
            # Extract points and convert to numpy
            if isinstance(point_cloud, torch.Tensor):
                point_cloud = point_cloud.squeeze(0).cpu().numpy()
                
            return point_cloud
            
        except Exception as e:
            print(f"Error in neural generation: {e}")
            return None
    
    def _parse_shape_params(self, text):
        """Parse shape parameters from text
        
        Args:
            text: Text description
            
        Returns:
            Tuple of (shape_name, params_dict) or None if no recognized pattern
        """
        text = text.lower()
        
        # Dictionary mapping shape variations to standard shapes
        shape_variants = {
            "cube": ["cube", "box", "square", "block"],
            "rectangular_prism": ["rectangular prism", "rect prism", "rectangle", "rectangular", "brick", "block"],
            "sphere": ["sphere", "ball", "orb", "globe"],
            "cylinder": ["cylinder", "tube", "pipe", "column"],
            "pyramid": ["pyramid", "triangular", "tetrahedron"],
            "torus": ["torus", "donut", "doughnut", "ring"],
            "cone": ["cone", "conical"],
            "ellipsoid": ["ellipsoid", "oval", "egg", "elliptical"],
            "capsule": ["capsule", "pill"],
            "star": ["star", "stellated", "radial"],
            "helix": ["helix", "spiral", "coil", "spring"]
        }
        
        # Check for shape types in the description
        detected_shape = None
        for shape_name, variants in shape_variants.items():
            for variant in variants:
                if variant in text:
                    detected_shape = shape_name
                    break
            if detected_shape:
                break
        
        if not detected_shape:
            return None
            
        # Initialize parameters with defaults
        params = {"num_points": 1024}
        
        # Extract size parameters
        # General size parameter
        size_match = re.search(r'size\s*[=:]\s*(\d+\.?\d*)', text) or re.search(r'(\d+\.?\d*)\s*(?:unit|size)', text)
        if size_match:
            params["size"] = float(size_match.group(1))
        
        # Extract specific dimensions
        for param_name, pattern_list in {
            "width": [r'width\s*[=:]\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:units?|inches?|cm)?\s*wide'],
            "height": [r'height\s*[=:]\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:units?|inches?|cm)?\s*(?:tall|high)'],
            "depth": [r'depth\s*[=:]\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:units?|inches?|cm)?\s*deep'],
            "radius": [r'radius\s*[=:]\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:units?|inches?|cm)?\s*radius'],
            "major_radius": [r'major[_\s]radius\s*[=:]\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:units?|inches?|cm)?\s*major radius'],
            "minor_radius": [r'minor[_\s]radius\s*[=:]\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:units?|inches?|cm)?\s*minor radius'],
            "base_size": [r'base[_\s]size\s*[=:]\s*(\d+\.?\d*)', r'base\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:units?|inches?|cm)?\s*base'],
        }.items():
            for pattern in pattern_list:
                match = re.search(pattern, text)
                if match:
                    params[param_name] = float(match.group(1))
                    break
        
        # Extract noise level
        noise_match = re.search(r'noise\s*[=:]\s*(\d+\.?\d*)', text) or re.search(r'(\d+\.?\d*)\s*noise', text)
        if noise_match:
            params["noise"] = float(noise_match.group(1))
        else:
            # Set a default noise value
            params["noise"] = 0.05
        
        # Shape-specific parameter inference
        if detected_shape == "rectangular_prism":
            # Map to the correct parameter names for rectangular prism
            if "width" in params:
                params["size_x"] = params.pop("width")
            if "height" in params:
                params["size_y"] = params.pop("height")
            if "depth" in params:
                params["size_z"] = params.pop("depth")
            # If only general size is given, create non-uniform dimensions
            if "size" in params and "size_x" not in params:
                base_size = params.pop("size")
                params["size_x"] = base_size
                params["size_y"] = base_size * 1.5  # Make it rectangular, not cubic
                params["size_z"] = base_size * 0.8
        
        # Handle special cases for torus
        if detected_shape == "torus" and "radius" in params and "major_radius" not in params:
            params["major_radius"] = params.pop("radius")
            params["minor_radius"] = params.get("minor_radius", params["major_radius"] / 3)
        
        # Handle special cases for cylinder
        if detected_shape == "cylinder" and "height" not in params and "size" in params:
            params["height"] = params["size"] * 2  # Default height is twice the size

        # Handle adjectives that imply dimensions
        if "tall" in text or "high" in text:
            if detected_shape in ["cylinder", "cone", "pyramid"]:
                params["height"] = params.get("height", 2.0) * 1.5
        
        if "wide" in text or "broad" in text:
            if detected_shape in ["cylinder", "cone", "torus"]:
                params["radius"] = params.get("radius", 1.0) * 1.5
            elif detected_shape == "rectangular_prism":
                params["size_x"] = params.get("size_x", 1.0) * 1.5
        
        if "thin" in text or "narrow" in text:
            if detected_shape in ["cylinder", "cone", "torus"]:
                params["radius"] = params.get("radius", 1.0) * 0.7
        
        # Handle hollow shapes
        if "hollow" in text and detected_shape in ["sphere", "cylinder", "cube", "rectangular_prism"]:
            params["hollow"] = True
            params["shell_thickness"] = 0.1  # Default shell thickness
            
            # Extract shell thickness if specified
            thickness_match = re.search(r'thickness\s*[=:]\s*(\d+\.?\d*)', text) or re.search(r'(\d+\.?\d*)\s*(?:units?|inches?|cm)?\s*thick', text)
            if thickness_match:
                params["shell_thickness"] = float(thickness_match.group(1))
        
        return (detected_shape, params)
    
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
