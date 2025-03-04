"""
Utilities for dataset management and tokenization.
"""
import os
import json
import re
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels=None, transform=None, num_points=1024):
        """
        Dataset for point clouds and optional labels
        
        Args:
            point_clouds: List of point cloud arrays (each of shape (num_points, 3))
            labels: Optional list of integer labels or text descriptions
            transform: Optional transform to apply to point clouds
            num_points: Target number of points for each point cloud (for padding/truncation)
        """
        self.point_clouds = point_clouds
        self.labels = labels
        self.transform = transform
        self.num_points = num_points
        
    def __len__(self):
        return len(self.point_clouds)
    
    def normalize_point_count(self, point_cloud):
        """Ensure point cloud has exactly num_points points"""
        current_points = point_cloud.shape[0]
        
        # If already correct number of points, return as is
        if current_points == self.num_points:
            return point_cloud
        
        # If too many points, truncate
        if current_points > self.num_points:
            return point_cloud[:self.num_points]
        
        # If too few points, pad by duplicating existing points
        if current_points < self.num_points:
            # Calculate number of points to add
            padding_needed = self.num_points - current_points
            
            # Choose random points to duplicate
            indices = np.random.choice(current_points, padding_needed)
            padding = point_cloud[indices]
            
            # Add small random noise to padded points to avoid exact duplicates
            noise = np.random.normal(0, 0.01, (padding_needed, 3))
            padding = padding + noise
            
            # Combine original and padded points
            return np.vstack([point_cloud, padding])
    
    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        
        # Normalize point count
        point_cloud = self.normalize_point_count(point_cloud)
        
        if self.transform:
            point_cloud = self.transform(point_cloud)
        
        # Convert to tensor
        point_cloud = torch.FloatTensor(point_cloud)
        
        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, str):
                # TODO: Convert text to tensor (for text-to-shape generation)
                return point_cloud, label
            else:
                # Classification label
                return point_cloud, torch.tensor(label, dtype=torch.long)
        else:
            return point_cloud


class ShapeDataset(PointCloudDataset):
    """
    Dataset specifically for 3D shape point clouds and shape labels
    This is a subclass of PointCloudDataset that provides specialized
    functionality for working with geometric shapes.
    """
    def __init__(self, point_clouds, labels, transform=None, num_points=1024):
        """
        Initialize a dataset for 3D shape point clouds
        
        Args:
            point_clouds: List of point cloud arrays (each of shape (n, 3))
            labels: List of integer labels (shape indices)
            transform: Optional transform to apply to point clouds
            num_points: Target number of points for each point cloud
        """
        super().__init__(point_clouds, labels, transform, num_points)

    def __getitem__(self, idx):
        """
        Get a point cloud and its corresponding shape label
        
        Returns:
            point_cloud: Tensor of shape (num_points, 3)
            label: Tensor containing the shape label
        """
        return super().__getitem__(idx)


class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counter = {}
        self.next_idx = 2
    
    def build_vocab(self, texts):
        """Build vocabulary from a list of texts"""
        # Count word frequencies
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_counter[word] = self.word_counter.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counter.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Add top words to vocabulary
        for word, _ in sorted_words[:self.vocab_size - 2]:  # -2 for PAD and UNK
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.next_idx
                self.idx_to_word[self.next_idx] = word
                self.next_idx += 1
    
    def _tokenize(self, text):
        """Simple tokenization (lowercase, split on spaces and punctuation)"""
        text = text.lower()
        # Replace punctuation with spaces and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def encode(self, text, max_len=10):
        """Convert text to a list of word indices, padded to max_len"""
        words = self._tokenize(text)
        indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad or truncate to max_len
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))  # Pad with 0 (<PAD>)
        else:
            indices = indices[:max_len]  # Truncate
        
        return indices
    
    def decode(self, indices):
        """Convert a list of word indices back to text"""
        return ' '.join([self.idx_to_word.get(idx, "<UNK>") for idx in indices if idx != 0])
    
    def save(self, filepath):
        """Save the vocabulary to a file"""
        vocab_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": {str(k): v for k, v in self.idx_to_word.items()},  # Convert int keys to strings for JSON
            "next_idx": self.next_idx
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f)
    
    def load(self, filepath):
        """Load the vocabulary from a file"""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.word_to_idx = vocab_data["word_to_idx"]
        self.idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}  # Convert keys back to ints
        self.next_idx = vocab_data["next_idx"]


def generate_enhanced_training_data(num_samples=1000, num_points=1024, output_dir="training_data"):
    """Generate a dataset of point clouds with labels and rich descriptions
    
    Args:
        num_samples: Number of samples to generate
        num_points: Number of points per point cloud
        output_dir: Directory to save the data
    """
    # Import locally to avoid circular imports
    from shape_generation import generate_point_cloud
    
    # Force NumPy to use a single thread to avoid OpenMP conflict
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define shape types and their parameters
    shape_types = [
        "cube", "rectangular_prism", "sphere", "cylinder", "pyramid", "torus"
    ]
    
    # Define descriptive elements for each shape type
    shape_descriptions = {
        "cube": {
            "adjectives": ["perfect", "solid", "cubic", "box-like", "square", "blocky"],
            "size_desc": ["tiny", "small", "medium-sized", "large", "huge", "massive"],
            "colors": ["red", "blue", "green", "white", "black", "yellow", "transparent"],
            "materials": ["wooden", "metal", "plastic", "glass", "stone", "marble", "concrete"],
            "templates": [
                "a {size} {adj} cube",
                "a {material} cube with {size} dimensions",
                "a {color} cubic shape that's {size}",
                "a {adj} {color} cube",
                "a {size} {material} box",
                "a {color} {material} cube" 
            ]
        },
        "rectangular_prism": {
            "adjectives": ["elongated", "narrow", "wide", "flat", "rectangular", "brick-like"],
            "size_desc": ["tiny", "small", "medium-sized", "large", "huge", "oblong"],
            "orientation": ["vertical", "horizontal", "standing", "resting on its side"],
            "materials": ["wooden", "metal", "plastic", "glass", "stone", "concrete"],
            "templates": [
                "a {adj} rectangular prism",
                "a {size} {orientation} rectangular block",
                "a {material} rectangular prism that's {adj}",
                "a {adj} {material} box with unequal sides",
                "a {orientation} {size} brick shape",
                "a {adj} {material} rectangular solid"
            ]
        },
        "sphere": {
            "adjectives": ["perfect", "round", "smooth", "spherical", "globe-like", "ball-shaped"],
            "size_desc": ["tiny", "small", "medium-sized", "large", "huge", "massive"],
            "colors": ["red", "blue", "green", "white", "black", "transparent"],
            "materials": ["wooden", "metal", "plastic", "glass", "rubber", "marble"],
            "templates": [
                "a {size} {adj} sphere",
                "a {material} sphere with {size} radius",
                "a {color} ball shape",
                "a {adj} {color} orb",
                "a {size} {material} spherical object",
                "a {color} {material} sphere"
            ]
        },
        "cylinder": {
            "adjectives": ["tall", "short", "wide", "narrow", "smooth", "cylindrical"],
            "size_desc": ["tiny", "small", "medium-sized", "large", "huge", "massive"],
            "orientation": ["vertical", "horizontal", "standing", "resting on its side"],
            "materials": ["wooden", "metal", "plastic", "glass", "stone", "cardboard"],
            "templates": [
                "a {adj} cylinder",
                "a {size} {orientation} cylindrical shape",
                "a {material} cylinder that's {adj}",
                "a {adj} {material} tube shape",
                "a {orientation} {size} pipe-like form",
                "a {adj} {material} rod"
            ]
        },
        "pyramid": {
            "adjectives": ["pointed", "angular", "triangular", "sharp", "pyramidal", "egyptian"],
            "size_desc": ["tiny", "small", "medium-sized", "large", "huge", "massive"],
            "base_desc": ["square", "rectangular", "wide", "narrow"],
            "materials": ["stone", "metal", "glass", "crystal", "wooden", "plastic"],
            "templates": [
                "a {adj} pyramid",
                "a {size} pyramid with a {base_desc} base",
                "a {material} pyramid that's {adj}",
                "a {adj} {material} triangular structure",
                "a {size} {material} pyramid shape",
                "a {adj} {material} pointed form"
            ]
        },
        "torus": {
            "adjectives": ["ring-shaped", "circular", "donut-like", "rounded", "toroidal", "hollow"],
            "size_desc": ["tiny", "small", "medium-sized", "large", "huge", "massive"],
            "thickness": ["thin", "thick", "slender", "fat", "wide", "narrow"],
            "materials": ["rubber", "metal", "plastic", "glass", "wooden", "stone"],
            "templates": [
                "a {adj} torus",
                "a {size} {thickness} ring shape",
                "a {material} torus that's {thickness}",
                "a {adj} {material} donut shape",
                "a {size} {material} circular ring",
                "a {thickness} {material} toroid"
            ]
        }
    }
    
    point_clouds = []
    shape_labels = []
    descriptions = []
    
    print(f"Generating {num_samples} point clouds...")
    for i in tqdm(range(num_samples)):
        # Randomly select a shape type
        shape_type = random.choice(shape_types)
        shape_label = shape_types.index(shape_type)
        
        # Randomize parameters based on shape type
        if shape_type == "cube":
            size = random.uniform(0.5, 2.0)
            point_cloud = generate_point_cloud(shape_type, num_points, size=size)
            
            # Generate rich description
            shape_desc = shape_descriptions[shape_type]
            template = random.choice(shape_desc["templates"])
            
            # Select random descriptive elements
            adj = random.choice(shape_desc["adjectives"])
            size_word = random.choice(shape_desc["size_desc"])
            color = random.choice(shape_desc["colors"]) if "{color}" in template else ""
            material = random.choice(shape_desc["materials"]) if "{material}" in template else ""
            
            # Format the description
            description = template.format(
                adj=adj, size=size_word, color=color, material=material
            )
            
        elif shape_type == "rectangular_prism":
            size_x = random.uniform(0.5, 2.0)
            size_y = random.uniform(0.5, 2.0)
            size_z = random.uniform(0.5, 2.0)
            point_cloud = generate_point_cloud(shape_type, num_points,
                                              size_x=size_x, size_y=size_y, size_z=size_z)
            
            # Generate rich description
            shape_desc = shape_descriptions[shape_type]
            template = random.choice(shape_desc["templates"])
            
            # Select random descriptive elements
            adj = random.choice(shape_desc["adjectives"])
            size_word = random.choice(shape_desc["size_desc"])
            orientation = random.choice(shape_desc["orientation"]) if "{orientation}" in template else ""
            material = random.choice(shape_desc["materials"]) if "{material}" in template else ""
            
            # Format the description
            description = template.format(
                adj=adj, size=size_word, orientation=orientation, material=material
            )
            
        elif shape_type == "sphere":
            radius = random.uniform(0.5, 1.5)
            point_cloud = generate_point_cloud(shape_type, num_points, radius=radius)
            
            # Generate rich description
            shape_desc = shape_descriptions[shape_type]
            template = random.choice(shape_desc["templates"])
            
            # Select random descriptive elements
            adj = random.choice(shape_desc["adjectives"])
            size_word = random.choice(shape_desc["size_desc"])
            color = random.choice(shape_desc["colors"]) if "{color}" in template else ""
            material = random.choice(shape_desc["materials"]) if "{material}" in template else ""
            
            # Format the description
            description = template.format(
                adj=adj, size=size_word, color=color, material=material
            )
            
        elif shape_type == "cylinder":
            radius = random.uniform(0.5, 1.5)
            height = random.uniform(1.0, 3.0)
            point_cloud = generate_point_cloud(shape_type, num_points, radius=radius, height=height)
            
            # Generate rich description
            shape_desc = shape_descriptions[shape_type]
            template = random.choice(shape_desc["templates"])
            
            # Select random descriptive elements
            adj = random.choice(shape_desc["adjectives"])
            size_word = random.choice(shape_desc["size_desc"])
            orientation = random.choice(shape_desc["orientation"]) if "{orientation}" in template else ""
            material = random.choice(shape_desc["materials"]) if "{material}" in template else ""
            
            # Format the description
            description = template.format(
                adj=adj, size=size_word, orientation=orientation, material=material
            )
            
        elif shape_type == "pyramid":
            base_size = random.uniform(0.8, 2.0)
            height = random.uniform(1.0, 2.5)
            point_cloud = generate_point_cloud(shape_type, num_points, base_size=base_size, height=height)
            
            # Generate rich description
            shape_desc = shape_descriptions[shape_type]
            template = random.choice(shape_desc["templates"])
            
            # Select random descriptive elements
            adj = random.choice(shape_desc["adjectives"])
            size_word = random.choice(shape_desc["size_desc"])
            base_desc = random.choice(shape_desc["base_desc"]) if "{base_desc}" in template else ""
            material = random.choice(shape_desc["materials"]) if "{material}" in template else ""
            
            # Format the description
            description = template.format(
                adj=adj, size=size_word, base_desc=base_desc, material=material
            )
            
        elif shape_type == "torus":
            major_radius = random.uniform(0.8, 1.5)
            minor_radius = random.uniform(0.2, 0.5)
            point_cloud = generate_point_cloud(shape_type, num_points, 
                                              major_radius=major_radius, minor_radius=minor_radius)
            
            # Generate rich description
            shape_desc = shape_descriptions[shape_type]
            template = random.choice(shape_desc["templates"])
            
            # Select random descriptive elements
            adj = random.choice(shape_desc["adjectives"])
            size_word = random.choice(shape_desc["size_desc"])
            thickness = random.choice(shape_desc["thickness"]) if "{thickness}" in template else ""
            material = random.choice(shape_desc["materials"]) if "{material}" in template else ""
            
            # Format the description
            description = template.format(
                adj=adj, size=size_word, thickness=thickness, material=material
            )
        
        # Add random rotation for data augmentation
        rotation = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        point_cloud = np.dot(point_cloud, rotation_matrix)
        
        # Save data
        point_clouds.append(point_cloud)
        shape_labels.append(shape_label)
        descriptions.append(description)
        
        # Save each point cloud to a separate file
        np.save(os.path.join(output_dir, f"pointcloud_{i}.npy"), point_cloud)
    
    # Save metadata (labels and descriptions)
    metadata = {
        "filenames": [f"pointcloud_{i}.npy" for i in range(num_samples)],
        "shape_labels": shape_labels,
        "descriptions": descriptions,
        "shape_types": shape_types
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    print(f"Successfully generated {num_samples} point clouds.")
    print(f"Data saved to {output_dir}")
    
    return point_clouds, shape_labels, descriptions


def load_training_data(data_dir="training_data"):
    """Load a dataset of point clouds with labels and descriptions
    
    Args:
        data_dir: Directory where the data is saved
    
    Returns:
        point_clouds: List of point cloud arrays
        shape_labels: List of shape labels (integers)
        descriptions: List of text descriptions
    """
    # Load metadata
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Extract information
    filenames = metadata["filenames"]
    shape_labels = metadata["shape_labels"]
    descriptions = metadata["descriptions"]
    
    # Load point clouds
    point_clouds = []
    for filename in tqdm(filenames, desc="Loading point clouds"):
        point_cloud = np.load(os.path.join(data_dir, filename))
        point_clouds.append(point_cloud)
    
    return point_clouds, shape_labels, descriptions

# Add missing torch import that's used in the Dataset class
import torch