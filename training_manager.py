"""
Training manager for adaptive 3D model generation
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from point_cloud_processor import (
    PointCloudDataset, 
    AdaptivePointCloudGenerator,
    process_point_cloud,
    extract_metadata
)

class ModelTrainer:
    def __init__(self, model_dir="models", device=None):
        self.model_dir = model_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_training_data(self, model_paths, num_points=2048):
        """Process 3D model files into training data"""
        point_clouds = []
        metadata_list = []
        labels = []
        
        print("Processing 3D models...")
        for idx, path in enumerate(tqdm(model_paths)):
            points = process_point_cloud(path, num_points)
            if points is not None:
                point_clouds.append(points)
                metadata = extract_metadata(points)
                metadata_list.append(metadata)
                labels.append(idx)
        
        return PointCloudDataset(
            np.array(point_clouds),
            np.array(labels),
            metadata_list
        )
    
    def train(self, dataset, batch_size=32, epochs=100, learning_rate=0.001):
        """Train the model on the provided dataset"""
        if self.model is None:
            self.model = AdaptivePointCloudGenerator(
                num_points=2048,
                embedding_dim=512,
                latent_dim=256
            ).to(self.device)
        
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        criterion = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print("Starting training...")
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                points = batch['points'].to(self.device)
                
                # Generate random text embeddings for training (128-dim to match architecture)
                text_embedding = torch.randn(points.size(0), 128).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(points, text_embedding)
                
                # Compute loss
                loss = criterion(output, points)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    def generate_model(self, input_points, description=None, num_variations=1):
        """Generate new 3D model variations based on input"""
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        self.model.eval()
        with torch.no_grad():
            input_points = torch.FloatTensor(input_points).unsqueeze(0).to(self.device)
            variations = []
            
            for _ in range(num_variations):
                # Generate random text embedding if no description provided
                if description is None:
                    text_embedding = torch.randn(1, 128).to(self.device)
                else:
                    # Here you would convert description to embedding
                    # For now, using random embedding
                    text_embedding = torch.randn(1, 128).to(self.device)
                
                output = self.model(input_points, text_embedding)
                variations.append(output.cpu().numpy()[0])
        
        return variations
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            self.model = AdaptivePointCloudGenerator(
                num_points=2048,
                embedding_dim=512,
                latent_dim=256
            ).to(self.device)
        
        # Load state dict with error handling
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print("Warning: Some model parameters did not match:")
            print(str(e))
            print("Attempting to load compatible parameters...")
            
            # Load compatible parameters only
            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters())
            
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Warning: Could not load optimizer state. Using fresh optimizer.")
        
        print(f"Checkpoint loaded from {path}")
