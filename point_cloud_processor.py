"""
Point cloud processing and adaptation module for 3D model generation
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import open3d as o3d

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels, metadata=None):
        self.point_clouds = point_clouds  # [N, num_points, 3]
        self.labels = labels
        self.metadata = metadata if metadata is not None else {}
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        label = self.labels[idx]
        meta = {k: v[idx] for k, v in self.metadata.items()} if self.metadata else {}
        return {
            'points': torch.FloatTensor(point_cloud),
            'label': label,
            'metadata': meta
        }

class PointCloudEncoder(nn.Module):
    def __init__(self, num_points=2048, embedding_dim=512):
        super().__init__()
        self.num_points = num_points
        
        # Point-wise feature extraction (matching checkpoint dimensions)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, embedding_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, x):
        x = x.transpose(2, 1)  # [batch_size, 3, num_points]
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        return x.view(-1, x.size(1))

class PointCloudDecoder(nn.Module):
    def __init__(self, embedding_dim=512, num_points=2048):
        super().__init__()
        self.num_points = num_points
        
        # Updated decoder architecture to match checkpoint dimensions
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, num_points * 3)
        
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x.view(-1, self.num_points, 3)

class AdaptivePointCloudGenerator(nn.Module):
    def __init__(self, num_points=2048, embedding_dim=512, latent_dim=256):
        super().__init__()
        self.encoder = PointCloudEncoder(num_points, embedding_dim)
        self.decoder = PointCloudDecoder(embedding_dim, num_points)
        
        # Updated adaptation layers to match checkpoint dimensions
        self.text_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        self.fc_latent = nn.Sequential(
            nn.Linear(1024, 512),  # 512 (encoder) + 512 (text) = 1024
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fc_generator = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),  # Match decoder input dimension
            nn.ReLU()
        )
        
    def forward(self, x, text_embedding=None):
        # Encode input point cloud
        features = self.encoder(x)  # [batch_size, 512]
        
        if text_embedding is not None:
            # Process text embedding
            text_features = self.text_encoder(text_embedding)  # [batch_size, 512]
            
            # Combine features
            combined = torch.cat([features, text_features], dim=1)  # [batch_size, 1024]
            latent = self.fc_latent(combined)  # [batch_size, 512]
            
            # Generate output through decoder
            features = self.fc_generator(latent)  # [batch_size, 512]
        
        output = self.decoder(features)
        return output

def process_point_cloud(file_path, num_points=2048):
    """Process a 3D model file into a point cloud"""
    try:
        # Read the 3D model
        mesh = o3d.io.read_triangle_mesh(file_path)
        
        # Sample points from the mesh
        point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(point_cloud.points)
        
        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / dist
        
        return points
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_metadata(point_cloud):
    """Extract geometric metadata from point cloud"""
    points = np.array(point_cloud)
    
    metadata = {
        'centroid': np.mean(points, axis=0),
        'bbox': {
            'min': np.min(points, axis=0),
            'max': np.max(points, axis=0)
        },
        'volume': np.prod(np.max(points, axis=0) - np.min(points, axis=0)),
        'surface_area_estimate': len(points) * np.mean(np.std(points, axis=0))
    }
    
    return metadata
