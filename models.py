"""
Neural network models for 3D shape classification and generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class PointCloudNetwork(nn.Module):
    def __init__(self, num_points=1024, num_classes=6):
        """
        A simplified PointNet-inspired network for point cloud classification
        
        Args:
            num_points: Number of points in the point cloud
            num_classes: Number of shape classes to identify
        """
        super().__init__()
        
        # Input transformation network
        self.input_transform = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        # Global feature extraction
        self.global_features = nn.Sequential(
            nn.MaxPool1d(num_points),
            nn.Flatten()
        )
        
        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input point cloud of shape (batch_size, num_points, 3)
        
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Ensure correct dimensions (B, 3, N)
        x = x.transpose(2, 1)
        
        # Feature extraction
        x = self.input_transform(x)
        
        # Global feature pooling
        x = self.global_features(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


class PointCloudGenerator(nn.Module):
    def __init__(self, latent_dim=128, text_embed_dim=32, num_points=1024):
        """
        A model that generates point clouds from text descriptions
        
        Args:
            latent_dim: Dimension of the latent space
            text_embed_dim: Dimension of the text embedding
            num_points: Number of points in the output point cloud
        """
        super().__init__()
        
        self.num_points = num_points
        
        # Text encoder (simple embedding layer with a vocabulary of common words)
        vocab_size = 5000  # Adjust based on your text corpus
        self.embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, text_embed_dim * 2)
        )
        
        # Combine text features with random noise
        self.latent_dim = latent_dim
        self.fc_latent = nn.Sequential(
            nn.Linear(text_embed_dim * 2 + latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Point cloud generator (MLP to generate 3D coordinates)
        self.fc_generator = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )
    
    def encode_text(self, word_indices):
        """
        Encode text input into a feature vector
        
        Args:
            word_indices: Tensor of word indices (batch_size, seq_len)
            
        Returns:
            Text embedding (batch_size, text_embed_dim * 2)
        """
        # Get word embeddings
        embeds = self.embedding(word_indices)  # (batch_size, seq_len, embed_dim)
        
        # Average word embeddings
        text_features = embeds.mean(dim=1)  # (batch_size, embed_dim)
        
        # Encode text features
        encoded = self.text_encoder(text_features)  # (batch_size, text_embed_dim * 2)
        
        return encoded
    
    def forward(self, word_indices, z=None):
        """
        Generate a point cloud from text and optional latent vector
        
        Args:
            word_indices: Tensor of word indices (batch_size, seq_len)
            z: Optional latent vectors (batch_size, latent_dim)
            
        Returns:
            Generated point clouds (batch_size, num_points, 3)
        """
        batch_size = word_indices.shape[0]
        
        # Encode text
        text_features = self.encode_text(word_indices)
        
        # Generate or use provided latent vector
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=word_indices.device)
        
        # Concatenate text features with latent vector
        latent = torch.cat([text_features, z], dim=1)
        
        # Generate point cloud
        features = self.fc_latent(latent)
        point_cloud_flat = self.fc_generator(features)
        
        # Reshape to (batch_size, num_points, 3)
        point_cloud = point_cloud_flat.view(batch_size, self.num_points, 3)
        
        return point_cloud

class EnhancedPointCloudGenerator(nn.Module):
    def __init__(self, latent_dim=256, text_embed_dim=64, num_points=1024):
        """
        Enhanced model that generates point clouds from text descriptions
        
        Args:
            latent_dim: Dimension of the latent space
            text_embed_dim: Dimension of the text embedding
            num_points: Number of points in the output point cloud
        """
        super().__init__()
        
        self.num_points = num_points
        
        # Text encoder (larger embedding and deeper processing)
        vocab_size = 5000
        self.embedding = nn.Embedding(vocab_size, text_embed_dim)
        
        # Text processing with bi-directional LSTM for better semantic understanding
        self.text_lstm = nn.LSTM(
            input_size=text_embed_dim,
            hidden_size=text_embed_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Text feature processor
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embed_dim * 2, 256),  # *2 for bidirectional
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, text_embed_dim * 4)
        )
        
        # Combine text features with random noise
        self.latent_dim = latent_dim
        self.fc_latent = nn.Sequential(
            nn.Linear(text_embed_dim * 4 + latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2)
        )
        
        # Point cloud generator (MLP to generate 3D coordinates)
        self.fc_generator = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_points * 3)
        )
        
        # Shape conditioning network
        self.shape_condition = nn.Sequential(
            nn.Linear(text_embed_dim * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 6)  # For 6 basic shape types
        )
    
    def encode_text(self, word_indices, attention_mask=None):
        """
        Encode text input into a feature vector
        
        Args:
            word_indices: Tensor of word indices (batch_size, seq_len)
            attention_mask: Optional mask for padding tokens
            
        Returns:
            Text embedding (batch_size, text_embed_dim * 4)
        """
        batch_size, seq_len = word_indices.shape
        
        # Get word embeddings
        embeds = self.embedding(word_indices)  # (batch_size, seq_len, embed_dim)
        
        # Process with LSTM
        if attention_mask is None:
            # Create mask for non-zero indices (assume 0 is padding)
            attention_mask = (word_indices != 0).float()
            
        # Pack the sequence to handle variable lengths
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, 
            attention_mask.sum(1).cpu().int(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process with LSTM
        lstm_out, (hidden, _) = self.text_lstm(packed_embeds)
        
        # Get final hidden states from both directions
        hidden = hidden.view(2, 2, batch_size, -1)  # [num_layers, num_directions, batch, hidden_size]
        final_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)  # [batch, hidden_size*2]
        
        # Encode text features
        encoded = self.text_encoder(final_hidden)  # (batch_size, text_embed_dim * 4)
        
        return encoded
    
    def forward(self, word_indices, z=None, attention_mask=None):
        """
        Generate a point cloud from text and optional latent vector
        
        Args:
            word_indices: Tensor of word indices (batch_size, seq_len)
            z: Optional latent vectors (batch_size, latent_dim)
            attention_mask: Optional mask for padding tokens
            
        Returns:
            Generated point clouds (batch_size, num_points, 3)
        """
        batch_size = word_indices.shape[0]
        
        # Encode text
        text_features = self.encode_text(word_indices, attention_mask)
        
        # Generate or use provided latent vector
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=word_indices.device)
        
        # Predict shape category conditioning signal
        shape_cond = self.shape_condition(text_features)
        
        # Concatenate text features with latent vector
        latent = torch.cat([text_features, z], dim=1)
        
        # Generate point cloud
        features = self.fc_latent(latent)
        point_cloud_flat = self.fc_generator(features)
        
        # Reshape to (batch_size, num_points, 3)
        point_cloud = point_cloud_flat.view(batch_size, self.num_points, 3)
        
        return point_cloud, shape_cond