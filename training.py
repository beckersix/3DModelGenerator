"""
Training functions for the point cloud classification and generation models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def train_classifier(model, train_dataloader, val_dataloader=None, 
                    num_epochs=50, learning_rate=0.001, device='cuda'):
    """Train the point cloud classification model
    
    Args:
        model: PointCloudNetwork model
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for point_clouds, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            point_clouds = point_clouds.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(point_clouds)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * point_clouds.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # Average training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for point_clouds, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    point_clouds = point_clouds.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(point_clouds)
                    loss = criterion(outputs, labels)
                    
                    # Track metrics
                    val_loss += loss.item() * point_clouds.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                
                # Average validation metrics
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    return model, history

def train_enhanced_generator(model, tokenizer, point_clouds, descriptions, shape_labels=None,
                            batch_size=16, num_epochs=100, learning_rate=0.0005, device='cuda'):
    """Train the enhanced point cloud generator model
    
    Args:
        model: EnhancedPointCloudGenerator model
        tokenizer: SimpleTokenizer for processing text
        point_clouds: List of point cloud arrays
        descriptions: List of text descriptions
        shape_labels: Optional list of shape labels for supervision
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    
    # Encode descriptions
    encoded_texts = [tokenizer.encode(desc) for desc in descriptions]
    
    # Create dataset (normalize points to same count)
    processed_points = []
    for pc in point_clouds:
        if len(pc) != 1024:
            # Normalize to 1024 points
            if len(pc) > 1024:
                processed_points.append(pc[:1024])
            else:
                # Pad by duplicating with small noise
                needed = 1024 - len(pc)
                indices = np.random.choice(len(pc), needed)
                padding = pc[indices] + np.random.normal(0, 0.01, (needed, 3))
                processed_points.append(np.vstack([pc, padding]))
        else:
            processed_points.append(pc)
    
    # Convert to tensors
    tensor_clouds = [torch.FloatTensor(pc) for pc in processed_points]
    tensor_texts = [torch.LongTensor(txt) for txt in encoded_texts]
    
    # Convert shape labels if provided
    tensor_labels = None
    if shape_labels is not None:
        tensor_labels = [torch.LongTensor([label]) for label in shape_labels]
    
    # Create dataset
    if tensor_labels is not None:
        dataset = list(zip(tensor_texts, tensor_clouds, tensor_labels))
    else:
        dataset = list(zip(tensor_texts, tensor_clouds))
    
    # Batch data manually to avoid collation issues
    def get_batches(dataset, batch_size):
        np.random.shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if len(batch) < batch_size:
                continue  # Skip last incomplete batch
                
            # Pad texts to same length
            if tensor_labels is not None:
                texts, clouds, labels = zip(*batch)
            else:
                texts, clouds = zip(*batch)
                
            max_len = max(txt.size(0) for txt in texts)
            padded_texts = []
            attention_masks = []
            
            for txt in texts:
                if txt.size(0) < max_len:
                    padding = torch.zeros(max_len - txt.size(0), dtype=torch.long)
                    padded_txt = torch.cat([txt, padding])
                    mask = torch.cat([torch.ones(txt.size(0)), torch.zeros(max_len - txt.size(0))])
                else:
                    padded_txt = txt
                    mask = torch.ones(max_len)
                    
                padded_texts.append(padded_txt)
                attention_masks.append(mask)
            
            # Stack into batches
            text_batch = torch.stack(padded_texts).to(device)
            mask_batch = torch.stack(attention_masks).to(device)
            cloud_batch = torch.stack(clouds).to(device)
            
            if tensor_labels is not None:
                label_batch = torch.cat(labels).to(device)
                yield text_batch, cloud_batch, mask_batch, label_batch
            else:
                yield text_batch, cloud_batch, mask_batch
    
    # Define loss function (Chamfer Distance) and optimizer
    def chamfer_distance(pred_points, gt_points):
        """Simple Chamfer Distance implementation"""
        # For each point in pred_points, find the nearest point in gt_points
        pred_to_gt = ((pred_points.unsqueeze(2) - gt_points.unsqueeze(1))**2).sum(3).min(2)[0]
        # For each point in gt_points, find the nearest point in pred_points
        gt_to_pred = ((gt_points.unsqueeze(2) - pred_points.unsqueeze(1))**2).sum(3).min(2)[0]
        # Average in both directions
        return pred_to_gt.mean(1) + gt_to_pred.mean(1)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Define classification loss if shape labels provided
    classification_criterion = nn.CrossEntropyLoss() if tensor_labels is not None else None
    
    # For tracking metrics
    history = {
        'total_loss': [],
        'chamfer_loss': [],
        'classification_loss': [] if tensor_labels is not None else None
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_chamfer_loss = 0.0
        epoch_class_loss = 0.0
        batch_count = 0
        
        # Get iterator with the right number of arguments
        if tensor_labels is not None:
            iterator = get_batches(dataset, batch_size)
        else:
            iterator = get_batches(dataset, batch_size)
        
        for batch_data in tqdm(iterator, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass (with or without labels)
            if tensor_labels is not None:
                texts, clouds, masks, labels = batch_data
                generated_clouds, shape_logits = model(texts, attention_mask=masks)
                
                # Compute Chamfer Distance loss
                chamfer_loss = chamfer_distance(generated_clouds, clouds).mean()
                
                # Compute classification loss
                class_loss = classification_criterion(shape_logits, labels)
                
                # Combined loss (weighting can be adjusted)
                loss = chamfer_loss + 0.1 * class_loss
                
                # Track individual losses
                epoch_chamfer_loss += chamfer_loss.item()
                epoch_class_loss += class_loss.item()
            else:
                texts, clouds, masks = batch_data
                generated_clouds, _ = model(texts, attention_mask=masks)
                
                # Compute Chamfer Distance loss only
                loss = chamfer_distance(generated_clouds, clouds).mean()
                epoch_chamfer_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_total_loss += loss.item()
            batch_count += 1
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Average epoch loss
        if batch_count > 0:
            epoch_total_loss = epoch_total_loss / batch_count
            epoch_chamfer_loss = epoch_chamfer_loss / batch_count
            
            history['total_loss'].append(epoch_total_loss)
            history['chamfer_loss'].append(epoch_chamfer_loss)
            
            if tensor_labels is not None:
                epoch_class_loss = epoch_class_loss / batch_count
                history['classification_loss'].append(epoch_class_loss)
            
            # Print progress
            if tensor_labels is not None:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Total Loss: {epoch_total_loss:.6f}, "
                      f"Chamfer Loss: {epoch_chamfer_loss:.6f}, "
                      f"Class Loss: {epoch_class_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Total Loss: {epoch_total_loss:.6f}, "
                      f"Chamfer Loss: {epoch_chamfer_loss:.6f}")
    
    return model, history

def train_generator(model, tokenizer, point_clouds, descriptions, 
                    batch_size=16, num_epochs=100, learning_rate=0.0005, device='cuda'):
    """Train the point cloud generator model
    
    Args:
        model: PointCloudGenerator model
        tokenizer: SimpleTokenizer for processing text
        point_clouds: List of point cloud arrays
        descriptions: List of text descriptions
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    
    # Encode descriptions
    encoded_texts = [tokenizer.encode(desc) for desc in descriptions]
    
    # Create dataset (normalize points to same count)
    processed_points = []
    for pc in point_clouds:
        if len(pc) != 1024:
            # Normalize to 1024 points
            if len(pc) > 1024:
                processed_points.append(pc[:1024])
            else:
                # Pad by duplicating with small noise
                needed = 1024 - len(pc)
                indices = np.random.choice(len(pc), needed)
                padding = pc[indices] + np.random.normal(0, 0.01, (needed, 3))
                processed_points.append(np.vstack([pc, padding]))
        else:
            processed_points.append(pc)
    
    # Convert to tensors
    tensor_clouds = [torch.FloatTensor(pc) for pc in processed_points]
    tensor_texts = [torch.LongTensor(txt) for txt in encoded_texts]
    
    # Create dataset
    dataset = list(zip(tensor_texts, tensor_clouds))
    
    # Batch data manually to avoid collation issues
    def get_batches(dataset, batch_size):
        np.random.shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if len(batch) < batch_size:
                continue  # Skip last incomplete batch
                
            # Pad texts to same length
            texts, clouds = zip(*batch)
            max_len = max(txt.size(0) for txt in texts)
            padded_texts = []
            for txt in texts:
                if txt.size(0) < max_len:
                    padding = torch.zeros(max_len - txt.size(0), dtype=torch.long)
                    padded_texts.append(torch.cat([txt, padding]))
                else:
                    padded_texts.append(txt)
            
            # Stack into batches
            text_batch = torch.stack(padded_texts).to(device)
            cloud_batch = torch.stack(clouds).to(device)
            
            yield text_batch, cloud_batch
    
    # Define loss function (Chamfer Distance) and optimizer
    def chamfer_distance(pred_points, gt_points):
        """Simple Chamfer Distance implementation"""
        # For each point in pred_points, find the nearest point in gt_points
        pred_to_gt = ((pred_points.unsqueeze(2) - gt_points.unsqueeze(1))**2).sum(3).min(2)[0]
        # For each point in gt_points, find the nearest point in pred_points
        gt_to_pred = ((gt_points.unsqueeze(2) - pred_points.unsqueeze(1))**2).sum(3).min(2)[0]
        # Average in both directions
        return pred_to_gt.mean(1) + gt_to_pred.mean(1)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking metrics
    history = {
        'loss': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for texts, clouds in tqdm(get_batches(dataset, batch_size), 
                                  desc=f"Epoch {epoch+1}/{num_epochs}",
                                  total=len(dataset)//batch_size):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            generated_clouds = model(texts)
            
            # Compute Chamfer Distance loss
            loss = chamfer_distance(generated_clouds, clouds).mean()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            batch_count += 1
        
        # Average epoch loss
        if batch_count > 0:
            epoch_loss = epoch_loss / batch_count
            history['loss'].append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f}")
    
    return model, history