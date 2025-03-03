"""
Enhanced language interface with learning capabilities.
Builds on existing CommandInterpreter to add user interaction learning.
"""
import os
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from command_interpreter import CommandInterpreter


class UserInteractionDataset(Dataset):
    """Dataset for user interactions and commands"""
    def __init__(self, interactions, tokenizer, max_length=50):
        self.interactions = interactions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        input_text = interaction["input"]
        command_type = interaction["command_type"]
        
        # Encode input text
        input_tokens = self.tokenizer.encode(input_text, max_len=self.max_length)
        
        # One-hot encode the command type
        command_idx = COMMAND_TYPES.index(command_type)
        
        return torch.tensor(input_tokens), torch.tensor(command_idx)


class CommandClassifier(nn.Module):
    """Model to classify user input into command types"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=7):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # LSTM returns output and (h_n, c_n)
        output, (hidden, _) = self.lstm(embedded)
        
        # Concatenate the last hidden state from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Fully connected layers
        x = self.dropout(hidden)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class IntelligentTokenizer:
    """Enhanced tokenizer with vocabulary adaptation"""
    def __init__(self, vocab_size=5000, min_count=2):
        self.vocab_size = vocab_size
        self.min_count = min_count
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counter = {}
        self.next_idx = 2
        
    def _tokenize(self, text):
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def update_vocab(self, texts):
        """Update vocabulary with new texts"""
        # Count words
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_counter[word] = self.word_counter.get(word, 0) + 1
        
        # Rebuild vocabulary with frequency threshold
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.next_idx = 2
        
        filtered_words = [(word, count) for word, count in self.word_counter.items() 
                          if count >= self.min_count]
        
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
        
        for word, _ in sorted_words[:self.vocab_size - 2]:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.next_idx
                self.idx_to_word[self.next_idx] = word
                self.next_idx += 1
                
        return len(self.word_to_idx)
    
    def encode(self, text, max_len=50):
        """Convert text to token indices"""
        words = self._tokenize(text)
        indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad or truncate
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            
        return indices
    
    def decode(self, indices):
        """Convert indices back to text"""
        return ' '.join([self.idx_to_word.get(idx, "<UNK>") for idx in indices if idx != 0])
    
    def save(self, filepath):
        """Save tokenizer state"""
        data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": {str(k): v for k, v in self.idx_to_word.items()},
            "word_counter": self.word_counter,
            "next_idx": self.next_idx,
            "vocab_size": self.vocab_size,
            "min_count": self.min_count
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath):
        """Load tokenizer state"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.word_to_idx = data["word_to_idx"]
        self.idx_to_word = {int(k): v for k, v in data["idx_to_word"].items()}
        self.word_counter = data.get("word_counter", {})
        self.next_idx = data["next_idx"]
        self.vocab_size = data.get("vocab_size", 5000)
        self.min_count = data.get("min_count", 2)


# List of command types
COMMAND_TYPES = ["create", "generate", "classify", "convert", "help", "quit", "unknown"]

class LearningCommandInterpreter:
    """Enhanced command interpreter with learning capabilities"""
    def __init__(self, base_interpreter=None, model_path=None):
        # Initialize base interpreter
        self.base_interpreter = base_interpreter or CommandInterpreter()
        
        # Initialize dataset for training
        self.interaction_history = []
        self.tokenizer = IntelligentTokenizer()
        
        # Initialize classifier model
        self.classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def interpret_command(self, text):
        """Interpret a command using both base interpreter and learned model"""
        if self.classifier is None or len(self.interaction_history) < 50:
            # Use only base interpreter if model not trained yet
            command = self.base_interpreter.interpret_command(text)
        else:
            # Try to use the learned model
            try:
                # Encode the input
                tokens = torch.tensor([self.tokenizer.encode(text)]).to(self.device)
                
                # Get prediction
                self.classifier.eval()
                with torch.no_grad():
                    outputs = self.classifier(tokens)
                    _, predicted = torch.max(outputs, 1)
                    predicted_command = COMMAND_TYPES[predicted.item()]
                
                # Generate base command for comparison
                base_command = self.base_interpreter.interpret_command(text)
                
                # Use predicted command if confidence is high enough
                probs = torch.softmax(outputs, dim=1)[0]
                confidence = probs[predicted].item()
                
                if confidence > 0.7:
                    # Use the learned command type but with parameters from base
                    command = base_command.copy()
                    command["command"] = predicted_command
                else:
                    # Fall back to base interpreter
                    command = base_command
            except Exception as e:
                print(f"Error using learned model: {e}")
                # Fall back to base interpreter
                command = self.base_interpreter.interpret_command(text)
        
        # Store interaction for learning
        self.interaction_history.append({
            "input": text,
            "command_type": command["command"]
        })
        
        return command
    
    def add_correction(self, text, correct_command):
        """Add a correction to the training data"""
        # Find the last matching input and update its command
        for interaction in reversed(self.interaction_history):
            if interaction["input"] == text:
                interaction["command_type"] = correct_command
                print(f"Added correction: '{text}' → {correct_command}")
                break
        else:
            # If not found, add as new
            self.interaction_history.append({
                "input": text,
                "command_type": correct_command
            })
            print(f"Added new example: '{text}' → {correct_command}")
    
    def train(self, batch_size=16, epochs=10, learning_rate=0.001):
        """Train the model on collected interactions"""
        if len(self.interaction_history) < 20:
            print("Not enough interaction data to train (need at least 20 examples)")
            return False
        
        print(f"Training on {len(self.interaction_history)} interactions...")
        
        # Update tokenizer with new data
        texts = [interaction["input"] for interaction in self.interaction_history]
        vocab_size = self.tokenizer.update_vocab(texts)
        
        # Create dataset
        dataset = UserInteractionDataset(self.interaction_history, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model if needed
        if self.classifier is None or vocab_size != len(self.tokenizer.word_to_idx):
            self.classifier = CommandClassifier(vocab_size=vocab_size, num_classes=len(COMMAND_TYPES))
        
        self.classifier = self.classifier.to(self.device)
        
        # Define optimizer and loss
        optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.classifier(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track stats
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == targets).sum().item()
                epoch_total += targets.size(0)
            
            # Print epoch stats
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {epoch_loss/len(dataloader):.4f}, "
                  f"Accuracy: {100*epoch_correct/epoch_total:.2f}%")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        # Final stats
        print(f"Training completed - "
              f"Average Loss: {total_loss/(epochs*len(dataloader)):.4f}, "
              f"Accuracy: {100*correct/total:.2f}%")
        
        return True
    
    def generate_response(self, command_result):
        """Generate a response using the base interpreter"""
        return self.base_interpreter.generate_response(command_result)
    
    def save(self, directory="models"):
        """Save the learned model and tokenizer"""
        os.makedirs(directory, exist_ok=True)
        
        # Save interaction history
        history_path = os.path.join(directory, "interaction_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.interaction_history, f)
        
        # Save tokenizer
        tokenizer_path = os.path.join(directory, "nl_tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        
        # Save model if available
        if self.classifier is not None:
            model_path = os.path.join(directory, "command_classifier.pth")
            torch.save(self.classifier.state_dict(), model_path)
            
        print(f"Learning interface saved to {directory}")
    
    def load(self, directory="models"):
        """Load the learned model and tokenizer"""
        try:
            # Load interaction history
            history_path = os.path.join(directory, "interaction_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.interaction_history = json.load(f)
            
            # Load tokenizer
            tokenizer_path = os.path.join(directory, "nl_tokenizer.json")
            if os.path.exists(tokenizer_path):
                self.tokenizer.load(tokenizer_path)
            
            # Load model if available
            model_path = os.path.join(directory, "command_classifier.pth")
            if os.path.exists(model_path):
                vocab_size = len(self.tokenizer.word_to_idx)
                self.classifier = CommandClassifier(vocab_size=vocab_size, num_classes=len(COMMAND_TYPES))
                self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                self.classifier = self.classifier.to(self.device)
                
            print(f"Learning interface loaded from {directory}")
            print(f"Loaded {len(self.interaction_history)} previous interactions")
            return True
        except Exception as e:
            print(f"Error loading learning interface: {e}")
            return False


def run_learning_interface(shape_ai=None, model_path="models"):
    """Run the learning interface with the user"""
    # Initialize the interpreter
    base_interpreter = CommandInterpreter()
    learning_interpreter = LearningCommandInterpreter(base_interpreter, model_path)
    
    print("\n=== Learning Natural Language Interface ===")
    print("This interface learns from your commands to get better over time.")
    print("Examples:")
    print("- 'Create a large red cube'")
    print("- 'Make a tall cylinder'")
    print("- 'I want a smooth sphere'")
    print("- 'Help me understand how to use this'")
    print("\nSpecial learning commands:")
    print("- '!train' - Train the model on your interactions")
    print("- '!save' - Save the learning model")
    print("- '!correct <command_type>' - Correct the last command interpretation")
    print("- '!stats' - Show learning statistics")
    print("\nType 'exit' or 'quit' to return to the main menu.")
    
    # Training prompt threshold
    training_prompt_threshold = 20
    last_training_count = 0
    
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Skip empty input
        if not user_input.strip():
            continue
        
        # Handle learning-specific commands
        if user_input.startswith("!"):
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command == "!train":
                learning_interpreter.train()
                continue
                
            elif command == "!save":
                learning_interpreter.save(model_path)
                continue
                
            elif command == "!correct" and len(parts) > 1:
                correction = parts[1].strip()
                if correction in COMMAND_TYPES:
                    # Find the last non-learning command
                    for interaction in reversed(learning_interpreter.interaction_history[:-1]):  # Skip this command
                        if not interaction["input"].startswith("!"):
                            learning_interpreter.add_correction(interaction["input"], correction)
                            break
                else:
                    print(f"Invalid command type. Choose from: {', '.join(COMMAND_TYPES)}")
                continue
                
            elif command == "!stats":
                # Show learning statistics
                command_counts = {}
                for interaction in learning_interpreter.interaction_history:
                    if not interaction["input"].startswith("!"):  # Skip learning commands
                        cmd = interaction["command_type"]
                        command_counts[cmd] = command_counts.get(cmd, 0) + 1
                
                print("\nLearning Statistics:")
                print(f"Total interactions: {len(learning_interpreter.interaction_history)}")
                print(f"Vocabulary size: {len(learning_interpreter.tokenizer.word_to_idx)}")
                print("\nCommand distribution:")
                for cmd, count in sorted(command_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {cmd}: {count} ({100*count/len(learning_interpreter.interaction_history):.1f}%)")
                continue
        
        # Interpret the command
        command = learning_interpreter.interpret_command(user_input)
        
        # Generate response
        response = learning_interpreter.generate_response(command)
        print(response)
        
        # Process the command (reuse code from main-app.py)
        if command["command"] == "quit":
            return  # Return to main menu
        
        # After processing, check if we should prompt for training
        interactions_count = len(learning_interpreter.interaction_history)
        if (interactions_count >= training_prompt_threshold and 
            interactions_count - last_training_count >= 10):
            print("\nI've learned from several new interactions. Would you like me to train my understanding? (y/n)")
            if input().lower().startswith('y'):
                learning_interpreter.train()
                learning_interpreter.save(model_path)
                last_training_count = interactions_count