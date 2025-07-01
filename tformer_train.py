import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the repository's data loading components
from datasets.caueeg_script import load_caueeg_task_datasets
from datasets.pipeline import (
    EegRandomCrop, EegDropChannels, EegToTensor, 
    EegToDevice, EegNormalizeAge, EegNormalizeMeanStd,
    eeg_collate_fn
)

class CAUEEGDataLoader:
    """Data loader for CAUEEG dementia classification dataset."""
    
    def __init__(
        self, 
        dataset_path: str,
        batch_size: int = 128,
        crop_length: int = 2000,  # 10 seconds at 200Hz
        device: str = 'cuda',
        file_format: str = 'feather',  # 'feather' is faster than 'edf'
        num_workers: int = 0
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.crop_length = crop_length
        self.device = device
        self.file_format = file_format
        self.num_workers = num_workers
        
        # Load datasets
        self._load_datasets()
        self._create_dataloaders()
    
    def _load_datasets(self):
        """Load the CAUEEG-Dementia datasets."""
        # Define transforms for data preprocessing
        transform = transforms.Compose([
            EegRandomCrop(
                crop_length=self.crop_length,
                latency=2000,  # 10 seconds latency
                multiple=35
            ),
            EegDropChannels([20]),  # Drop photic channel (index 20)
            EegToTensor()
        ])
        
        # Load the dementia task datasets
        self.config_data, self.train_dataset, self.val_dataset, self.test_dataset = \
            load_caueeg_task_datasets(
                dataset_path=self.dataset_path,
                task='dementia',  # 3-class: Normal, MCI, Dementia
                load_event=False,
                file_format=self.file_format,
                transform=transform,
                verbose=True
            )
        
        print(f"Dataset loaded:")
        print(f"- Train samples: {len(self.train_dataset)}")
        print(f"- Validation samples: {len(self.val_dataset)}")
        print(f"- Test samples: {len(self.test_dataset)}")
        print(f"- Classes: {self.config_data['class_label_to_name']}")
        signal_data = self.train_dataset[0]['signal']
        if isinstance(signal_data, list):
            print(f"- Signal shape after preprocessing: {signal_data[0].shape}")
        else:
            print(f"- Signal shape after preprocessing: {signal_data.shape}")
    
    def _create_dataloaders(self):
        """Create PyTorch DataLoaders."""
        # Set up device-specific parameters
        if self.device == 'cpu':
            pin_memory = False
        else:
            pin_memory = True
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            collate_fn=eeg_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            collate_fn=eeg_collate_fn
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            collate_fn=eeg_collate_fn
        )
    
    def _calculate_age_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate age normalization statistics."""
        ages = []
        for sample in self.train_dataset:
            ages.append(sample['age'])
        
        ages = torch.stack(ages)
        return ages.mean().unsqueeze(0), ages.std().unsqueeze(0)
    
    def _calculate_signal_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate signal normalization statistics."""
        all_signals = []
        
        # Sample a subset for efficiency
        sample_size = min(1000, len(self.train_dataset))
        indices = np.random.choice(len(self.train_dataset), sample_size, replace=False)
        
        for idx in indices:
            signal = self.train_dataset[idx]['signal']
            # Handle both single signal and list of signals (multiple crops)
            if isinstance(signal, list):
                all_signals.extend(signal)  # Flatten list of signals
            else:
                all_signals.append(signal)
        
        all_signals = torch.stack(all_signals)  # (batch, channels, time)
        
        # Calculate statistics across batch and time dimensions
        signal_mean = all_signals.mean(dim=(0, 2))  # (channels,)
        signal_std = all_signals.std(dim=(0, 2))    # (channels,)
        
        return signal_mean, signal_std
    
    def get_preprocessing_transforms(self):
        """Get preprocessing transforms for model training."""
        # Calculate normalization statistics from training data
        age_mean, age_std = self._calculate_age_statistics()
        signal_mean, signal_std = self._calculate_signal_statistics()
        
        # Create preprocessing pipeline - simplified approach
        def preprocess_batch(batch):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
                        
            # Normalize signal - proper broadcasting
            signal = batch['signal']  # (batch_size, channels, time) - already stacked by collate_fn
            # Reshape mean and std for proper broadcasting
            mean = signal_mean.view(1, -1, 1).to(self.device)  # (1, channels, 1)
            std = signal_std.view(1, -1, 1).to(self.device)    # (1, channels, 1)
            batch['signal'] = (signal - mean) / (std + 1e-8)
            
            return batch
        
        return preprocess_batch
    
    def get_dataloaders(self):
        """Return the data loaders."""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset configuration information."""
        return {
            'num_classes': len(self.config_data['class_label_to_name']),
            'class_names': self.config_data['class_label_to_name'],
            'class_mapping': self.config_data['class_name_to_label'],
            'num_channels': 20,  # After dropping photic channel
            'sequence_length': self.crop_length,
            'sampling_rate': 200  # Hz
        }

# Example usage and simple transformer training setup
class EEGTransformer(nn.Module):
    """Simple transformer model for EEG classification."""
    
    def __init__(
        self, 
        num_channels: int = 20,
        sequence_length: int = 2000,
        num_classes: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Channel embedding
        self.channel_embedding = nn.Linear(sequence_length, d_model)
        
        # Positional encoding for channels
        self.pos_encoding = nn.Parameter(torch.randn(num_channels, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
                
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: (batch_size, num_channels, sequence_length)
        """
        batch_size, num_channels, seq_len = signal.shape
        
        # Embed each channel's time series
        x = self.channel_embedding(signal)  # (batch, channels, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)

        # Apply transformer
        x = self.transformer(x)  # (batch, channels+1, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits

def evaluate_model(model, dataloader, criterion, preprocess_fn, device):
    """Evaluate model on a dataset and return loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Apply preprocessing
            batch = preprocess_fn(batch)
            
            # Get inputs
            signals = batch['signal'].to(device)
            labels = batch['class_label'].to(device)
            
            # Forward pass
            logits = model(signals)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    """Example usage of the CAUEEG data loader with transformer training."""
    
    # Set up data loader
    dataset_path = "local/datasets/caueeg-dataset"  # Adjust path as needed
    data_loader = CAUEEGDataLoader(
        dataset_path=dataset_path,
        batch_size=32,
        crop_length=1000,  # 10 seconds at 200Hz
        device='cuda' if torch.cuda.is_available() else 'cpu',
        file_format='edf'  # Use feather for speed
    )
    
    # Get data loaders and preprocessing
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    preprocess_fn = data_loader.get_preprocessing_transforms()
    dataset_info = data_loader.get_dataset_info()
    
    print(f"Dataset info: {dataset_info}")
    
    # Initialize model
    model = EEGTransformer(
        num_channels=dataset_info['num_channels'],
        sequence_length=dataset_info['sequence_length'],
        num_classes=dataset_info['num_classes']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print("Starting training...")
    
    # Training loop
    num_epochs = 100
    best_val_acc = 0.0
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for training progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_pbar):
            # Apply preprocessing
            batch = preprocess_fn(batch)
            
            # Get inputs
            signals = batch['signal'].to(device)
            labels = batch['class_label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            
            # Update progress bar
            current_train_acc = 100 * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_train_acc:.2f}%'
            })
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase with progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
        val_loss, val_accuracy = evaluate_model(model, val_pbar, criterion, preprocess_fn, device)
        
        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_eeg_transformer.pth')
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        print("-" * 80)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    
    # Load best model
    model.load_state_dict(torch.load('best_eeg_transformer.pth'))
    
    # Test evaluation
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, preprocess_fn, device)
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")
    
    # Detailed test results by class
    model.eval()
    class_correct = [0] * dataset_info['num_classes']
    class_total = [0] * dataset_info['num_classes']
    
    with torch.no_grad():
        for batch in test_loader:
            batch = preprocess_fn(batch)
            signals = batch['signal'].to(device)
            labels = batch['class_label'].to(device)
            
            logits = model(signals)
            _, predicted = torch.max(logits, 1)
            
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    print("\nPer-class Test Accuracy:")
    for i, class_name in enumerate(dataset_info['class_names']):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_name}: No samples")

def evaluate_model_with_pbar(model, dataloader_pbar, criterion, preprocess_fn, device):
    """Evaluate model with tqdm progress bar."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader_pbar:
            # Apply preprocessing
            batch = preprocess_fn(batch)
            
            # Get inputs
            signals = batch['signal'].to(device)
            labels = batch['class_label'].to(device)
            
            # Forward pass
            logits = model(signals)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            # Update progress bar
            current_acc = 100 * correct / total
            dataloader_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader_pbar.iterable)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training curves saved as 'training_curves.png'")

if __name__ == "__main__":
    main()