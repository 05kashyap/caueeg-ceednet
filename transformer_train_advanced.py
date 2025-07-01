import torch
import torch.nn as nn
import torch.optim as optim
from datasets.caueeg_script import build_dataset_for_train
from models.advanced_eeg_transformer import create_advanced_eeg_transformer
from train.train_script import train_script
from models.utils import count_parameters
import pprint


def main():
    """Train the advanced EEG transformer using the existing infrastructure."""
    
    # Configuration for training
    config = {
        # Dataset configuration
        "dataset_path": "local/datasets/caueeg-dataset",
        "task": "dementia",  # or "abnormal"
        "load_event": False,
        "file_format": "feather",  # or "feather" for speed
        
        # Model configuration
        "model": "AdvancedEEGTransformer",
        "model_size": "base",  # tiny, small, base, large, huge - start smaller for debugging large is overkill
        "use_age": "no",  # "fc", "conv", or "no" - DISABLE AGE
        "fc_stages": 3,
        
        # Training configuration
        "minibatch": 32,  # Reduce batch size for debugging
        "crop_multiple": 32,  # Number of crops per sample - reduce for debugging
        "test_crop_multiple": 32,
        "seq_length": 200,  # 1 seconds at 200Hz
        "total_samples": 20000,  # Increase total samples now that it works
        "warmup_ratio": 0.1,
        "warmup_min": 100,
        "num_history": 50,  # Number of evaluation points
        
        # Data preprocessing
        "EKG": "X",  # Use EKG channel: "O" or "X"
        "photic": "X",  # Use photic channel: "O" or "X" 
        "input_norm": "dataset",  # "dataset", "datapoint", or "no"
        "latency": 2000,  # 10 seconds latency for random crop
        
        # Augmentation
        "awgn": 0.02,  # Additive white Gaussian noise std
        "mgn": 0.01,   # Multiplicative Gaussian noise std
        "awgn_age": 0.01,  # Age noise std
        
        # Optimizer configuration
        "base_lr": 1e-4,  # Will be adjusted by lr search if enabled
        "weight_decay": 1e-4,
        "lr_scheduler_type": "cosine_decay_with_warmup_half",  # Use valid scheduler type
        "search_lr": False,  # Disable for now to avoid the error
        "search_multiplier": 1.0,
        
        # Loss function
        "criterion": "cross-entropy",  # "cross-entropy", "multi-bce", or "svm"
        
        # Mixed precision training
        "mixed_precision": True,
        "clip_grad_norm": 1.0,
        
        # Regularization
        "mixup": 0.2,  # Mixup alpha, 0 to disable
        
        # Device configuration
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "ddp": False,  # Distributed training
        
        # Logging and saving
        "use_wandb": False,  # Set to True to use Weights & Biases
        "project": "advanced-eeg-transformer",
        "save_model": True,
        "draw_result": True,
        "watch_model": False,
        
        # Evaluation
        "check_accuracy_repeat": 10,
        
        # Working directory
        "cwd": "/home/kashyap/Documents/cbr@iisc/caueeg-ceednet",
    }
    
    print("Building dataset...")
    train_loader, val_loader, test_loader, multicrop_test_loader = build_dataset_for_train(
        config, verbose=True
    )
    
    # Create the advanced transformer model
    print(f"\nCreating {config['model_size']} Advanced EEG Transformer...")
    model = create_advanced_eeg_transformer(
        seq_length=config["seq_length"],
        in_channels=config["in_channels"],
        out_dims=config["out_dims"],
        model_size=config["model_size"],
        use_age=config["use_age"],
        fc_stages=config["fc_stages"],
        dropout=0.1,
        activation="gelu"
    )
    
    # Move model to device
    model = model.to(config["device"])
    
    # Print model information
    num_params = count_parameters(model)
    print(f"Model created with {num_params:,} parameters")
    
    # Print model architecture summary
    print("\nModel Architecture:")
    print(f"  - Sequence length: {config['seq_length']}")
    print(f"  - Input channels: {config['in_channels']}")
    print(f"  - Output classes: {config['out_dims']}")
    print(f"  - Model size: {config['model_size']}")
    print(f"  - Embedding dimension: {model.embed_dim}")
    print(f"  - Number of heads: {model.transformer_layers[0].num_heads}")
    print(f"  - Number of layers: {len(model.transformer_layers)}")
    print(f"  - Total patches: {model.total_patches}")
    print(f"  - Use age: {config['use_age']}")
    print(f"  - FC stages: {config['fc_stages']}")
    
    # Print final configuration
    print(f"\n{'*'*50} Final Configuration {'*'*50}")
    pprint.pprint(config, width=120)
    print(f"{'*'*116}")
    
    # Start training using existing infrastructure
    print(f"\nStarting training...")
    train_script(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        multicrop_test_loader=multicrop_test_loader,
        preprocess_train=config["preprocess_train"],
        preprocess_test=config["preprocess_test"]
    )


if __name__ == "__main__":
    main()
