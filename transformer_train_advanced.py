import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from datasets.caueeg_script import build_dataset_for_train
from models.advanced_eeg_transformer import create_advanced_eeg_transformer
from train.train_script import train_script
from models.utils import count_parameters
import pprint


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train Advanced EEG Transformer")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, default="local/datasets/caueeg-dataset",
                       help="Path to the dataset")
    parser.add_argument("--task", type=str, default="dementia", choices=["dementia", "abnormal"],
                       help="Task to train on")
    parser.add_argument("--load_event", action="store_true", default=False,
                       help="Load event data")
    parser.add_argument("--file_format", type=str, default="feather", choices=["feather", "csv"],
                       help="File format for dataset")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="AdvancedEEGTransformer",
                       help="Model type")
    parser.add_argument("--model_size", type=str, default="base", 
                       choices=["tiny", "small", "base", "large", "huge"],
                       help="Model size")
    parser.add_argument("--use_age", type=str, default="no", choices=["fc", "conv", "no"],
                       help="How to use age information")
    parser.add_argument("--fc_stages", type=int, default=3,
                       help="Number of FC stages")
    
    # Training configuration
    parser.add_argument("--minibatch", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--crop_multiple", type=int, default=32,
                       help="Number of crops per sample")
    parser.add_argument("--test_crop_multiple", type=int, default=32,
                       help="Number of crops per sample for testing")
    parser.add_argument("--seq_length", type=int, default=200,
                       help="Sequence length")
    parser.add_argument("--total_samples", type=int, default=20000,
                       help="Total number of samples")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--warmup_min", type=int, default=100,
                       help="Minimum warmup steps")
    parser.add_argument("--num_history", type=int, default=50,
                       help="Number of evaluation points")
    
    # Data preprocessing
    parser.add_argument("--EKG", type=str, default="X", choices=["O", "X"],
                       help="Use EKG channel")
    parser.add_argument("--photic", type=str, default="X", choices=["O", "X"],
                       help="Use photic channel")
    parser.add_argument("--input_norm", type=str, default="dataset", 
                       choices=["dataset", "datapoint", "no"],
                       help="Input normalization type")
    parser.add_argument("--latency", type=int, default=2000,
                       help="Latency for random crop")
    
    # Augmentation
    parser.add_argument("--awgn", type=float, default=0.02,
                       help="Additive white Gaussian noise std")
    parser.add_argument("--mgn", type=float, default=0.01,
                       help="Multiplicative Gaussian noise std")
    parser.add_argument("--awgn_age", type=float, default=0.01,
                       help="Age noise std")
    
    # Optimizer configuration
    parser.add_argument("--base_lr", type=float, default=1e-4,
                       help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_decay_with_warmup_half",
                       help="Learning rate scheduler type")
    parser.add_argument("--search_lr", action="store_true", default=False,
                       help="Enable learning rate search")
    parser.add_argument("--search_multiplier", type=float, default=1.0,
                       help="Learning rate search multiplier")
    
    # Loss function
    parser.add_argument("--criterion", type=str, default="cross-entropy",
                       choices=["cross-entropy", "multi-bce", "svm"],
                       help="Loss criterion")
    
    # Mixed precision training
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    
    # Regularization
    parser.add_argument("--mixup", type=float, default=0.2,
                       help="Mixup alpha (0 to disable)")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--ddp", action="store_true", default=False,
                       help="Use distributed training")
    
    # Logging and saving
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Use Weights & Biases logging")
    parser.add_argument("--project", type=str, default="advanced-eeg-transformer",
                       help="WandB project name")
    parser.add_argument("--save_model", action="store_true", default=True,
                       help="Save trained model")
    parser.add_argument("--draw_result", action="store_true", default=True,
                       help="Draw training results")
    parser.add_argument("--watch_model", action="store_true", default=False,
                       help="Watch model gradients in WandB")
    
    # Evaluation
    parser.add_argument("--check_accuracy_repeat", type=int, default=10,
                       help="Number of accuracy check repeats")
    
    # Working directory
    parser.add_argument("--cwd", type=str, default="/home/kashyap/Documents/cbr@iisc/caueeg-ceednet",
                       help="Working directory")
    
    return parser.parse_args()


def main():
    """Train the advanced EEG transformer using the existing infrastructure."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Convert args to config dictionary
    config = vars(args)
    
    # Handle device configuration
    if config["device"] == "auto":
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        config["device"] = torch.device(config["device"])
    
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
