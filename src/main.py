import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.data.advanced_dataset import AdvancedCovidDataset
from src.models.advanced_model import AdvancedCovidClassifier
from src.training.trainer import Trainer
from src.utils.data_utils import get_data_splits

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Set random seeds
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    
    # Get data splits
    train_df, val_df, test_df = get_data_splits(cfg.data.data_dir)
    
    # Create datasets
    train_dataset = AdvancedCovidDataset(train_df, cfg.data, mode="train")
    val_dataset = AdvancedCovidDataset(val_df, cfg.data, mode="valid")
    test_dataset = AdvancedCovidDataset(test_df, cfg.data, mode="test")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )
    
    # Create model
    model = AdvancedCovidClassifier(cfg.model)
    
    # Create trainer
    trainer = Trainer(model, cfg)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_metrics = trainer.validate(test_loader)
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    main()
