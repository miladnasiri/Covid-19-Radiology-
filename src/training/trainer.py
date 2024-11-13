import os
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import MetricTracker
from src.utils.losses import get_loss_fn
from src.utils.optimizers import SAM
from src.utils.schedulers import get_scheduler

class Trainer:
    """Advanced training class with modern techniques."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup model
        self.model = model.to(self.device)
        if config.training.use_ema:
            self.ema_model = ModelEMA(model, config.training.ema_decay)
            
        # Setup training components
        self.criterion = get_loss_fn(config)
        self.optimizer = self._get_optimizer()
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Setup metric tracking
        self.metric_tracker = MetricTracker()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize W&B
        self.init_wandb()
        
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on config."""
        if self.config.training.use_sam:
            return SAM(
                self.model.parameters(),
                optim.AdamW,
                lr=self.config.training.initial_lr,
                weight_decay=self.config.training.weight_decay
            )
        
        if self.config.training.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.initial_lr,
                weight_decay=self.config.training.weight_decay
            )
        # Add more optimizer options here
        
    def train_epoch(self, train_loader: DataLoader):
        """Train one epoch with advanced techniques."""
        self.model.train()
        self.metric_tracker.reset()
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Mixed precision training
            with autocast(enabled=self.config.training.use_amp):
                if self.config.training.use_sam:
                    # First forward-backward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.optimizer.first_step(zero_grad=True)
                    
                    # Second forward-backward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.optimizer.second_step(zero_grad=True)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    self.optimizer.zero_grad()
                    if self.config.training.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
            
            # Update EMA model if enabled
            if self.config.training.use_ema:
                self.ema_model.update(self.model)
            
            # Update metrics
            self.metric_tracker.update("train_loss", loss.item())
            self.metric_tracker.update_metrics(outputs, labels)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{self.metric_tracker.get_metric('accuracy'):.4f}"
            })
            
            # Log to W&B
            if batch_idx % self.config.log_interval == 0:
                self.log_metrics("train", batch_idx)
        
        return self.metric_tracker.get_metrics()
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        """Validate the model."""
        model = self.ema_model.ema if self.config.training.use_ema else self.model
        model.eval()
        self.metric_tracker.reset()
        
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = model(images)
            loss = self.criterion(outputs, labels)
            
            # Update metrics
            self.metric_tracker.update("val_loss", loss.item())
            self.metric_tracker.update_metrics(outputs, labels)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{self.metric_tracker.get_metric('accuracy'):.4f}"
            })
        
        self.log_metrics("val")
        return self.metric_tracker.get_metrics()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        best_metric = float('inf')
        for epoch in range(self.config.training.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.config.training.epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(f"Training metrics: {train_metrics}")
            
            # Validation
            val_metrics = self.validate(val_loader)
            self.logger.info(f"Validation metrics: {val_metrics}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["val_loss"])
            
            # Save best model
            if val_metrics["val_loss"] < best_metric:
                best_metric = val_metrics["val_loss"]
                self.save_checkpoint(epoch, val_metrics["val_loss"])
            
            # Early stopping
            if self.early_stopping(val_metrics["val_loss"]):
                self.logger.info("Early stopping triggered")
                break
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config
        }
        
        if self.config.training.use_ema:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()
            
        save_path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.config.output_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
    
    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.experiment_name,
            config=self.config,
            name=f"{self.config.model.model_name}_{wandb.util.generate_id()}"
        )
    
    def log_metrics(self, prefix: str, step: Optional[int] = None):
        """Log metrics to W&B."""
        metrics = self.metric_tracker.get_metrics()
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
    
    def early_stopping(self, loss: float) -> bool:
        """Check if training should be stopped early."""
        if not hasattr(self, "early_stopping_counter"):
            self.early_stopping_counter = 0
            self.best_loss = float('inf')
            
        if loss < self.best_loss:
            self.best_loss = loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            
        return self.early_stopping_counter >= self.config.training.early_stopping_patience
