import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path

from data.dataset import get_data_loaders
from models.model import CovidClassifier

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
        
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def main():
    # Initialize wandb
    wandb.init(project="covid-xray-classification")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Data loaders
    data_dir = "data/raw/COVID-19_Radiography_Dataset"
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )
    
    print(f"Classes: {class_to_idx}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    model = CovidClassifier(
        num_classes=len(class_to_idx),
        model_name='efficientnet_b0'
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler()
    
    # Training loop
    num_epochs = 30
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / 'best_model.pth')
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')

if __name__ == '__main__':
    main()
