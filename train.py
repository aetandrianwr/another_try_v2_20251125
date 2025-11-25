"""
Training script for next-location prediction
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import pickle
from datetime import datetime

from config import Config
from dataset import GeolifeDataset, collate_fn
from model import NextLocationPredictor
from metrics import calculate_correct_total_prediction, get_performance_dict


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Load datasets
        print("Loading datasets...")
        train_dataset = GeolifeDataset(os.path.join(config.data_dir, config.train_file))
        val_dataset = GeolifeDataset(os.path.join(config.data_dir, config.val_file))
        test_dataset = GeolifeDataset(os.path.join(config.data_dir, config.test_file))
        
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                       shuffle=True, collate_fn=collate_fn, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                                     shuffle=False, collate_fn=collate_fn, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                                      shuffle=False, collate_fn=collate_fn, num_workers=2)
        
        # Initialize model
        print("Initializing model...")
        self.model = NextLocationPredictor(config).to(self.device)
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Loss and optimizer
        self.criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay, betas=(0.9, 0.999))
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_stats = np.zeros(8, dtype=np.float32)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            # Move to device
            locations = batch['locations'].to(self.device)
            users = batch['users'].to(self.device)
            weekdays = batch['weekdays'].to(self.device)
            start_minutes = batch['start_minutes'].to(self.device)
            durations = batch['durations'].to(self.device)
            time_diffs = batch['time_diffs'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(locations, users, weekdays, start_minutes,
                              durations, time_diffs, mask)
            
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                stats, _, _ = calculate_correct_total_prediction(logits, targets)
                all_stats += stats
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update learning rate
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        perf = get_performance_dict({
            "correct@1": all_stats[0],
            "correct@3": all_stats[1],
            "correct@5": all_stats[2],
            "correct@10": all_stats[3],
            "rr": all_stats[4],
            "ndcg": all_stats[5],
            "f1": 0,
            "total": all_stats[6],
        })
        
        return avg_loss, perf
    
    @torch.no_grad()
    def evaluate(self, data_loader, phase="Val"):
        self.model.eval()
        total_loss = 0
        all_stats = np.zeros(8, dtype=np.float32)
        
        for batch in tqdm(data_loader, desc=f"{phase} Evaluation"):
            # Move to device
            locations = batch['locations'].to(self.device)
            users = batch['users'].to(self.device)
            weekdays = batch['weekdays'].to(self.device)
            start_minutes = batch['start_minutes'].to(self.device)
            durations = batch['durations'].to(self.device)
            time_diffs = batch['time_diffs'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            logits = self.model(locations, users, weekdays, start_minutes,
                              durations, time_diffs, mask)
            
            loss = self.criterion(logits, targets)
            total_loss += loss.item()
            
            # Calculate metrics
            stats, _, _ = calculate_correct_total_prediction(logits, targets)
            all_stats += stats
        
        avg_loss = total_loss / len(data_loader)
        perf = get_performance_dict({
            "correct@1": all_stats[0],
            "correct@3": all_stats[1],
            "correct@5": all_stats[2],
            "correct@10": all_stats[3],
            "rr": all_stats[4],
            "ndcg": all_stats[5],
            "f1": 0,
            "total": all_stats[6],
        })
        
        return avg_loss, perf
    
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config.__dict__,
        }
        
        path = os.path.join(self.config.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
    
    def train(self):
        print(f"\nStarting training on {self.device}...")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}\n")
        
        for epoch in range(1, self.config.max_epochs + 1):
            self.epoch = epoch
            
            # Train
            train_loss, train_perf = self.train_epoch()
            
            # Validate
            val_loss, val_perf = self.evaluate(self.val_loader, "Val")
            
            # Log results
            print(f"\nEpoch {epoch}/{self.config.max_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc@1: {train_perf['acc@1']:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc@1: {val_perf['acc@1']:.2f}% | "
                  f"Val Acc@5: {val_perf['acc@5']:.2f}% | Val MRR: {val_perf['mrr']:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            self.train_history.append({
                'epoch': epoch,
                'loss': train_loss,
                'acc@1': train_perf['acc@1'],
                'acc@5': train_perf['acc@5'],
            })
            
            self.val_history.append({
                'epoch': epoch,
                'loss': val_loss,
                'acc@1': val_perf['acc@1'],
                'acc@5': val_perf['acc@5'],
                'mrr': val_perf['mrr'],
            })
            
            # Save checkpoint
            is_best = val_perf['acc@1'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_perf['acc@1']
                self.patience_counter = 0
                print(f"New best validation Acc@1: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Load best model and evaluate on test set
        print("\nLoading best model for final evaluation...")
        best_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth')
        self.load_checkpoint(best_path)
        
        test_loss, test_perf = self.evaluate(self.test_loader, "Test")
        
        print("\n" + "="*80)
        print("FINAL TEST RESULTS")
        print("="*80)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc@1: {test_perf['acc@1']:.2f}%")
        print(f"Test Acc@5: {test_perf['acc@5']:.2f}%")
        print(f"Test Acc@10: {test_perf['acc@10']:.2f}%")
        print(f"Test MRR: {test_perf['mrr']:.2f}%")
        print(f"Test NDCG: {test_perf['ndcg']:.2f}%")
        print("="*80)
        
        # Save results
        results = {
            'test_performance': test_perf,
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        with open(os.path.join(self.config.log_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return test_perf


def main():
    config = Config()
    trainer = Trainer(config)
    test_perf = trainer.train()
    return test_perf


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
