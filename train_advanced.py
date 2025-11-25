"""
Advanced training script with modern techniques:
- Focal Loss for class imbalance
- Mixup augmentation
- Mixed precision training
- Cosine annealing with warmup
- EMA (Exponential Moving Average)
- Gradient accumulation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
from collections import Counter

from config_advanced import AdvancedConfig
from dataset import GeolifeDataset, collate_fn
from model_advanced import AdvancedNextLocationPredictor
from metrics import calculate_correct_total_prediction, get_performance_dict


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=1187):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


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


def mixup_data(x_dict, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = y.size(0)
    index = torch.randperm(batch_size).to(y.device)
    
    mixed_dict = {}
    for key, value in x_dict.items():
        if key in ['locations', 'users', 'weekdays', 'time_diffs']:
            # For discrete features, use one or the other
            mixed_dict[key] = value if lam > 0.5 else value[index]
        else:
            # For continuous features, interpolate
            mixed_dict[key] = lam * value + (1 - lam) * value[index]
    
    y_a, y_b = y, y[index]
    return mixed_dict, y_a, y_b, lam


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class AdvancedTrainer:
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
        print("Initializing advanced model...")
        self.model = AdvancedNextLocationPredictor(config).to(self.device)
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Initialize location bias from training data
        self._init_location_bias(train_dataset)
        
        # Loss function
        if config.use_focal_loss:
            self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma,
                                      num_classes=config.num_locations)
        else:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
        
        # Optimizer with different learning rates for different parts
        param_groups = [
            {'params': self.model.loc_embedding.parameters(), 'lr': config.learning_rate * 0.5},
            {'params': self.model.user_pref_embed.parameters(), 'lr': config.learning_rate * 0.1},
            {'params': self.model.loc_pref_embed.parameters(), 'lr': config.learning_rate * 0.1},
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'loc_embedding' not in n and 'user_pref' not in n and 'loc_pref' not in n], 
             'lr': config.learning_rate}
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay,
                                     betas=(0.9, 0.999), eps=1e-8)
        
        # Learning rate scheduler
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=config.warmup_epochs
        )
        self.main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs - config.warmup_epochs,
            eta_min=config.min_lr
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # EMA
        self.ema = EMA(self.model, decay=0.999)
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []
    
    def _init_location_bias(self, dataset):
        """Initialize output bias based on location frequency"""
        location_counts = Counter()
        for sample in dataset.data:
            location_counts[sample['Y']] += 1
        
        # Create frequency array
        freq_array = np.zeros(self.config.num_locations)
        for loc_id, count in location_counts.items():
            freq_array[loc_id] = count
        
        self.model.set_location_bias(freq_array)
        print(f"Initialized location bias from {len(location_counts)} unique locations")
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_stats = np.zeros(8, dtype=np.float32)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            # Move to device
            batch_dict = {
                'locations': batch['locations'].to(self.device),
                'users': batch['users'].to(self.device),
                'weekdays': batch['weekdays'].to(self.device),
                'start_minutes': batch['start_minutes'].to(self.device),
                'durations': batch['durations'].to(self.device),
                'time_diffs': batch['time_diffs'].to(self.device),
                'mask': batch['mask'].to(self.device)
            }
            targets = batch['targets'].to(self.device)
            
            # Mixup augmentation
            if self.config.use_mixup and np.random.rand() < 0.5:
                batch_dict, targets_a, targets_b, lam = mixup_data(batch_dict, targets, 
                                                                    self.config.mixup_alpha)
                use_mixup = True
            else:
                use_mixup = False
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast():
                    logits = self.model(**batch_dict)
                    if use_mixup:
                        loss = lam * self.criterion(logits, targets_a) + \
                               (1 - lam) * self.criterion(logits, targets_b)
                    else:
                        loss = self.criterion(logits, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(**batch_dict)
                if use_mixup:
                    loss = lam * self.criterion(logits, targets_a) + \
                           (1 - lam) * self.criterion(logits, targets_b)
                else:
                    loss = self.criterion(logits, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # Update EMA
            self.ema.update()
            
            # Calculate metrics (use original targets)
            with torch.no_grad():
                stats, _, _ = calculate_correct_total_prediction(logits, targets if not use_mixup else targets_a)
                all_stats += stats
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update learning rate
        if self.epoch < self.config.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        perf = get_performance_dict({
            "correct@1": all_stats[0],
            "correct@3": all_stats[1],
            "correct@5": all_stats[2],
            "correct@10": all_stats[3],
            "rr": all_stats[5],
            "ndcg": all_stats[6],
            "f1": all_stats[4],
            "total": all_stats[7],
        })
        
        return avg_loss, perf
    
    @torch.no_grad()
    def evaluate(self, data_loader, phase="Val", use_ema=False):
        # Apply EMA if requested
        if use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0
        all_stats = np.zeros(8, dtype=np.float32)
        
        for batch in tqdm(data_loader, desc=f"{phase} Evaluation"):
            batch_dict = {
                'locations': batch['locations'].to(self.device),
                'users': batch['users'].to(self.device),
                'weekdays': batch['weekdays'].to(self.device),
                'start_minutes': batch['start_minutes'].to(self.device),
                'durations': batch['durations'].to(self.device),
                'time_diffs': batch['time_diffs'].to(self.device),
                'mask': batch['mask'].to(self.device)
            }
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            logits = self.model(**batch_dict)
            loss = self.criterion(logits, targets)
            total_loss += loss.item()
            
            # Calculate metrics
            stats, _, _ = calculate_correct_total_prediction(logits, targets)
            all_stats += stats
        
        # Restore original weights if using EMA
        if use_ema:
            self.ema.restore()
        
        avg_loss = total_loss / len(data_loader)
        perf = get_performance_dict({
            "correct@1": all_stats[0],
            "correct@3": all_stats[1],
            "correct@5": all_stats[2],
            "correct@10": all_stats[3],
            "rr": all_stats[5],
            "ndcg": all_stats[6],
            "f1": all_stats[4],
            "total": all_stats[7],
        })
        
        return avg_loss, perf
    
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config.__dict__,
        }
        
        path = os.path.join(self.config.checkpoint_dir, 'last_checkpoint_advanced.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint_advanced.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
    
    def train(self):
        print(f"\nStarting advanced training on {self.device}...")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}\n")
        
        for epoch in range(1, self.config.max_epochs + 1):
            self.epoch = epoch
            
            # Train
            train_loss, train_perf = self.train_epoch()
            
            # Validate (with EMA)
            val_loss, val_perf = self.evaluate(self.val_loader, "Val", use_ema=True)
            
            # Log results
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{self.config.max_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc@1: {train_perf['acc@1']:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc@1: {val_perf['acc@1']:.2f}% | "
                  f"Val Acc@5: {val_perf['acc@5']:.2f}% | Val MRR: {val_perf['mrr']:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
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
                print(f"✨ New best validation Acc@1: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Load best model and evaluate on test set
        print("\nLoading best model for final evaluation...")
        best_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint_advanced.pth')
        self.load_checkpoint(best_path)
        
        test_loss, test_perf = self.evaluate(self.test_loader, "Test", use_ema=True)
        
        print("\n" + "="*80)
        print("FINAL TEST RESULTS (Advanced Model)")
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
            'test_performance': {k: float(v) for k, v in test_perf.items()},
            'best_val_acc': float(self.best_val_acc),
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        with open(os.path.join(self.config.log_dir, 'results_advanced.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return test_perf


def main():
    import torch.nn.functional as F
    config = AdvancedConfig()
    trainer = AdvancedTrainer(config)
    test_perf = trainer.train()
    return test_perf


if __name__ == "__main__":
    main()
