import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device, use_amp):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler(enabled=use_amp)

    def train_one_epoch(self, loader, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
        return epoch_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        total_loss = 0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                all_labels.extend(y.float().cpu().numpy())
                all_preds.extend(torch.sigmoid(logits).float().cpu().numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        return total_loss / len(loader), auc