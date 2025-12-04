"""
Multi-Horizon LSTM Model for ETF Price Prediction.

This module implements a shared-backbone LSTM with two prediction heads:
- head_5d: 5-day ahead return prediction
- head_1m: 1-month (21-day) ahead return prediction

The architecture follows modern deep learning practices:
- Shared LSTM backbone learns temporal patterns
- Separate dense heads specialize for each horizon
- Combined loss with configurable weights
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import MULTI_HORIZON_CONFIG, PROCESSED_DATA_DIR


# ==============================================================================
# Dataset
# ==============================================================================

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for multi-horizon time series prediction.
    
    Expects:
        X: numpy array of shape (n_samples, sequence_length, n_features)
        y: numpy array of shape (n_samples, n_horizons)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# Model Architecture
# ==============================================================================

class MultiHorizonLSTM(nn.Module):
    """
    LSTM model with shared backbone and multiple prediction heads.
    
    Architecture:
        Input -> LSTM (shared backbone) -> [head_5d, head_1m]
        
    Each head is a small MLP that produces a single scalar prediction.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: LSTM hidden state dimension (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate between LSTM layers (default: 0.2)
        n_horizons: Number of prediction horizons (default: 2)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_horizons: int = 2,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_horizons = n_horizons
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Layer normalization after LSTM
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Prediction heads: one for each horizon
        # head_5d (index 0): 5-day ahead prediction
        # head_1m (index 1): 1-month (21-day) ahead prediction
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )
            for _ in range(n_horizons)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor of shape (batch_size, n_horizons) with predictions for each horizon
        """
        # LSTM backbone
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply layer norm
        last_hidden = self.layer_norm(last_hidden)
        
        # Apply each prediction head
        predictions = []
        for head in self.heads:
            pred = head(last_hidden)  # (batch_size, 1)
            predictions.append(pred)
        
        # Concatenate predictions: (batch_size, n_horizons)
        output = torch.cat(predictions, dim=1)
        
        return output
    
    @classmethod
    def from_config(cls, input_size: int, config: dict = None):
        """Create model from config dictionary."""
        if config is None:
            config = MULTI_HORIZON_CONFIG["model"]
        
        n_horizons = len(MULTI_HORIZON_CONFIG.get("horizons", {"5d": 5, "1m": 21}))
        
        return cls(
            input_size=input_size,
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
            n_horizons=n_horizons,
        )


# ==============================================================================
# Trainer
# ==============================================================================

class MultiHorizonTrainer:
    """
    Trainer class for multi-horizon LSTM model.
    
    Handles training loop, validation, early stopping, and checkpointing.
    
    Args:
        model: MultiHorizonLSTM model
        device: 'cuda' or 'cpu'
        loss_weights: Dict or list of weights for each horizon loss
        learning_rate: Optimizer learning rate
        checkpoint_dir: Directory for saving model checkpoints
    """
    
    def __init__(
        self,
        model: MultiHorizonLSTM,
        device: str = None,
        loss_weights: Dict[str, float] = None,
        learning_rate: float = 0.001,
        checkpoint_dir: Path = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Loss weights for multi-horizon
        if loss_weights is None:
            loss_weights = MULTI_HORIZON_CONFIG.get("loss_weights", {"5d": 1.0, "1m": 1.0})
        self.loss_weights = list(loss_weights.values()) if isinstance(loss_weights, dict) else loss_weights
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Scheduler for learning rate decay
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss function
        self.criterion = nn.MSELoss(reduction='none')
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir or Path(PROCESSED_DATA_DIR) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_loss_5d': [],
            'train_loss_1m': [],
            'val_loss_5d': [],
            'val_loss_1m': [],
        }
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted multi-horizon loss.
        
        Args:
            predictions: (batch_size, n_horizons)
            targets: (batch_size, n_horizons)
            
        Returns:
            Tuple of (total_loss, dict of per-horizon losses)
        """
        # Per-horizon MSE
        horizon_losses = self.criterion(predictions, targets).mean(dim=0)  # (n_horizons,)
        
        # Weighted sum
        weights = torch.tensor(self.loss_weights, device=self.device)
        total_loss = (horizon_losses * weights).sum()
        
        # Return breakdown for logging
        loss_dict = {
            'loss_5d': horizon_losses[0].item(),
            'loss_1m': horizon_losses[1].item() if len(horizon_losses) > 1 else 0,
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        loss_breakdown = {'loss_5d': 0, 'loss_1m': 0}
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(X_batch)
            loss, losses = self.compute_loss(predictions, y_batch)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            for k in loss_breakdown:
                loss_breakdown[k] += losses.get(k, 0)
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_breakdown = {k: v / n_batches for k, v in loss_breakdown.items()}
        
        return avg_loss, avg_breakdown
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        loss_breakdown = {'loss_5d': 0, 'loss_1m': 0}
        n_batches = 0
        
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            predictions = self.model(X_batch)
            loss, losses = self.compute_loss(predictions, y_batch)
            
            total_loss += loss.item()
            for k in loss_breakdown:
                loss_breakdown[k] += losses.get(k, 0)
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_breakdown = {k: v / max(n_batches, 1) for k, v in loss_breakdown.items()}
        
        return avg_loss, avg_breakdown
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"Training Multi-Horizon LSTM on {self.device}")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_breakdown = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_breakdown = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_loss_5d'].append(train_breakdown['loss_5d'])
            self.history['train_loss_1m'].append(train_breakdown['loss_1m'])
            self.history['val_loss_5d'].append(val_breakdown['loss_5d'])
            self.history['val_loss_1m'].append(val_breakdown['loss_1m'])
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
            
            # Logging
            if verbose and (epoch + 1) % 5 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {train_loss:.6f} (5d: {train_breakdown['loss_5d']:.6f}, 1m: {train_breakdown['loss_1m']:.6f}) | "
                      f"Val: {val_loss:.6f} (5d: {val_breakdown['loss_5d']:.6f}, 1m: {val_breakdown['loss_1m']:.6f}) | "
                      f"LR: {lr:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
        
        # Load best model
        self.load_checkpoint("best_model.pt")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'history' in checkpoint:
                self.history = checkpoint['history']
    
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: numpy array of shape (n_samples, sequence_length, n_features)
            
        Returns:
            numpy array of shape (n_samples, n_horizons)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = self.model(X_tensor)
        return predictions.cpu().numpy()


# ==============================================================================
# Utility Functions
# ==============================================================================

def create_data_loaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        X_train, X_val: Feature arrays
        y_train, y_val: Target arrays
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader


def get_default_feature_columns() -> List[str]:
    """
    Get default feature columns for multi-horizon model.
    
    Returns list of feature column names based on typical dataset structure.
    """
    # Technical indicators
    tech_features = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'ema_12', 'rsi_14', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower',
    ]
    
    # Sentiment features
    sentiment_features = [
        'sentiment_score', 'sentiment_std', 'sentiment_min', 'sentiment_max',
        'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
        'news_count', 'sentiment_ma_5', 'sentiment_ma_10',
    ]
    
    # Lagged features
    lag_features = [
        'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10',
        'volume_lag_1', 'volume_lag_3', 'volume_lag_5',
        'sentiment_lag_1', 'sentiment_lag_3',
        'rsi_lag_1', 'rsi_lag_3',
    ]
    
    return tech_features + sentiment_features + lag_features


# ==============================================================================
# Test / Demo
# ==============================================================================

if __name__ == "__main__":
    print("Multi-Horizon LSTM Model Module")
    print("=" * 50)
    
    # Test model creation
    input_size = 30  # number of features
    model = MultiHorizonLSTM.from_config(input_size=input_size)
    print(f"\nModel created: {model}")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Num layers: {model.num_layers}")
    print(f"  Num horizons: {model.n_horizons}")
    
    # Test forward pass
    batch_size = 16
    seq_length = 20
    x = torch.randn(batch_size, seq_length, input_size)
    out = model(x)
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")  # Should be (16, 2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
