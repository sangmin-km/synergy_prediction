"""
Drug Synergy Model Evaluation
10-fold cross-validation evaluation for SynG2Net model.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

# Import model
from models.net import DrugDrugInteractionNet

warnings.filterwarnings('ignore')

# CUDA settings
torch.backends.cuda.matmul.allow_tf32 = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'


class DrugDataset(torch.utils.data.Dataset):
    """Custom dataset class for drug pair data."""
    
    def __init__(self, data_list):
        super(DrugDataset, self).__init__()
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


def train(model, device, loader, optimizer, criterion):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_a, batch_b in loader:
        batch_a = batch_a.to(device)
        batch_b = batch_b.to(device)
        
        optimizer.zero_grad()
        output = model(batch_a, batch_b).view(-1)
        loss = criterion(output, batch_a.y.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, device, loader):
    """Evaluate model on given data loader."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_a, batch_b in loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            
            output = model(batch_a, batch_b).view(-1)
            predictions.extend(output.cpu().numpy())
            actuals.extend(batch_a.y.view(-1).cpu().numpy())
    
    return np.array(predictions), np.array(actuals)


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    
    return mse, rmse, mae, r2, pearson


if __name__ == '__main__':
    # Hyperparameters
    config = {
        'num_epochs': 200,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'k_folds': 10,
        'hidden_dim': 128,
        'gt_layers': 3,
        'gt_heads': 4,
        'dropout': 0.4,
        'random_seed': 42
    }
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Load datasets
    print('Loading datasets...')
    drug_a_dataset = torch.load('drug_datasets/drug_a_dataset.pt')
    drug_b_dataset = torch.load('drug_datasets/drug_b_dataset.pt')
    all_data = list(zip(drug_a_dataset, drug_b_dataset))
    
    print(f'Total samples: {len(all_data)}')
    
    # K-Fold Cross Validation
    print(f'\nStarting {config["k_folds"]}-fold cross validation')
    
    kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=config['random_seed'])
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data), 1):
        print(f"Fold {fold}/{config['k_folds']}")
        
        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_data, 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        test_loader = DataLoader(
            test_data, 
            batch_size=config['batch_size']
        )
        
        # Initialize model
        model = DrugDrugInteractionNet(
            hidden_dim=config['hidden_dim'],
            gt_layers=config['gt_layers'],
            gt_heads=config['gt_heads'],
            dropout=config['dropout']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()
        
        best_test_loss = float('inf')
        
        # Training loop
        pbar = tqdm(range(config['num_epochs']), desc=f"Training Fold {fold}")
        for epoch in pbar:
            # Train
            train_loss = train(model, device, train_loader, optimizer, criterion)
            
            # Evaluate
            test_predictions, test_actuals = evaluate(model, device, test_loader)
            test_loss = mean_squared_error(test_actuals, test_predictions)
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pt')
            
            pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}', 
                'Test Loss': f'{test_loss:.4f}',
                'Best': f'{best_test_loss:.4f}'
            })
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pt'))
        test_predictions, test_actuals = evaluate(model, device, test_loader)
        metrics = calculate_metrics(test_actuals, test_predictions)
        fold_metrics.append(metrics)
        
        # Print fold results
        print(f"\nFold {fold} Results:")
        print(f"  MSE: {metrics[0]:.4f}")
        print(f"  RMSE: {metrics[1]:.4f}")
        print(f"  MAE: {metrics[2]:.4f}")
        print(f"  R2: {metrics[3]:.4f}")
        print(f"  Pearson: {metrics[4]:.4f}")
    
    # Print overall results
    print("\nOverall Results (Mean ± Std)")
    fold_metrics = np.array(fold_metrics)
    mean_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)
    
    print(f"MSE     : {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"RMSE    : {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"MAE     : {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"R2      : {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
    print(f"Pearson : {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")
