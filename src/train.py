#!/usr/bin/env python
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.model import NeuralMarkovNet

def load_preprocessed_data(data_dir='data/processed'):
    """Directly load preprocessed numpy files using the 'scaled' suffix."""
    print(f"📂 Loading preprocessed data from {data_dir}...")
    
    required_files = ['X_train_scaled.npy', 'y_train_scaled.npy', 'X_test_scaled.npy', 'y_test_scaled.npy']
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            print(f"❌ Missing {os.path.join(data_dir, file)}")
            return None, None, None, None
    
    X_train = np.load(os.path.join(data_dir, 'X_train_scaled.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_scaled.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_scaled.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_scaled.npy'))
    
    print(f"✅ Data loaded successfully! Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = True if torch.cuda.is_available() else False
    print(f"\n--- ACTIVE DEVICE: {device} ---")

    os.makedirs('models', exist_ok=True)

    # Dataset Preparation
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=use_pin)
    
    model = NeuralMarkovNet(num_states=4).to(device)
    
    # Loss Weights for State 2 (SOH < 0.88 cliff)
    weights = torch.tensor([1.0, 1.2, 3.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'loss_gap': []}

    print(f"\nStarting Training for {epochs} epochs...\n" + "="*90)
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss, train_correct, train_total = 0, 0, 0
        
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            
            # FIX: Mapping 4 columns to 5 model inputs
            state_oh = features[:, :4]  
            norm_age = features[:, 3:4] 
            
            outputs = model(state_oh, norm_age) 
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for features, target in val_loader:
                features, target = features.to(device), target.to(device)
                outputs = model(features[:, :4], features[:, 3:4])
                loss = criterion(outputs, target)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        loss_gap = abs(avg_train_loss - avg_val_loss)

        # Store History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['loss_gap'].append(loss_gap)
        
        # Display Every 10 Epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"Epoch {(epoch+1):03d} | "
                       f"T-Loss: {avg_train_loss:.4f} | V-Loss: {avg_val_loss:.4f} | "
                       f"Gap: {loss_gap:.4f} | T-Acc: {train_acc:.2f}% | V-Acc: {val_acc:.2f}%")

    # --- SAVE RESULTS & PLOTS ---
    print("\n" + "="*90)
    print("Training Complete. Saving Results...")
    
    # Save Model Checkpoint
    model_path = "models/best_model.pth"
    torch.save({'state_dict': model.state_dict(), 'history': history}, model_path)
    print(f"✅ Model Results Saved To: {os.path.abspath(model_path)}")
    
    # Plotting Learning Curves
    plot_path = 'models/training_curves.png'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Convergence'); axes[0].legend(); axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_title('Accuracy Trend'); axes[1].legend(); axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"✅ Training Plots Saved To: {os.path.abspath(plot_path)}")
    
    return history

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_preprocessed_data()
    if X_train is not None:
        train_model(X_train, y_train, X_test, y_test, epochs=100)