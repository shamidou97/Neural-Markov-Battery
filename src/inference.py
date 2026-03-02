#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
MAX_CYCLES_CONST = 300.0  # Must match training normalization

# --- FIX 1: Robust Path Setup ---
project_root = os.getcwd()
while project_root != os.path.dirname(project_root):
    if 'src' in os.listdir(project_root) and 'models' in os.listdir(project_root):
        break
    project_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import NeuralMarkovNet

class BatteryMarkovInference:
    def __init__(self, model_path='models/best_model.pth', device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        self.model = NeuralMarkovNet(num_states=4).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        
        self.state_names = {0: "New", 1: "Good", 2: "Degraded", 3: "Failed"}
        print("✅ Inference model initialized")

    def _load_model(self, model_path):
        """FIX 2: Handle absolute paths and nested state_dicts"""
        if not os.path.isabs(model_path):
            full_path = os.path.join(project_root, model_path)
        else:
            full_path = model_path
            
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"❌ Model not found: {full_path}")
        
        checkpoint = torch.load(full_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict') or checkpoint
        self.model.load_state_dict(state_dict)
        print(f"📦 Loaded weights successfully from {full_path}")

    def predict_next_state(self, state_oh, age):
        """FIX 3: Ensure 2D tensor shapes for the forward pass"""
        state_tensor = state_oh.view(1, 4).to(self.device)
        # Age must be normalized (0-1) before this step
        age_tensor = torch.tensor([[float(age)]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(state_tensor, age_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return np.argmax(probs), probs

    def predict_from_npy(self, X_data, y_data=None):
        """
        FIX 4: Normalize raw age (80.16) and slice 5-column vector
        """
        correct = 0
        preds = []
        X_tensor = torch.from_numpy(X_data).float().to(self.device)
        
        print(f"🔮 Predicting {len(X_data)} transitions...")
        for i in tqdm(range(len(X_data)), desc="Running Inference"):
            # S0-S3 One-Hot (Columns 0, 1, 2, 3)
            state_slice = X_tensor[i, :4] 
            
            # Raw Age (Column 4, e.g., 80.16) -> Convert to 0.267
            raw_age = X_tensor[i, 4].item()
            normalized_age = min(raw_age / MAX_CYCLES_CONST, 1.0) 
            
            pred, probs = self.predict_next_state(state_slice, normalized_age)
            
            if y_data is not None:
                if pred == int(y_data[i]):
                    correct += 1
            
            preds.append({'pred': pred, 'conf': float(probs[pred])})
            
        accuracy = (correct / len(X_data)) * 100 if y_data is not None else 0
        return {'predictions': preds, 'accuracy': accuracy}

def main():
    model_file = os.path.join('models', 'best_model.pth')
    infer = BatteryMarkovInference(model_path=model_file)
    
    X_path = os.path.join(project_root, 'data', 'processed', 'X_test_scaled.npy')
    y_path = os.path.join(project_root, 'data', 'processed', 'y_test_scaled.npy')
    
    if os.path.exists(X_path):
        X_test = np.load(X_path)
        y_test = np.load(y_path) if os.path.exists(y_path) else None
        
        results = infer.predict_from_npy(X_test, y_test)
        
        if y_test is not None:
            # Accuracy should jump from 24% to 98%+
            print(f"\n📊 Balanced Test Set Accuracy: {results['accuracy']:.2f}%")
    else:
        print(f"❌ Test data not found at {X_path}")

if __name__ == "__main__":
    main()