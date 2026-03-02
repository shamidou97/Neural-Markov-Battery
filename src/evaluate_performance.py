import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. GLOBAL CONFIGURATION ---
project_root = os.getcwd() 
while project_root != os.path.dirname(project_root):
    if 'src' in os.listdir(project_root):
        break
    project_root = os.path.dirname(project_root)

# Explicitly define and create plots directory at the start
PLOTS_DIR = os.path.join(project_root, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import NeuralMarkovNet

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define absolute paths based on project_root
    model_path = os.path.join(project_root, "models", "best_model.pth")
    data_path = os.path.join(project_root, "data", "processed", "X_test_scaled.npy")
    label_path = os.path.join(project_root, "data", "processed", "y_test_scaled.npy")

    # 2. LOAD MODEL
    if not os.path.exists(model_path):
        print(f"❌ MODEL NOT FOUND AT: {model_path}")
        return

    model = NeuralMarkovNet(num_states=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 3. LOAD DATA
    X_test = np.load(data_path)
    y_test = np.load(label_path)

    # 4. INFERENCE & MATRIX
    with torch.no_grad():
        features = torch.tensor(X_test).float().to(device)
        # Slicing: Reverted to your working [:, :4] and [:, 3:4] split
        outputs = model(features[:, :4], features[:, 3:4])
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()

    # --- HEATMAP ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.5) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['S0', 'S1', 'S2', 'S3'],
                yticklabels=['S0', 'S1', 'S2', 'S3'])
    plt.title('Confusion Matrix: Battery States', fontsize=28, fontweight='bold')
    plt.ylabel('True State', fontsize=22, fontweight='bold')
    plt.xlabel('Predicted State', fontsize=22, fontweight='bold')

    heatmap_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved: {heatmap_path}")
    plt.show()

    # --- 5. THE PERCENTAGE TABLE WITH WATERMARK ---
    report_dict = classification_report(y_test, y_pred, 
                                        target_names=['S0', 'S1', 'S2', 'S3'], 
                                        output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Convert metrics to percentages
    for col in ['precision', 'recall', 'f1-score']:
        report_df[col] = report_df[col].apply(lambda x: f"{x*100:.2f}%")
    report_df['support'] = report_df['support'].apply(lambda x: str(int(x)))

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    tbl = ax.table(cellText=report_df.values, 
                   colLabels=report_df.columns, 
                   rowLabels=report_df.index, 
                   loc='center', cellLoc='center')
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(16)
    tbl.scale(1.2, 3.2) 

    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')

    # Portfolio Watermark for MS Physics & EE [cite: 2026-02-28]
    watermark_text = (
        "Model: NeuralMarkovNet\n"
        "Project: Battery SOH Forecasting\n"
        "Author: Hamidou, MS Physics & EE\n"
        "Status: Validated on Scaled Test Set"
    )
    plt.text(0.95, 0.05, watermark_text, transform=ax.transAxes,
             fontsize=12, color='gray', ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.title("Classification Performance Metrics (%)", fontsize=24, fontweight='bold', pad=20)
    table_path = os.path.join(PLOTS_DIR, "confusion_matrix_results.png")
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"✅ Table saved: {table_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_model()