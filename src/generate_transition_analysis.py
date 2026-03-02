import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
epsilon = 1e-8  # Small constant to prevent division by zero
# --- 1. CONFIGURATION & STYLE ---
plt.rcParams.update({
    'font.size': 24, 
    'axes.labelsize': 26, 
    'axes.titlesize': 28, 
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.linewidth': 3
})

SOH_THRESHOLDS = {'S0': 0.98, 'S1': 0.94, 'S2': 0.88}

# --- Add project root to sys.path to import the inference class ---
#project_root = r"D:\DEEP-LEARNING\Markov_Model\DTMC"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_root = "."

# Best practice: use os.path.join for cross-platform compatibility
TEST_FILE = os.path.join(project_root, "data", "raw", "unforeseen", "3C_battery-4.mat")


if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference import BatteryMarkovInference

MODEL_PATH = os.path.join(project_root, "models", "best_model.pth")
MAX_CYCLES = 300.0  # Normalization constant used during training

# --- 2. REQUIRED FUNCTIONS ---

def extract_numeric_array(data):
    """Recursively extract numeric values from nested structures"""
    if data is None: return []
    if isinstance(data, (int, float, np.float64)): return [float(data)]
    if isinstance(data, np.ndarray):
        if data.dtype.names is not None:
            return extract_numeric_array(data[data.dtype.names[0]])
        flattened = []
        for item in data.flatten():
            flattened.extend(extract_numeric_array(item))
        return flattened
    return []

def extract_soh_robustly(data_dict):
    """Safely navigates XJTU nested structs to find capacity data"""
    if 'data' in data_dict:
        struct = data_dict['data'][0, 0]
        if hasattr(struct, 'dtype'):
            for k in ['capacity_Ah', 'SOH', 'Capacity', 'Capacity_Ah']:
                if k in struct.dtype.names:
                    print(f"✅ Found field: {k}")
                    return np.array(extract_numeric_array(struct[k]))
    return None

def get_markov_state(soh):
    """Maps SOH to discrete Markov health phases"""
    if soh >= SOH_THRESHOLDS['S0']: return 0
    if soh >= SOH_THRESHOLDS['S1']: return 1
    if soh >= SOH_THRESHOLDS['S2']: return 2
    return 3

# --- 3. DATA PROCESSING ---
print(f"📂 Loading: {TEST_FILE}")
mat = sio.loadmat(TEST_FILE)
raw_capacity = extract_soh_robustly(mat)

if raw_capacity is not None and len(raw_capacity) > 0:
    # Smooth to match the "Clean" graph
    smoothed = np.convolve(raw_capacity, np.ones(5)/5, mode='valid')
    baseline = np.max(smoothed[:10])
    soh_norm = smoothed / (baseline +epsilon)  # Normalize to vendor's nominal capacity

    # Actual Markov states for all cycles
    actual_states = [get_markov_state(s) for s in soh_norm]

    # --- Initialize the trained model ---
    print("🔧 Loading model...")
    infer = BatteryMarkovInference(model_path=MODEL_PATH)

    # --- 4. GENERATE MODEL PREDICTIONS (1-step ahead) ---
    predicted_next = []          # will hold model's prediction for cycle i+1
    for i in range(len(actual_states) - 1):
        # Current state
        curr_state = actual_states[i]
        # One-hot encoding
        state_oh = torch.zeros(4)
        state_oh[curr_state] = 1.0
        # Normalized age (current cycle / MAX_CYCLES)
        norm_age = min(i / MAX_CYCLES, 1.0)
        # Get model prediction
        pred_idx, _ = infer.predict_next_state(state_oh, norm_age)
        predicted_next.append(pred_idx)

    # --- 5. PREPARE COMPARISON DATA ---
    # Actual next states (for cycles 1..end) correspond to predictions made at cycles 0..end-1
    actual_next = actual_states[1:]
    # Both lists have the same length (N-1)
    x_axis = np.arange(len(actual_next))   # i = current cycle index

    # --- 6. THE COMPARISON PLOT ---

    # 1. Standardizes all minus signs and hyphens
    plt.rcParams['axes.unicode_minus'] = False 

    # 2. Use a font that handles specialized characters better than Arial
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(22, 11))

    # Actual next state (blue solid)
    plt.step(x_axis, actual_next, where='post', color='#1f77b4', 
             linewidth=7, label='Actual Next State', alpha=0.9)
    
    # Model forecast (orange dashed)
    plt.step(x_axis, predicted_next, where='post', color='#ff7f0e', 
             linewidth=5, linestyle='--', label='Model Forecast (1‑step ahead)', alpha=0.8)

    # Invert Y-axis: Healthy at top (0), Failure at bottom (3)
    plt.yticks([0, 1, 2, 3], ['S0: Healthy', 'S1: Good', 'S2: Warning', 'S3: Failed'])
    plt.gca().invert_yaxis() 

    plt.title("Model Verification: 1‑Cycle Forecast vs. Ground Truth", fontweight='bold', pad=30)
    plt.xlabel("Current Cycle Number (i)", fontweight='bold', labelpad=20)
    plt.ylabel("Markov State at Cycle i+1", fontweight='bold', labelpad=20)
    
    plt.legend(loc='center', frameon=True, shadow=True, borderpad=1)
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)

    # Save the plot before showing
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/actual_vs_predicted_model.png", dpi=200)
    plt.tight_layout()

    # show the plot
    plt.show()
else:
    print("❌ Error: Extraction failed.")