import os
import numpy as np
import scipy.io as sio
import warnings

# --- CONFIGURATION ---
RAW_PATH = 'data/raw/XJTU'
PROCESSED_PATH = 'data/processed'
MAX_CYCLES = 300  # Normalization constant for age
STRIDE = 10 
SOH_THRESHOLDS = {'S0': 0.98, 'S1': 0.94, 'S2': 0.88}

def get_markov_state(soh):
    if soh > SOH_THRESHOLDS['S0']: return 0
    if soh > SOH_THRESHOLDS['S1']: return 1
    if soh > SOH_THRESHOLDS['S2']: return 2
    return 3

def extract_soh_robustly(data_dict):
    user_keys = [k for k in data_dict.keys() if not k.startswith('__')]
    for key in user_keys:
        val = data_dict[key]
        if isinstance(val, np.ndarray) and val.dtype.names is not None:
            for field in ['Capacity', 'SOH', 'capacity_Ah', 'Capacity_Ah']:
                if field in val.dtype.names:
                    data = val[field][0, 0] if val[field].ndim > 1 else val[field]
                    return data.flatten()
        try:
            struct = val[0, 0]
            if hasattr(struct, 'dtype') and struct.dtype.names is not None:
                for field in ['Capacity', 'SOH', 'capacity_Ah', 'Capacity_Ah']:
                    if field in struct.dtype.names:
                        return struct[field].flatten()
        except:
            continue
    return None

# Renamed to match the naming convention for the project package
def prepare_markov_features(soh, stride=STRIDE):
    """
    FIX: Added epsilon to baseline to prevent RuntimeWarning: divide by zero
    """
    baseline = np.max(soh)
    if baseline < 0.1: return None, None
    
    epsilon = 1e-10
    soh_norm = soh / (baseline + epsilon) # Fixed divide by zero
    
    X, y = [], []
    for t in range(0, len(soh_norm) - stride, stride):
        curr_s = get_markov_state(soh_norm[t])
        next_s = get_markov_state(soh_norm[t + stride])
        
        oh = np.zeros(4); oh[curr_s] = 1
        age_norm = t / MAX_CYCLES
        X.append(np.append(oh, age_norm))
        y.append(next_s)
    return np.array(X), np.array(y)

def preprocess_and_balance():
    if not os.path.exists(RAW_PATH):
        print(f"❌ Path not found: {RAW_PATH}")
        return

    files = [os.path.join(RAW_PATH, f) for f in os.listdir(RAW_PATH) if f.endswith('.mat')]
    X_all, y_all = [], []

    print(f"🔄 Processing {len(files)} files...")

    for f_path in files:
        try:
            mat = sio.loadmat(f_path)
            soh = extract_soh_robustly(mat)
            if soh is None: continue

            X_batt, y_batt = prepare_markov_features(soh)
            if X_batt is not None:
                X_all.extend(X_batt)
                y_all.extend(y_batt)
        except Exception as e:
            print(f"⚠️  Skipped {os.path.basename(f_path)}: {e}")

    if not X_all:
        print("❌ No data extracted.")
        return

    X_all, y_all = np.array(X_all), np.array(y_all)
    u, counts = np.unique(y_all, return_counts=True)
    min_size = min(counts)
    
    print(f"⚖️ Balancing states to {min_size} samples each.")
    
    b_X, b_y = [], []
    for state in u:
        indices = np.where(y_all == state)[0]
        sel = np.random.choice(indices, min_size, replace=False)
        b_X.extend(X_all[sel]); b_y.extend(y_all[sel])

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    b_X, b_y = np.array(b_X), np.array(b_y)
    perm = np.random.permutation(len(b_X))
    split = int(0.8 * len(perm))
    
    np.save(f'{PROCESSED_PATH}/X_train_scaled.npy', b_X[perm[:split]])
    np.save(f'{PROCESSED_PATH}/y_train_scaled.npy', b_y[perm[:split]])
    np.save(f'{PROCESSED_PATH}/X_test_scaled.npy', b_X[perm[split:]])
    np.save(f'{PROCESSED_PATH}/y_test_scaled.npy', b_y[perm[split:]])

    print(f"✅ Success! Processed files saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess_and_balance()