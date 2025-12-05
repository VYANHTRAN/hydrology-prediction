import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import local modules
from src.config import Config
from src.loader import CamelsLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import CamelsPreprocessor
from src.model import LSTM, LSTM_Seq2Seq

# --- UTILS ---
def calc_nse(obs, sim):
    """
    Calculate metric Nash Sutcliffe efficiency for hydrological model. 
    """
    denominator = np.sum((obs - np.mean(obs)) ** 2) + 1e-6
    numerator = np.sum((sim - obs) ** 2)
    return 1 - (numerator / denominator)

def save_results(results, params, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        params_clean = {k: str(v) for k, v in params.items()}
        json.dump(params_clean, f, indent=4)
    print(f"Results saved to {output_dir}")

# --- DATA PREPARATION ---

def prepare_data(cfg, loader, engineer, preprocessor, num_basins=5):
    """
    1. Loads Data & Splits (keeping NaNs initially)
    2. Fits Imputer on Train Split Only (Learns Climatology)
    3. Imputes Missing Values in Train, Val, Test using learned stats
    4. Fits Scaler on Cleaned Train Data
    """
    df_basins = loader.get_basin_list()
    if df_basins.empty: 
        raise ValueError("No basins found.")
    
    if num_basins > 0:
        print(f"Found {len(df_basins)} basins. Loading first {num_basins}...")
        df_basins = df_basins.head(num_basins)
    
    basin_ids = df_basins['gauge_id'].tolist()
    df_static = loader.load_static_attributes(basin_ids) if cfg.USE_STATIC else None

    # Temporary storage for raw splits 
    raw_train = {}
    raw_val = {}
    raw_test = {}

    print("Loading, Cleaning outliers, and Splitting data...")
    for _, row in tqdm(df_basins.iterrows(), total=len(df_basins)):
        gid = row['gauge_id']
        region = row['region']
        
        # 1. Load Raw
        df = loader.load_dynamic_data(gid, region)
        if df is None: 
            continue

        # 2. Clean Physical Outliers 
        df = preprocessor.clean_physical_outliers(df)
        
        # 3. Feature Engineering (Generates Lags/Rolling which introduces NaNs)
        df = engineer.transform(df)
        df = preprocessor.add_date_features(df)

        # 4. Split Data (Preserving NaNs for now)
        df_tr = df.loc[cfg.TRAIN_START:cfg.TRAIN_END].copy()
        df_val = df.loc[cfg.VAL_START:cfg.VAL_END].copy()
        df_te = df.loc[cfg.TEST_START:cfg.TEST_END].copy()

        if not df_tr.empty:
            raw_train[gid] = df_tr
            raw_val[gid] = df_val
            raw_test[gid] = df_te

    # 5. Fit Imputer (Learn Climatology from Training Data)
    preprocessor.fit_imputer(raw_train)

    # 6. Apply Imputation
    train_data, val_data, test_data = {}, {}, {}

    print("Imputing missing data...")
    for gid in raw_train.keys():
        # Pass gauge_id so preprocessor looks up correct climatology
        train_data[gid] = preprocessor.handle_missing_data(raw_train[gid], gauge_id=gid, train=True)
        
        # Check if validation data exists for this basin
        if gid in raw_val:
            val_data[gid] = preprocessor.handle_missing_data(raw_val[gid], gauge_id=gid, train=False)
        
        # Check if test data exists for this basin
        if gid in raw_test:
            test_data[gid] = preprocessor.handle_missing_data(raw_test[gid], gauge_id=gid, train=False)

    # 7. Fit Scaler ONLY on Cleaned Training Data
    print("Fitting Scaler on Training Set...")
    preprocessor.fit(train_data, df_static)
    
    return train_data, val_data, test_data, df_static, basin_ids

# --- DATASET GENERATORS ---

def get_task1_dataset(cfg, data_dict, df_static, preprocessor, basin_ids):
    X_list, y_list = [], []

    for gid in basin_ids:
        if gid not in data_dict: 
            continue
        df = data_dict[gid] 
        if df.empty: 
            continue

        data_matrix, static_vec = preprocessor.transform(df, df_static, gid)
        X, y = preprocessor.create_sequences(data_matrix, static_vec if cfg.USE_STATIC else None, mode='task1')
        
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)
    
    if not X_list: 
        return None, None
    return np.concatenate(X_list), np.concatenate(y_list)

def get_task2_dataset(cfg, data_dict, df_static, preprocessor, basin_ids):
    X_past_list, X_future_list, Static_list, Y_list = [], [], [], []
    forcing_indices = [i for i, f in enumerate(cfg.DYNAMIC_FEATURES) if f in cfg.FORCING_FEATURES]
    
    for gid in basin_ids:
        if gid not in data_dict: continue
        df = data_dict[gid]
        if df.empty: continue

        data_matrix, static_vec = preprocessor.transform(df, df_static, gid)
        
        seq_len = cfg.SEQ_LENGTH
        steps = cfg.PREDICT_STEPS
        total = len(data_matrix)
        n_dyn = len(cfg.DYNAMIC_FEATURES)

        # Future forcing: Known Forcing indices + Date features (Indices n_dyn, n_dyn+1)
        future_feat_indices = forcing_indices + [n_dyn, n_dyn + 1] 
        
        # Prepare Static
        use_static = (cfg.USE_STATIC and static_vec is not None)
        static_repeated = np.tile(static_vec, (seq_len, 1)) if use_static else None
        
        c_xp, c_xf, c_st, c_y = [], [], [], []

        for i in range(total - seq_len - steps + 1):
            # Past Window: [i, i+seq]
            past_window = data_matrix[i : i+seq_len, :] 
            
            # Future Window: [i+seq, i+seq+steps]
            future_window = data_matrix[i+seq_len : i+seq_len+steps, future_feat_indices]
            
            # Target Sequence: [i+seq, i+seq+steps]
            target_seq = data_matrix[i+seq_len : i+seq_len+steps, -1]
            
            if np.isnan(past_window).any() or np.isnan(future_window).any() or np.isnan(target_seq).any():
                continue

            if use_static:
                full_past = np.hstack([past_window, static_repeated])
            else:
                full_past = past_window
            
            c_xp.append(full_past)
            c_xf.append(future_window)
            
            if use_static:
                c_st.append(static_vec)
            
            c_y.append(target_seq)

        if len(c_xp) > 0:
            X_past_list.append(np.array(c_xp))
            X_future_list.append(np.array(c_xf))
            if use_static:
                Static_list.append(np.array(c_st))
            Y_list.append(np.array(c_y))

    if not X_past_list: return None, None, None, None
    
    # Handle the case where no static features are used
    if not Static_list:
        Static_final = None
    else:
        Static_final = np.concatenate(Static_list)

    return (np.concatenate(X_past_list), np.concatenate(X_future_list), 
            Static_final, np.concatenate(Y_list))

# --- EXECUTION FUNCTIONS ---

def run_task_1(args, cfg, device, train_data, val_data, test_data, df_static, preprocessor, basin_ids):
    print("TASK 1: Single Step Prediction (t+2)")

    X_train, y_train = get_task1_dataset(cfg, train_data, df_static, preprocessor, basin_ids)
    X_val, y_val = get_task1_dataset(cfg, val_data, df_static, preprocessor, basin_ids)

    if X_train is None:
        print("Not enough data for Task 1.")
        return

    train_ds = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_ds = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train.shape[2]
    model = LSTM(input_dim=input_dim, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    save_path = os.path.join('results', 'task1', 'best_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b).squeeze(), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b).squeeze(), y_b).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:02d}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print("Evaluating on Test Set...")
    X_test, y_test = get_task1_dataset(cfg, test_data, df_static, preprocessor, basin_ids)
    if X_test is not None:
        model.load_state_dict(torch.load(save_path))
        model.eval()
        with torch.no_grad():
            preds = model(torch.Tensor(X_test).to(device)).cpu().numpy().squeeze()
        
        nse = calc_nse(y_test, preds)
        print(f"Task 1 Test NSE: {nse:.4f}")
        save_results({'NSE': nse, 'Test_MSE': float(np.mean((y_test-preds)**2))}, vars(args), 'results/task1')

def run_task_2(args, cfg, device, train_data, val_data, test_data, df_static, preprocessor, basin_ids):
    print("\n--- TASK 2: Multi-Step Sequence (t+1..t+5) ---")

    tr_data = get_task2_dataset(cfg, train_data, df_static, preprocessor, basin_ids)
    val_data_tuple = get_task2_dataset(cfg, val_data, df_static, preprocessor, basin_ids)

    if tr_data[0] is None: return

    # Handle Static inputs dynamically
    # If tr_data[2] is None (no static features), we don't pass it to TensorDataset yet
    if tr_data[2] is not None:
        train_ds = TensorDataset(*[torch.Tensor(x) for x in tr_data])
        val_ds = TensorDataset(*[torch.Tensor(x) for x in val_data_tuple])
        static_dim = tr_data[2].shape[1]
    else:
        # Pass dummy placeholder or handle inside loop? 
        # Easier to pass tensors: X_past, X_fut, Y
        train_ds = TensorDataset(torch.Tensor(tr_data[0]), torch.Tensor(tr_data[1]), torch.Tensor(tr_data[3]))
        val_ds = TensorDataset(torch.Tensor(val_data_tuple[0]), torch.Tensor(val_data_tuple[1]), torch.Tensor(val_data_tuple[3]))
        static_dim = 0

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = tr_data[0].shape[2]
    future_dim = tr_data[1].shape[2]

    model = LSTM_Seq2Seq(input_dim, 64, future_dim, static_dim, cfg.PREDICT_STEPS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    save_path = os.path.join('results', 'task2', 'best_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if static_dim > 0:
                xp, xf, st, y = batch
                xp, xf, st, y = xp.to(device), xf.to(device), st.to(device), y.to(device)
            else:
                xp, xf, y = batch
                st = None
                xp, xf, y = xp.to(device), xf.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(xp, xf, st, target_seq=y, teacher_forcing_ratio=0.5)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if static_dim > 0:
                    xp, xf, st, y = batch
                    xp, xf, st, y = xp.to(device), xf.to(device), st.to(device), y.to(device)
                else:
                    xp, xf, y = batch
                    st = None
                    xp, xf, y = xp.to(device), xf.to(device), y.to(device)
                    
                preds = model(xp, xf, st, target_seq=None, teacher_forcing_ratio=0)
                val_loss += criterion(preds, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}: Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print("Evaluating on Test Set...")
    te_data = get_task2_dataset(cfg, test_data, df_static, preprocessor, basin_ids)
    if te_data[0] is not None:
        if static_dim > 0:
            test_ds = TensorDataset(*[torch.Tensor(x) for x in te_data])
        else:
             test_ds = TensorDataset(torch.Tensor(te_data[0]), torch.Tensor(te_data[1]), torch.Tensor(te_data[3]))
             
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        model.load_state_dict(torch.load(save_path))
        model.eval()
        
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                if static_dim > 0:
                    xp, xf, st, y = batch
                    xp, xf, st, y = xp.to(device), xf.to(device), st.to(device), y.to(device)
                else:
                    xp, xf, y = batch
                    st = None
                    xp, xf, y = xp.to(device), xf.to(device), y.to(device)

                out = model(xp, xf, st, target_seq=None, teacher_forcing_ratio=0)
                all_preds.append(out.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        preds_flat = np.concatenate(all_preds).flatten()
        targets_flat = np.concatenate(all_targets).flatten()
        nse = calc_nse(targets_flat, preds_flat)
        print(f"Task 2 Test NSE: {nse:.4f}")
        save_results({'NSE': nse}, vars(args), 'results/task2')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['1', '2', 'all'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_basins', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    
    cfg = Config()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Running on Device: {device}")
    
    loader = CamelsLoader()
    engineer = FeatureEngineer()
    preprocessor = CamelsPreprocessor()
    
    # Updated return signature
    train_data, val_data, test_data, df_static, basin_ids = prepare_data(
        cfg, loader, engineer, preprocessor, args.num_basins
    )
    
    if args.task in ['1', 'all']:
        run_task_1(args, cfg, device, train_data, val_data, test_data, df_static, preprocessor, basin_ids)
        
    if args.task in ['2', 'all']:
        run_task_2(args, cfg, device, train_data, val_data, test_data, df_static, preprocessor, basin_ids)