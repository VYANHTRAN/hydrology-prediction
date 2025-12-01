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

# Import local modules
from config import Config
from loader import CamelsLoader
from feature_engineering import FeatureEngineer
from preprocessing import CamelsPreprocessor
from model import LSTM, LSTM_Seq2Seq

# --- UTILS ---

def calc_nse(obs, sim):
    """Nash-Sutcliffe Efficiency"""
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
    Loads data, applies engineering, fits preprocessor.
    num_basins: int, number of basins to load. 0 means load all.
    """
    print("--- 1. Loading Data ---")
    df_basins = loader.get_basin_list()
    
    if df_basins.empty:
        raise ValueError("No basins found or all filtered out.")
    
    total_found = len(df_basins)
    if num_basins > 0:
        print(f"Found {total_found} basins. Loading first {num_basins} for experiment...")
        df_basins = df_basins.head(num_basins)
    else:
        print(f"Loading all {total_found} basins...")
    
    basin_ids = df_basins['gauge_id'].tolist()
    df_static = loader.load_static_attributes(basin_ids) if cfg.USE_STATIC else None

    dynamic_data = {}
    print("Loading dynamic data...")
    for _, row in tqdm(df_basins.iterrows(), total=len(df_basins)):
        gid = row['gauge_id']
        region = row['region']
        
        df = loader.load_dynamic_data(gid, region)
        if df is not None:
            df = preprocessor.clean_physical_outliers(df)
            df = preprocessor.handle_missing_data(df)
            df = engineer.transform(df)
            df = preprocessor.add_date_features(df)
            dynamic_data[gid] = df
    
    print("Fitting preprocessor...")
    preprocessor.fit(dynamic_data, df_static)
    
    return dynamic_data, df_static, basin_ids

# --- DATASET GENERATORS ---

def get_task1_dataset(cfg, dynamic_data, df_static, preprocessor, basin_ids, split='train'):
    X_list, y_list = [], []
    if split == 'train': start, end = cfg.TRAIN_START, cfg.TRAIN_END
    elif split == 'val': start, end = cfg.VAL_START, cfg.VAL_END
    else: start, end = cfg.TEST_START, cfg.TEST_END

    for gid in basin_ids:
        if gid not in dynamic_data: continue
        df = dynamic_data[gid].loc[start:end]
        if df.empty: continue

        data_matrix, static_vec = preprocessor.transform(df, df_static, gid)
        X, y = preprocessor.create_sequences(data_matrix, static_vec if cfg.USE_STATIC else None, mode='task1')
        
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)
    
    if not X_list: return None, None
    return np.concatenate(X_list), np.concatenate(y_list)

def get_task2_dataset(cfg, dynamic_data, df_static, preprocessor, basin_ids, split='train'):
    X_past_list, X_future_list, Static_list, Y_list = [], [], [], []
    
    if split == 'train': start, end = cfg.TRAIN_START, cfg.TRAIN_END
    elif split == 'val': start, end = cfg.VAL_START, cfg.VAL_END
    else: start, end = cfg.TEST_START, cfg.TEST_END

    # Assuming standardized column indices from Preprocessor
    forcing_indices = [0, 1, 2, 3, 4] 
    
    for gid in basin_ids:
        if gid not in dynamic_data: continue
        df = dynamic_data[gid].loc[start:end]
        if df.empty: continue

        data_matrix, static_vec = preprocessor.transform(df, df_static, gid)
        
        seq_len = cfg.SEQ_LENGTH
        steps = cfg.PREDICT_STEPS
        total = len(data_matrix)
        
        n_dyn = len(cfg.DYNAMIC_FEATURES)
        future_feat_indices = forcing_indices + [n_dyn, n_dyn + 1] # Forcing + Sin + Cos

        static_repeated = np.tile(static_vec, (seq_len, 1)) if (cfg.USE_STATIC and static_vec is not None) else None
        
        c_xp, c_xf, c_st, c_y = [], [], [], []

        for t in range(seq_len, total - steps + 1):
            past_window = data_matrix[t-seq_len:t, :] 
            future_window = data_matrix[t : t+steps, future_feat_indices]
            target_seq = data_matrix[t : t+steps, -1]
            
            if np.isnan(past_window).any() or np.isnan(future_window).any() or np.isnan(target_seq).any():
                continue

            if static_repeated is not None:
                full_past = np.hstack([past_window, static_repeated])
            else:
                full_past = past_window
            
            c_xp.append(full_past)
            c_xf.append(future_window)
            c_st.append(static_vec if static_vec is not None else np.zeros(1))
            c_y.append(target_seq)

        if len(c_xp) > 0:
            X_past_list.append(np.array(c_xp))
            X_future_list.append(np.array(c_xf))
            Static_list.append(np.array(c_st))
            Y_list.append(np.array(c_y))

    if not X_past_list: return None, None, None, None
    return (np.concatenate(X_past_list), np.concatenate(X_future_list), 
            np.concatenate(Static_list), np.concatenate(Y_list))

# --- EXECUTION FUNCTIONS ---

def run_task_1(args, cfg, device, dynamic_data, df_static, preprocessor, basin_ids):
    print("\n" + "="*40)
    print(" TASK 1: Single Step Prediction (t+2)")
    print("="*40)

    # 1. Dataset
    print("Preparing datasets...")
    X_train, y_train = get_task1_dataset(cfg, dynamic_data, df_static, preprocessor, basin_ids, 'train')
    X_val, y_val = get_task1_dataset(cfg, dynamic_data, df_static, preprocessor, basin_ids, 'val')

    if X_train is None:
        print("Not enough data for Task 1.")
        return

    train_ds = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_ds = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Model
    input_dim = X_train.shape[2]
    model = LSTM(input_dim=input_dim, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 3. Train
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
            print("  -> Saved Best Model")

    # 4. Test
    print("Evaluating on Test Set...")
    X_test, y_test = get_task1_dataset(cfg, dynamic_data, df_static, preprocessor, basin_ids, 'test')
    if X_test is not None:
        model.load_state_dict(torch.load(save_path))
        model.eval()
        with torch.no_grad():
            preds = model(torch.Tensor(X_test).to(device)).cpu().numpy().squeeze()
        
        nse = calc_nse(y_test, preds)
        print(f"Task 1 Test NSE: {nse:.4f}")
        save_results({'NSE': nse, 'Test_MSE': float(np.mean((y_test-preds)**2))}, vars(args), 'results/task1')

def run_task_2(args, cfg, device, dynamic_data, df_static, preprocessor, basin_ids):
    print("\n" + "="*40)
    print(" TASK 2: Multi-Step Sequence (t+1..t+5)")
    print("="*40)

    # 1. Dataset
    print("Preparing datasets...")
    tr_data = get_task2_dataset(cfg, dynamic_data, df_static, preprocessor, basin_ids, 'train')
    val_data = get_task2_dataset(cfg, dynamic_data, df_static, preprocessor, basin_ids, 'val')

    if tr_data[0] is None:
        print("Not enough data for Task 2.")
        return

    train_ds = TensorDataset(*[torch.Tensor(x) for x in tr_data])
    val_ds = TensorDataset(*[torch.Tensor(x) for x in val_data])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Model
    input_dim = tr_data[0].shape[2]
    future_dim = tr_data[1].shape[2]
    static_dim = tr_data[2].shape[1]

    model = LSTM_Seq2Seq(input_dim, 64, future_dim, static_dim, cfg.PREDICT_STEPS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 3. Train
    best_loss = float('inf')
    save_path = os.path.join('results', 'task2', 'best_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for xp, xf, st, y in train_loader:
            xp, xf, st, y = xp.to(device), xf.to(device), st.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(xp, xf, st, target_seq=y, teacher_forcing_ratio=0.5)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xp, xf, st, y in val_loader:
                xp, xf, st, y = xp.to(device), xf.to(device), st.to(device), y.to(device)
                preds = model(xp, xf, st, target_seq=None, teacher_forcing_ratio=0)
                val_loss += criterion(preds, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:02d}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("  -> Saved Best Model")

    # 4. Test
    print("Evaluating on Test Set...")
    te_data = get_task2_dataset(cfg, dynamic_data, df_static, preprocessor, basin_ids, 'test')
    if te_data[0] is not None:
        test_ds = TensorDataset(*[torch.Tensor(x) for x in te_data])
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        model.load_state_dict(torch.load(save_path))
        model.eval()
        
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xp, xf, st, y in test_loader:
                xp, xf, st = xp.to(device), xf.to(device), st.to(device)
                out = model(xp, xf, st, target_seq=None, teacher_forcing_ratio=0)
                all_preds.append(out.cpu().numpy())
                all_targets.append(y.numpy())
        
        preds_flat = np.concatenate(all_preds).flatten()
        targets_flat = np.concatenate(all_targets).flatten()
        nse = calc_nse(targets_flat, preds_flat)
        print(f"Task 2 Test NSE: {nse:.4f}")
        save_results({'NSE': nse}, vars(args), 'results/task2')

# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camels Basin Prediction Pipeline")
    
    # Arguments
    parser.add_argument('--task', type=str, choices=['1', '2', 'all'], required=True, 
                        help='Task to run: "1" (t+2 prediction), "2" (Sequence t+1..t+5), or "all"')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--num_basins', type=int, default=5, 
                        help='Number of basins to process (0 = All Basins). Low number for quick testing.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Config & Device
    cfg = Config()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Running on Device: {device}")
    
    # Initialize Classes
    loader = CamelsLoader()
    engineer = FeatureEngineer()
    preprocessor = CamelsPreprocessor()
    
    # Load Data Once
    dynamic_data, df_static, basin_ids = prepare_data(cfg, loader, engineer, preprocessor, args.num_basins)
    
    # Run Tasks
    if args.task in ['1', 'all']:
        run_task_1(args, cfg, device, dynamic_data, df_static, preprocessor, basin_ids)
        
    if args.task in ['2', 'all']:
        run_task_2(args, cfg, device, dynamic_data, df_static, preprocessor, basin_ids)