import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config

class CamelsPreprocessor:
    def __init__(self, config):
        self.cfg = config
        self.scalers = {}  # Stores mean/std for global features
        self.basin_scalers = {} # Stores mean/std for target per basin

    def add_date_features(self, df):
        """Adds Cyclical Day-of-Year features."""
        day_of_year = df.index.dayofyear
        df['sin_doy'] = np.sin(2 * np.pi * day_of_year / 365.0)
        df['cos_doy'] = np.cos(2 * np.pi * day_of_year / 365.0)
        return df

    def clean_streamflow(self, df):
        """
        Handles missing streamflow (-999 converted to NaN in loader).
        Strategy: Linear Interpolation.
        """
        if df[self.cfg.TARGET].isna().sum() > 0:
            # Linear interpolate, limit direction='both' to catch ends
            df[self.cfg.TARGET] = df[self.cfg.TARGET].interpolate(method='linear', limit_direction='both')
        
        # If any remain (e.g. empty dataframe), fill with 0
        df[self.cfg.TARGET] = df[self.cfg.TARGET].fillna(0)
        return df

    def fit(self, dynamic_data_dict, static_df):
        """
        Computes Mean/Std for normalization based on TRAINING data.
        
        dynamic_data_dict: {gauge_id: DataFrame}
        static_df: DataFrame of static attributes
        """
        print("Computing global statistics...")
        
        # 1. Global Stats for Dynamic Inputs (Precip, Temp, etc.)
        # We aggregate a subset to avoid memory overflow, or run incremental mean
        dyn_vals = []
        for gid, df in dynamic_data_dict.items():
            # Only fit on Training Period
            train_slice = df.loc[self.cfg.TRAIN_START:self.cfg.TRAIN_END]
            if not train_slice.empty:
                dyn_vals.append(train_slice[self.cfg.DYNAMIC_FEATURES].values)
        
        all_dyn = np.vstack(dyn_vals)
        self.scalers['dynamic_mean'] = np.nanmean(all_dyn, axis=0)
        self.scalers['dynamic_std']  = np.nanstd(all_dyn, axis=0) + 1e-6 # Epsilon for zero div

        # 2. Global Stats for Static Inputs
        # Log Transform Area first
        if 'area_gages2' in static_df.columns:
            static_df['area_gages2'] = np.log10(static_df['area_gages2'] + 1)
            
        self.scalers['static_mean'] = static_df.mean().values
        self.scalers['static_std']  = static_df.std().values + 1e-6

        # 3. Basin-Specific Stats for Target (Flow)
        print("Computing basin-specific target statistics...")
        for gid, df in dynamic_data_dict.items():
            train_slice = df.loc[self.cfg.TRAIN_START:self.cfg.TRAIN_END]
            if not train_slice.empty:
                q_mean = train_slice[self.cfg.TARGET].mean()
                q_std = train_slice[self.cfg.TARGET].std() + 1e-6
                self.basin_scalers[gid] = {'mean': q_mean, 'std': q_std}
            else:
                self.basin_scalers[gid] = {'mean': 0, 'std': 1} # Fallback

    def transform(self, df_dynamic, df_static, gauge_id):
        """
        Normalizes data for a specific basin.
        """
        # 1. Normalize Dynamic Inputs (Global Z-Score)
        dyn_cols = self.cfg.DYNAMIC_FEATURES
        X_dyn = df_dynamic[dyn_cols].values
        X_dyn_norm = (X_dyn - self.scalers['dynamic_mean']) / self.scalers['dynamic_std']
        
        # 2. Normalize Target (Basin Z-Score)
        target = df_dynamic[self.cfg.TARGET].values
        b_stats = self.basin_scalers.get(gauge_id, {'mean': 0, 'std': 1})
        y_norm = (target - b_stats['mean']) / b_stats['std']
        
        # 3. Normalize Static Inputs (Global Z-Score)
        # Handle Area Log Transform
        static_vals = df_static.loc[gauge_id].values.copy()
        # Find index of area to log transform manually or assume df_static passed here is already transformed?
        # Better: Assume fit() modifies df_static in place or we re-apply log here.
        # For safety, let's assume raw input and re-apply log if needed.
        if 'area_gages2' in df_static.columns:
            area_idx = df_static.columns.get_loc('area_gages2')
            # Check if it's already logged (simple heuristic or just re-log if raw)
            # To correspond with fit(), we should apply log here.
            static_vals[area_idx] = np.log10(static_vals[area_idx] + 1)

        X_stat_norm = (static_vals - self.scalers['static_mean']) / self.scalers['static_std']
        
        # 4. Attach Date Features (No normalization needed for Sin/Cos)
        date_feats = df_dynamic[['sin_doy', 'cos_doy']].values
        
        # Concatenate: [Dynamic_Norm, Date, Target_Norm]
        # Note: We keep target in the matrix for easy windowing, will separate later
        data_matrix = np.column_stack([X_dyn_norm, date_feats, y_norm])
        
        return data_matrix, X_stat_norm

    def create_sequences(self, data_matrix, static_vec, mode='task1'):
        """
        Generates (X, y) sequences.
        data_matrix shape: [Time, Dyn_Feats + 2 + 1] (Last col is Target)
        static_vec shape:  [Static_Feats]
        """
        X_seq, y_seq = [], []
        seq_len = self.cfg.SEQ_LENGTH
        
        # Extract columns
        # Inputs = Dynamic (All except last) + Static (Repeated)
        # Target = Last column (Flow)
        
        num_dyn_feats = data_matrix.shape[1] - 1 # Exclude target
        total_samples = len(data_matrix)
        
        if mode == 'task1': # Predict t+k
            horizon = self.cfg.PREDICT_HORIZON # e.g. 2
            
            for t in range(seq_len, total_samples - horizon + 1):
                # Input: t-seq_len to t
                dyn_window = data_matrix[t-seq_len:t, :-1] # Exclude target from input features? 
                # OPTION A: Previous Flow IS an input (Auto-regressive). 
                # Yes, EDA showed high autocorrelation. We MUST include Flow(t-1) in input.
                # So Input window is ALL columns (Dyn + Date + Flow) up to t.
                
                # Let's reshape: Input is [Dyn, Date, Flow_Norm]
                window_data = data_matrix[t-seq_len:t, :] 
                
                # Repeat static features for each timestep
                static_repeated = np.tile(static_vec, (seq_len, 1))
                
                # Full X: [Time, Dyn + Date + Flow + Static]
                full_X = np.hstack([window_data, static_repeated])
                
                # Target: Flow at t + horizon - 1 (index adjustment)
                # If t is current time, we want to predict (t + horizon)
                # Data index t corresponds to "today" relative to the window ending at t.
                target_val = data_matrix[t + horizon - 1, -1] # The Flow column
                
                X_seq.append(full_X)
                y_seq.append(target_val)

        elif mode == 'task2': # Predict sequence t+1...t+5
            steps = self.cfg.PREDICT_STEPS
            
            for t in range(seq_len, total_samples - steps + 1):
                window_data = data_matrix[t-seq_len:t, :]
                static_repeated = np.tile(static_vec, (seq_len, 1))
                full_X = np.hstack([window_data, static_repeated])
                
                # Target: Sequence from t to t+steps
                target_seq = data_matrix[t : t+steps, -1]
                
                X_seq.append(full_X)
                y_seq.append(target_seq)
                
        return np.array(X_seq), np.array(y_seq)