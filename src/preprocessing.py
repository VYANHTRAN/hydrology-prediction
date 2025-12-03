import numpy as np
import pandas as pd 
from config import Config

class CamelsPreprocessor:
    def __init__(self):
        self.cfg = Config()
        self.scalers = {} 
        self.basin_scalers = {} 
        
        # Physical Constraints
        self.PHYSICAL_LIMITS = {
            'PRCP': {'min': 0.0, 'max': None},
            'Q_cms': {'min': 0.0, 'max': None},
            'Tmax': {'min': -60.0, 'max': 60.0},
            'Tmin': {'min': -60.0, 'max': 60.0}
        }
        self.MAX_INTERPOLATE_GAP = 5 

    def add_date_features(self, df):
        day_of_year = df.index.dayofyear
        df['sin_doy'] = np.sin(2 * np.pi * day_of_year / 365.0)
        df['cos_doy'] = np.cos(2 * np.pi * day_of_year / 365.0)
        return df

    def clean_physical_outliers(self, df):
        # 1. Negative Rain/Flow -> 0
        for col in ['PRCP', self.cfg.TARGET]:
            if col in df.columns:
                mask = df[col] < 0
                if mask.any(): df.loc[mask, col] = 0.0
        
        # 2. Unrealistic Temp -> NaN
        for col in ['Tmax', 'Tmin']:
            if col in df.columns:
                limits = self.PHYSICAL_LIMITS[col]
                mask = (df[col] < limits['min']) | (df[col] > limits['max'])
                if mask.any(): df.loc[mask, col] = np.nan
        return df

    def handle_missing_data(self, df):
        cols_to_fix = [self.cfg.TARGET] + self.cfg.DYNAMIC_FEATURES
        cols_to_fix = [c for c in cols_to_fix if c in df.columns]

        for col in cols_to_fix:
            # Linear Interpolate short gaps only
            df[col] = df[col].interpolate(method='linear', limit=self.MAX_INTERPOLATE_GAP, limit_direction='both')
            # Handle edges
            df[col] = df[col].ffill().bfill()
        return df

    def fit(self, dynamic_data_dict, static_df=None):
        """
        Computes stats. Handles case where static_df is None.
        """
        print("Computing global statistics...")
        
        # 1. Dynamic Stats
        dyn_vals = []
        for gid, df in dynamic_data_dict.items():
            train_slice = df.loc[self.cfg.TRAIN_START:self.cfg.TRAIN_END]
            if not train_slice.empty:
                valid_rows = train_slice[self.cfg.DYNAMIC_FEATURES].dropna()
                if not valid_rows.empty:
                    dyn_vals.append(valid_rows.values)
        
        if dyn_vals:
            all_dyn = np.vstack(dyn_vals)
            self.scalers['dynamic_mean'] = np.mean(all_dyn, axis=0)
            self.scalers['dynamic_std']  = np.std(all_dyn, axis=0) + 1e-6
        else:
            self.scalers['dynamic_mean'] = 0
            self.scalers['dynamic_std'] = 1

        # 2. Static Stats (Only if provided)
        # Note: Log-transform of area_gages2 is done in transform() method to avoid double transformation
        if static_df is not None:
            static_df_copy = static_df.copy()
            if 'area_gages2' in static_df_copy.columns:
                static_df_copy['area_gages2'] = np.log10(np.maximum(static_df_copy['area_gages2'], 1e-3))
            self.scalers['static_mean'] = static_df_copy.mean().values
            self.scalers['static_std']  = static_df_copy.std().values + 1e-6
        else:
            print("-> Skipping Static Stats (Static Data not provided)")

        # 3. Basin Target Stats
        print("Computing basin-specific target statistics...")
        for gid, df in dynamic_data_dict.items():
            train_slice = df.loc[self.cfg.TRAIN_START:self.cfg.TRAIN_END]
            clean_target = train_slice[self.cfg.TARGET].dropna()
            
            if not clean_target.empty:
                self.basin_scalers[gid] = {'mean': clean_target.mean(), 'std': clean_target.std() + 1e-6}
            else:
                self.basin_scalers[gid] = {'mean': 0, 'std': 1}

    def transform(self, df_dynamic, df_static=None, gauge_id=None):
        """
        Returns: 
           data_matrix (Time, Features)
           static_norm (Vector) OR None (if df_static is None)
        """
        # 1. Norm Dynamic
        dyn_cols = self.cfg.DYNAMIC_FEATURES
        X_dyn = df_dynamic[dyn_cols].values
        X_dyn_norm = (X_dyn - self.scalers['dynamic_mean']) / self.scalers['dynamic_std']
        
        # 2. Norm Target
        target = df_dynamic[self.cfg.TARGET].values
        b_stats = self.basin_scalers.get(gauge_id, {'mean': 0, 'std': 1})
        y_norm = (target - b_stats['mean']) / b_stats['std']
        
        # 3. Norm Static (Conditional)
        X_stat_norm = None
        if df_static is not None and gauge_id in df_static.index:
            static_vals = df_static.loc[gauge_id].values.copy()
            if 'area_gages2' in df_static.columns:
                area_idx = df_static.columns.get_loc('area_gages2')
                static_vals[area_idx] = np.log10(np.maximum(static_vals[area_idx], 1e-3))
            X_stat_norm = (static_vals - self.scalers['static_mean']) / self.scalers['static_std']
        
        # 4. Date Features
        date_feats = df_dynamic[['sin_doy', 'cos_doy']].values
        
        # Matrix: [Dynamic_Norm, Date, Target_Norm]
        data_matrix = np.column_stack([X_dyn_norm, date_feats, y_norm])
        
        return data_matrix, X_stat_norm

    def create_sequences(self, data_matrix, static_vec=None, mode='task1'):
        """
        Creates sequences.
        If static_vec is None -> Returns sequence of only dynamic features.
        If static_vec exists -> Returns sequence of [Dynamic + Static].
        """
        X_seq, y_seq = [], []
        seq_len = self.cfg.SEQ_LENGTH
        total_samples = len(data_matrix)
        
        # Check if we are adding static features
        use_static = (static_vec is not None)
        
        if use_static:
            static_repeated = np.tile(static_vec, (seq_len, 1))

        if mode == 'task1': 
            horizon = self.cfg.PREDICT_HORIZON
            for t in range(seq_len, total_samples - horizon + 1):
                window_data = data_matrix[t-seq_len:t, :]
                target_val = data_matrix[t + horizon - 1, -1]
                
                # Check for long gaps (NaNs)
                if np.isnan(window_data).any() or np.isnan(target_val):
                    continue
                
                if use_static:
                    full_X = np.hstack([window_data, static_repeated])
                else:
                    full_X = window_data
                
                X_seq.append(full_X)
                y_seq.append(target_val)

        elif mode == 'task2':
            steps = self.cfg.PREDICT_STEPS
            for t in range(seq_len, total_samples - steps + 1):
                window_data = data_matrix[t-seq_len:t, :]
                target_seq = data_matrix[t : t+steps, -1]
                
                if np.isnan(window_data).any() or np.isnan(target_seq).any():
                    continue
                
                if use_static:
                    full_X = np.hstack([window_data, static_repeated])
                else:
                    full_X = window_data
                
                X_seq.append(full_X)
                y_seq.append(target_seq)
                
        return np.array(X_seq), np.array(y_seq)