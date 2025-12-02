import numpy as np
import pandas as pd 
from .config import Config

class CamelsPreprocessor:
    def __init__(self):
        self.cfg = Config()
        self.scalers = {} 
        self.basin_scalers = {} 
        
        self.climatology_means = None 
        self.global_means = None 
        
        # Physical Constraints
        self.PHYSICAL_LIMITS = {
            'PRCP': {'min': 0.0, 'max': None},
            'Q_cms': {'min': 0.0, 'max': None},
            'Tmax': {'min': -60.0, 'max': 60.0},
            'Tmin': {'min': -60.0, 'max': 60.0}
        }
        self.MAX_INTERPOLATE_GAP = 2

    def add_date_features(self, df):
        """Encode Time Series into Cyclical Features."""
        day_of_year = df.index.dayofyear
        df['sin_doy'] = np.sin(2 * np.pi * day_of_year / 365.0)
        df['cos_doy'] = np.cos(2 * np.pi * day_of_year / 365.0)
        return df

    def clean_physical_outliers(self, df):
        """Ensure the data does not exceed logical constraints."""
        # Ensure that the precipiation could not be negative.
        for col in ['PRCP', self.cfg.TARGET]:
            if col in df.columns:
                mask = df[col] < 0
                if mask.any(): 
                    df.loc[mask, col] = 0.0
        
        # Ensure that the temperature does not exceed the logical constraints. 
        for col in ['Tmax', 'Tmin']:
            if col in df.columns:
                limits = self.PHYSICAL_LIMITS[col]
                mask = (df[col] < limits['min']) | (df[col] > limits['max'])
                if mask.any(): 
                    df.loc[mask, col] = np.nan
        return df

    def fit_imputer(self, train_data_dict):
        """
        Learn seasonal patterns from training set and impute using that patterns.
        Example: if a value in 15-01-1990, impute using values from the same day in different years.
        """     
        dfs = []
        for _, df in train_data_dict.items():
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            print("Warning: No training data to fit imputer.")
            return

        full_train = pd.concat(dfs, axis=0)
        
        cols_to_learn = self.cfg.DYNAMIC_FEATURES + [self.cfg.TARGET]
        cols_to_learn = [c for c in cols_to_learn if c in full_train.columns]
        
        # 1. Calculate global mean for fallback 
        self.global_means = full_train[cols_to_learn].mean()

        # 2. Calculate Climatological Mean 
        # Group by day of year 
        full_train['doy'] = full_train.index.dayofyear
        self.climatology_means = full_train.groupby('doy')[cols_to_learn].mean()
        
        # Handle rare cases of dates that have not appeared in training set 
        # Fill missing values by global mean in these cases 
        expected_days = np.arange(1, 367)
        self.climatology_means = self.climatology_means.reindex(expected_days).fillna(self.global_means)

    def handle_missing_data(self, df):
        """
        1. Linear Interpolation for short gaps.
        2. Fill bigger gaps by climatology means.
        """
        cols_to_fix = [self.cfg.TARGET] + self.cfg.DYNAMIC_FEATURES
        cols_to_fix = [c for c in cols_to_fix if c in df.columns]

        # 1. Linear Interpolate short gaps 
        for col in cols_to_fix:
            df[col] = df[col].interpolate(method='linear', limit=self.MAX_INTERPOLATE_GAP, limit_direction='both')

        # 2. Fill remaining missing values by climatological mean. 
        if self.climatology_means is not None:
            # Use day of year as indexes for df 
            doy_series = df.index.dayofyear
            
            # Create a reference df with climatological mean 
            # df.index ->  dayofyear -> search reference df for self.climatology_means
            climatology_values = self.climatology_means.loc[doy_series, cols_to_fix]
            
            # Reset index of climatology_values 
            climatology_values.index = df.index
            
            # Fill NaN by climatological mean
            df.fillna(climatology_values, inplace=True)
        
        # Fallback with Global Mean 
        if self.global_means is not None:
            df.fillna(self.global_means, inplace=True)
            
        df.fillna(0, inplace=True)
        
        return df

    def fit(self, dynamic_data_dict, static_df=None):
        """
        Calculate Mean/Std (only fit on training data). 
        """
        # 1. Dynamic Stats
        dyn_vals = []
        for gid, df in dynamic_data_dict.items():
            valid_rows = df[self.cfg.DYNAMIC_FEATURES]
            if not valid_rows.empty:
                dyn_vals.append(valid_rows.values)
        
        if dyn_vals:
            all_dyn = np.vstack(dyn_vals)
            self.scalers['dynamic_mean'] = np.mean(all_dyn, axis=0)
            self.scalers['dynamic_std']  = np.std(all_dyn, axis=0) + 1e-6
        else:
            self.scalers['dynamic_mean'] = 0
            self.scalers['dynamic_std'] = 1

        # 2. Static Stats
        if static_df is not None:
            s_df = static_df.copy()
            if 'area_gages2' in s_df.columns:
                s_df['area_gages2'] = np.log10(np.maximum(s_df['area_gages2'], 1e-3))
            self.scalers['static_mean'] = s_df.mean().values
            self.scalers['static_std']  = s_df.std().values + 1e-6

        # 3. Basin Target Stats
        for gid, df in dynamic_data_dict.items():
            clean_target = df[self.cfg.TARGET]
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