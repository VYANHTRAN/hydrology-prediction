import pandas as pd
from config import Config

class FeatureEngineer:
    def __init__(self):
        self.cfg = Config()

    def add_rolling_features(self, df):
        """
        Calculates rolling statistics for Precipitation.
        Purpose: Represents accumulated soil moisture / wetness.
        """
        # Ensure PRCP exists (it might be named differently if not cleaned, 
        # but our data_loader standardizes it to 'PRCP')
        if 'PRCP' in df.columns:
            # 3-Day Rolling Mean (Short-term wetness)
            df['PRCP_roll3'] = df['PRCP'].rolling(window=3, min_periods=1).mean()
            
            # 7-Day Rolling Mean (Medium-term saturation)
            df['PRCP_roll7'] = df['PRCP'].rolling(window=7, min_periods=1).mean()
        
        return df

    def add_lag_features(self, df):
        """
        Adds explicit lag features for Streamflow.
        Purpose: Give the model explicit access to yesterday's flow value 
        at the current timestep t, without relying solely on LSTM memory.
        """
        target = self.cfg.TARGET # 'Q_cms'
        
        if target in df.columns:
            # Flow yesterday (t-1)
            df['Q_lag1'] = df[target].shift(1)
            
            # Flow 2 days ago (t-2)
            df['Q_lag2'] = df[target].shift(2)
            
            # Flow 3 days ago (t-3)
            df['Q_lag3'] = df[target].shift(3)
            
            # Note: Shifting creates NaNs at the top of the dataframe.
            # These must be handled (backfilled) before normalization.
            df[[f'Q_lag{i}' for i in range(1, 4)]] = df[[f'Q_lag{i}' for i in range(1, 4)]].bfill()
            
        return df

    def transform(self, df):
        """
        Wrapper to run all engineering steps.
        Should be run AFTER cleaning missing data, but BEFORE normalization.
        """
        df = self.add_rolling_features(df)
        df = self.add_lag_features(df)
        return df