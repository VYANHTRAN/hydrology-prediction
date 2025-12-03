import os
import glob
import pandas as pd
import numpy as np
from config import Config

class CamelsLoader:
    def __init__(self):
        self.cfg = Config()

    def load_bad_basins(self):
        """Returns a list of basin IDs to exclude."""
        if not os.path.exists(self.cfg.BAD_BASINS_FILE):
            return []
        
        bad_ids = []

        try:
            with open(self.cfg.BAD_BASINS_FILE, 'r') as f:
                next(f) 
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        bad_ids.append(parts[1])
        except Exception as e:
            print(f"Failed to parse bad basins file: {e}")
        return bad_ids

    def get_basin_list(self):
        """Scans directories and filters out bad basins."""
        bad_basins = self.load_bad_basins()
        
        # Construct file paths for streamflow files 
        search_path = os.path.join(self.cfg.FLOW_DIR, '**', '*_streamflow_qc.txt')
        files = glob.glob(search_path, recursive=True)
        
        basins = []
        for f in files:
            # Extract folder ID and gauge ID of each basins 
            parts = f.split(os.sep)
            region = parts[-2]
            gauge_id = parts[-1].split('_')[0]
            
            if gauge_id not in bad_basins:
                basins.append({'gauge_id': gauge_id, 'region': region})
                
        return pd.DataFrame(basins)

    def load_dynamic_data(self, gauge_id, region):
        """
        Loads Streamflow + Forcing. 
        Returns cleaned DataFrame with standard column names.
        """
        # 1. Load Streamflow
        flow_path = os.path.join(self.cfg.FLOW_DIR, region, f'{gauge_id}_streamflow_qc.txt')
        try:
            df_flow = pd.read_csv(flow_path, sep=r'\s+', header=None,
                                  names=['gauge_id', 'Year', 'Month', 'Day', 'Q_cfs', 'QC'])
        except: 
            return None

        df_flow['Date'] = pd.to_datetime(df_flow[['Year', 'Month', 'Day']])
        df_flow.set_index('Date', inplace=True)
        # Convert to CMS and handle missing values later (in preprocessing)
        df_flow['Q_cms'] = df_flow['Q_cfs'].replace(-999, np.nan) * self.cfg.CFS_TO_CMS

        # 2. Load Forcing (NLDAS)
        forcing_path = os.path.join(self.cfg.FORCING_DIR, region, f'{gauge_id}_lump_nldas_forcing_leap.txt')
        if not os.path.exists(forcing_path): 
            return None

        try:
            df_force = pd.read_csv(forcing_path, sep=r'\s+', skiprows=3)
        except: return None

        # Rename columns 
        col_map = {
            'Mnth': 'Month', 'month': 'Month', 'mo': 'Month',
            'year': 'Year', 'yr': 'Year',
            'day': 'Day', 'dy': 'Day',
            'prcp(mm/day)': 'PRCP', 'srad(w/m2)': 'SRAD', 
            'tmax(c)': 'Tmax', 'tmin(c)': 'Tmin', 'vp(pa)': 'Vp'
        }
        
        new_cols = {}
        for c in df_force.columns:
            clean = c.strip()
            # Check lowercase match against map
            if clean.lower() in col_map:
                new_cols[c] = col_map[clean.lower()]
            elif clean in col_map:
                new_cols[c] = col_map[clean]
        
        df_force.rename(columns=new_cols, inplace=True)
        
        # Create Date Index
        try:
            df_force['Date'] = pd.to_datetime(df_force[['Year', 'Month', 'Day']])
            df_force.set_index('Date', inplace=True)
        except KeyError:
            return None

        # 3. Merge
        # Inner join to ensure alignment
        cols_to_use = [c for c in self.cfg.DYNAMIC_FEATURES if c in df_force.columns]
        df_merged = df_flow[['Q_cms']].join(df_force[cols_to_use], how='inner')
        
        return df_merged

    def load_static_attributes(self, basins_list=None):
        """
        Loads all attribute files, merges them, and filters for requested features.
        """
        files = ['camels_topo.txt', 'camels_soil.txt', 'camels_clim.txt', 
                 'camels_vege.txt', 'camels_geol.txt']
        dfs = []

        for filename in files:
            path = os.path.join(self.cfg.BASE_DIR, filename)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, sep=';')
                    df.columns = [c.strip() for c in df.columns]
                    if 'gauge_id' in df.columns:
                        df['gauge_id'] = df['gauge_id'].astype(str).str.zfill(8)
                        df.set_index('gauge_id', inplace=True)
                        dfs.append(df)
                except: pass
        
        if not dfs: 
            return None

        # Merge all static files
        df_static = pd.concat(dfs, axis=1)
        # Remove duplicate columns
        df_static = df_static.loc[:, ~df_static.columns.duplicated()]

        # Filter for the features defined in Config
        available_feats = [f for f in self.cfg.STATIC_FEATURES if f in df_static.columns]
        df_final = df_static[available_feats]

        if basins_list is not None:
             # Only keep basins that exist in our dynamic list
            df_final = df_final.reindex(basins_list)
            
        return df_final