import os

class Config:
    BASE_DIR = '../basin_dataset_public'
    
    # Sub-directories
    FORCING_DIR = os.path.join(BASE_DIR, 'basin_mean_forcing', 'nldas')
    FLOW_DIR = os.path.join(BASE_DIR, 'usgs_streamflow')
    BAD_BASINS_FILE = os.path.join(BASE_DIR, 'basin_size_errors_10_percent.txt')

    # --- CONSTANTS ---
    CFS_TO_CMS = 0.0283168
    
    # --- HYPERPARAMETERS ---
    SEQ_LENGTH = 60          # Lookback window (days). 60 covers the lag response we saw.
    PREDICT_HORIZON = 2      # For Task 1: Predict t + k
    PREDICT_STEPS = 5        # For Task 2: Predict sequence t+1...t+5
    
    # --- DATA SPLIT (Hydrological Years) ---
    # Standard CAMELS split:
    # Train: 1980 - 1995 (Calibration)
    # Val:   1995 - 2000
    # Test:  2000 - 2010
    TRAIN_START = '1980-10-01'
    TRAIN_END   = '1995-09-30'
    VAL_START   = '1995-10-01'
    VAL_END     = '2000-09-30'
    TEST_START  = '2000-10-01'
    TEST_END    = '2010-09-30'

    # --- FEATURE SELECTION ---
    # Dynamic Inputs (Time-Series)
    # Naming convention matches the cleaned output from data_loader
    DYNAMIC_FEATURES = [
        'PRCP', 'SRAD', 'Tmax', 'Tmin', 'Vp'
    ]

    # Static Inputs (Basin Attributes)
    # Based on EDA correlations and distributions
    STATIC_FEATURES = [
        'area_gages2',  # Catchment Area (Will be Log transformed)
        'elev_mean',    # Mean Elevation
        'slope_mean',   # Topography
        'sand_frac',    # Soil Type
        'clay_frac',    # Soil Type
        'frac_forest',  # Vegetation
        'lai_max',      # Leaf Area Index
        'p_mean',       # Long-term climate
        'aridity'       # Climate Index
    ]

    TARGET = 'Q_cms'