import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import src.load as load

def check_data_for_issues():
    """Check for common data issues that can cause NaNs"""
    print("Loading data...")
    meta, y, schedule, mappings, reverse_mappings = load.load_data('../data/processed/')
    
    print("\n=== Basic Data Checks ===")
    print(f"Meta shape: {meta.shape}")
    print(f"Y shape: {y.shape}")
    print(f"Schedule shape: {schedule.shape}")
    
    print("\n=== Checking for problematic values ===")
    
    # Check y data
    y_sums = np.array(y.sum(axis=1))
    print(f"Y data - Min sum: {np.min(y_sums)}")
    print(f"Y data - Zero sums: {np.sum(y_sums == 0)}")
    print(f"Y data - Near-zero sums (< 1e-10): {np.sum(y_sums < 1e-10)}")
    
    # Check hectares (used for division)
    ha_values = np.array(meta['Ha'])
    print(f"Ha values - Min: {np.min(ha_values)}")
    print(f"Ha values - Zero values: {np.sum(ha_values == 0)}")
    print(f"Ha values - Near-zero values (< 1e-10): {np.sum(ha_values < 1e-10)}")
    
    print("\n=== Checking derived calculations ===")
    
    # Check kilo_dist calculation
    try:
        kilo_dist = (y.to_numpy() / y.to_numpy().sum(axis=1, keepdims=True)).cumsum(axis=1)
        print(f"Kilo_dist - NaNs: {np.isnan(kilo_dist).sum()}")
        print(f"Kilo_dist - Infs: {np.isinf(kilo_dist).sum()}")
        print(f"Kilo_dist - Min: {np.nanmin(kilo_dist)}")
        print(f"Kilo_dist - Max: {np.nanmax(kilo_dist)}")
    except Exception as e:
        print(f"Error in kilo_dist calculation: {e}")
    
    # Check yield_log calculation
    try:
        yield_totals = y.to_numpy().sum(axis=1)
        yield_per_ha = yield_totals / meta['Ha'].to_numpy()
        yield_log = np.log1p(yield_per_ha)
        print(f"Yield_log - NaNs: {np.isnan(yield_log).sum()}")
        print(f"Yield_log - Infs: {np.isinf(yield_log).sum()}")
        print(f"Yield_log - Min: {np.nanmin(yield_log)}")
        print(f"Yield_log - Max: {np.nanmax(yield_log)}")
    except Exception as e:
        print(f"Error in yield_log calculation: {e}")
    
    # Check yield_dist calculation  
    try:
        yield_dist = np.log1p(y.to_numpy() / meta['Ha'].to_numpy()[:, np.newaxis])
        print(f"Yield_dist - NaNs: {np.isnan(yield_dist).sum()}")
        print(f"Yield_dist - Infs: {np.isinf(yield_dist).sum()}")
        print(f"Yield_dist - Min: {np.nanmin(yield_dist)}")
        print(f"Yield_dist - Max: {np.nanmax(yield_dist)}")
    except Exception as e:
        print(f"Error in yield_dist calculation: {e}")
    
    # Check climate data
    try:
        climate_data = np.array(meta.ClimateSeries.to_list())
        print(f"Climate_data - NaNs: {np.isnan(climate_data).sum()}")
        print(f"Climate_data - Infs: {np.isinf(climate_data).sum()}")
    except Exception as e:
        print(f"Error in climate_data: {e}")

if __name__ == "__main__":
    check_data_for_issues() 