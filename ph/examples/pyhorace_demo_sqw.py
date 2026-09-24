import os
import numpy as np
from pyhorace.io.formats import load_file

def main():
    # Path to a Horace v4 HDF5 .sqw file
    sqw_path = os.path.join("_test", "common_data", "faccess_sqw_v4_sample.sqw")
    
    if not os.path.exists(sqw_path):
        print(f"Error: Could not find {sqw_path}. Please run this script from the Horace project root.")
        return

    print(f"Loading modern HDF5 SQW file: {sqw_path}...")
    
    # load_file delegates to SQWFileReader for .sqw extensions
    sqw_data = load_file(sqw_path)
    
    print("\n--- Main Header ---")
    main_header = sqw_data.get('main_header', {})
    for k, v in main_header.items():
        # Clean up byte strings if necessary
        if isinstance(v, bytes):
            v = v.decode('utf-8', errors='ignore')
        print(f"{k}: {v}")
        
    print("\n--- Experiment Info ---")
    experiment_info = sqw_data.get('experiment_info', [])
    print(f"Number of experimental runs: {len(experiment_info)}")
    if len(experiment_info) > 0:
        first_run = experiment_info[0]
        # Safely print a few key fields if they exist
        ei = first_run.get('en', first_run.get('efix', 'Unknown'))
        print(f"Run 1 Incident Energy (Ei): {ei}")

    print("\n--- Data (DnD Grid) ---")
    grid = sqw_data.get('data', {})
    
    if 's' in grid:
        signal = grid['s']
        print(f"Signal shape: {signal.shape}")
        print(f"Dimensions: {signal.ndim}D dataset")
    
    if 'p' in grid:
        axes = grid['p']
        print(f"Number of axes: {len(axes)}")
        for i, ax_bins in enumerate(axes):
            print(f"  Axis {i+1} bins: {len(ax_bins)} (min: {ax_bins[0]:.3f}, max: {ax_bins[-1]:.3f})")

if __name__ == "__main__":
    main()
