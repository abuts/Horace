import os
import numpy as np
import matplotlib.pyplot as plt
from pyhorace.io.formats import load_file

def main():
    # Path to the example NeXus file in the Horace repository
    # We use map5935_small.nxspe as it includes the elastic line (E=0)
    nxspe_path = os.path.join("_test", "common_data", "map5935_small.nxspe")
    
    if not os.path.exists(nxspe_path):
        print(f"Error: Could not find {nxspe_path}. Please run this script from the Horace project root.")
        return

    print(f"Loading {nxspe_path}...")
    
    # load_file auto-detects the format and uses read_nxspe
    data = load_file(nxspe_path)
    
    print("\nData keys extracted from NXSPE:")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f" - {k}: array of shape {v.shape}")
        else:
            print(f" - {k}: {v}")

    # Extract components for plotting
    # S has shape (n_detectors, n_energy_bins)
    signal = data['S']
    energy_bins = data['en']
    theta = data.get('theta', np.arange(signal.shape[0]))
    
    print(f"\nPlotting signal matrix (Detectors vs Energy Transfer)...")
    
    # We transpose the signal matrix to match standard (X, Y) -> (Energy, Detector) plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # We use energy bins for X axis, and detector index (or theta) for Y axis
    # Here we just use detector index for simplicity
    detectors = np.arange(signal.shape[0] + 1)
    
    # Handle NaNs and 0s
    sig_masked = np.ma.masked_invalid(signal)
    sig_masked = np.ma.masked_less_equal(sig_masked, 0)
    
    # Use log scale and set a sensible vmax since there are huge outliers (max is 37000, mean is 39)
    from matplotlib.colors import LogNorm
    im = ax.pcolormesh(energy_bins, detectors, sig_masked, cmap='viridis', shading='flat', 
                       norm=LogNorm(vmin=1e-1, vmax=np.nanpercentile(signal, 99)))
    fig.colorbar(im, ax=ax, label='Signal Intensity')
    
    ax.set_title("NeXus SPE Data (MAP11014)")
    ax.set_xlabel("Energy Transfer (meV)")
    ax.set_ylabel("Detector Index")
    
    output_img = "example_nxspe_plot.png"
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{output_img}'")

if __name__ == "__main__":
    main()
