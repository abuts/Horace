import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    # RbMnF3 is a simple cubic antiferromagnet.
    # Its spin wave dispersion along [H, 0, 0] can be approximated by:
    # E(q) = sqrt( E_0^2 + (c * sin(pi * q))^2 )
    
    # Define momentum transfer Q along [H, 0, 0]
    q = np.linspace(-0.5, 1.5, 200)
    # Define Energy transfer E
    e = np.linspace(0, 15, 200)
    
    Q, E = np.meshgrid(q, e)
    
    # Spin wave parameters for RbMnF3 (approximate)
    c = 10.0 # Effective velocity
    E_0 = 1.0 # Spin gap
    
    # Theoretical dispersion relation
    dispersion = np.sqrt(E_0**2 + (c * np.sin(np.pi * Q))**2)
    
    # Simulate instrument resolution (Gaussian broadening)
    resolution_e = 0.5
    
    # Calculate intensity (Bose factor ignored for simplicity, assuming low T)
    # Intensity roughly scales with 1/E
    intensity_scale = np.where(dispersion > 0, 1.0 / dispersion, 0)
    
    # Broaden the dispersion curve
    signal = intensity_scale * np.exp(-0.5 * ((E - dispersion) / resolution_e)**2)
    
    # Add random background noise
    np.random.seed(42)
    noise = np.random.uniform(0.01, 0.1, signal.shape)
    signal += noise
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot using a logarithmic color scale to mimic real neutron data
    im = ax.pcolormesh(q, e, signal, cmap='viridis', norm=LogNorm(vmin=0.05, vmax=np.max(signal)), shading='auto')
    
    # Add labels and title
    ax.set_title("Simulated RbMnF3 2D Cut: [H, 0, 0] vs Energy")
    ax.set_xlabel("Momentum Transfer [H, 0, 0] (rlu)")
    ax.set_ylabel("Energy Transfer (meV)")
    
    fig.colorbar(im, ax=ax, label='Scattering Intensity (arb. units)')
    
    # Save the plot
    plt.savefig('rbmnf3_simulated_cut.png', dpi=200, bbox_inches='tight')
    print("Plot saved to rbmnf3_simulated_cut.png")

if __name__ == "__main__":
    main()
