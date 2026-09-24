import numpy as np
import matplotlib.pyplot as plt
from pyhorace.sqw.pixel_data import PixelDataMemory
from pyhorace.sqw.dnd import D2d
from pyhorace.sqw.sqw import SQW
from pyhorace.projection.line_proj import LineProj
from pyhorace.algorithms.cut import cut_sqw
from pyhorace.plotting.plot_2d import da
from pyhorace.crystal.oriented_lattice import OrientedLattice

def main():
    print("Generating synthetic 4D Horace dataset...")
    
    # 1. Generate some synthetic pixel data (1 million events)
    np.random.seed(42)
    n_events = 1_000_000
    
    # Random uniform Q-space
    u1 = np.random.uniform(-5, 5, n_events)
    u2 = np.random.uniform(-5, 5, n_events)
    u3 = np.random.uniform(-5, 5, n_events)
    
    # Energy transfer (dE)
    dE = np.random.uniform(0, 100, n_events)
    
    # Create a synthetic signal: A peak at Q=(1, 0, 0) and dE=50 meV
    # We use a simple Gaussian for the signal intensity
    q_dist_sq = (u1 - 1.0)**2 + (u2 - 0.0)**2 + (u3 - 0.0)**2
    e_dist_sq = (dE - 50.0)**2
    
    signal = 100.0 * np.exp(-0.5 * (q_dist_sq / 0.1 + e_dist_sq / 25.0))
    # Add some background noise
    signal += np.random.uniform(0, 5, n_events)
    variance = signal.copy() # Poisson statistics assumption
    
    # Construct the 9-column pixel array
    pix_array = np.column_stack((
        u1, u2, u3, dE, 
        np.ones(n_events),   # run_idx
        np.ones(n_events),   # detector_idx
        np.ones(n_events),   # energy_idx
        signal, variance
    ))
    
    pix = PixelDataMemory(pix_array)
    
    # 2. Create an empty D2d grid to act as the 'data' container for the full SQW
    # In a real scenario, gen_sqw would populate this
    empty_d2d = D2d(
        signal=np.array([[]]), error=np.array([[]]), npix=np.array([[]]),
        p=[np.array([-5, 5]), np.array([-5, 5])]
    )
    
    # 3. Create the SQW object
    sqw_obj = SQW(data=empty_d2d, pix=pix)
    print(f"Created SQW object with {sqw_obj.pix.num_pixels} pixels.")
    
    # 4. Define a projection
    # Cut along [H, 0, 0] and energy transfer, integrating over K and L
    proj = LineProj(u=[1, 0, 0], v=[0, 1, 0])
    
    print("Taking a 2D cut [H, 0, 0] vs Energy...")
    
    # 5. Take the cut using Numba acceleration
    # p1: [H, 0, 0] axis -> Plot from -3 to 3 with step 0.1
    # p2: [0, K, 0] axis -> Integrate from -0.5 to 0.5
    # p3: [0, 0, L] axis -> Integrate from -0.5 to 0.5
    # p4: Energy axis    -> Plot from 0 to 100 with step 2
    cut2d = cut_sqw(
        sqw_obj, 
        proj,
        p1_bin=[-3, 0.1, 3],
        p2_bin=[-0.5, 0.5],
        p3_bin=[-0.5, 0.5],
        p4_bin=[0, 2, 100]
    )
    
    cut2d.title = "Synthetic Spin Wave Peak at Q=(1,0,0), E=50 meV"
    cut2d.x_label = "Q along [H, 0, 0] (rlu)"
    cut2d.y_label = "Energy Transfer (meV)"
    
    print(f"Cut produced a {cut2d.ndim}D dataset with shape {cut2d.signal.shape}")
    
    # 6. Plot the result
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    da(cut2d, ax=ax, cmap='viridis', vmax=np.nanmax(cut2d.signal)*0.8)
    
    # Save the plot
    plt.savefig('demo_cut.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'demo_cut.png'")

if __name__ == "__main__":
    main()
