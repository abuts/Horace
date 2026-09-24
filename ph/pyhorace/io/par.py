import numpy as np
import pandas as pd
from typing import Dict, Tuple

def read_par(filepath: str) -> Dict[str, np.ndarray]:
    """
    Reads an ASCII .par or .phx detector parameter file.
    
    Expected format (columns):
    1. Distance (L2) from sample to detector (m)
    2. Scattering angle (2theta) (deg)
    3. Azimuthal angle (phi) (deg)
    4. Detector width (m)
    5. Detector height (m)
    6. Detector ID (optional)
    
    Args:
        filepath: Path to the .par/.phx file
        
    Returns:
        Dictionary containing detector arrays: 'distance', 'theta', 'phi', 'width', 'height', 'det_id'
    """
    # Try using numpy to loadtxt
    # Ignore comments starting with # or %
    try:
        data = np.loadtxt(filepath, comments=['#', '%'])
    except ValueError:
        # Fallback to pandas if there are irregular lines
        import pandas as pd
        df = pd.read_csv(filepath, comment='#', delim_whitespace=True, header=None)
        data = df.values

    # Check number of detectors (rows)
    # The first line might be the number of detectors in some formats
    if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1 and data.shape[0] > 1):
        if data.size > 0 and len(data) % 5 == 0 or len(data) % 6 == 0:
            pass # We could reshape, but typically it's tabular
            
    # Remove first row if it's just the detector count
    if data.ndim == 2 and data.shape[1] in (5, 6) and data[0, 0] == data.shape[0] - 1:
        data = data[1:, :]
    elif data.ndim == 1 and data[0] * 5 == len(data) - 1:
        n_det = int(data[0])
        data = data[1:].reshape(n_det, 5)
    elif data.ndim == 1 and data[0] * 6 == len(data) - 1:
        n_det = int(data[0])
        data = data[1:].reshape(n_det, 6)

    if data.ndim != 2:
        raise ValueError(f"Could not parse .par file into a 2D array. Shape: {data.shape}")

    n_cols = data.shape[1]
    n_det = data.shape[0]

    res = {
        'distance': data[:, 0],
        'theta': data[:, 1],
        'phi': data[:, 2],
        'width': data[:, 3],
        'height': data[:, 4],
        'det_id': np.arange(1, n_det + 1)
    }

    if n_cols >= 6:
        res['det_id'] = data[:, 5].astype(int)

    return res
