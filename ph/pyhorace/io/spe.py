import numpy as np
from typing import Dict, Any

def read_spe(filepath: str) -> Dict[str, Any]:
    """
    Reads a legacy ASCII .spe file containing neutron scattering data.
    
    Returns:
        Dictionary with:
        - 'en': Energy bin boundaries (nebins + 1)
        - 'S': Signal array of shape (ndets, nebins)
        - 'ERR': Error array of shape (ndets, nebins)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Clean up empty lines and comments
    lines = [line.strip() for line in lines if line.strip() and not line.startswith(('#', '%'))]
    
    if not lines:
        raise ValueError("Empty .spe file")
        
    # First line: ndet, nebins
    header = lines[0].split()
    ndet = int(header[0])
    nebins = int(header[1])
    
    # Read all numbers
    all_vals = []
    for line in lines[1:]:
        all_vals.extend([float(x) for x in line.split()])
        
    all_vals = np.array(all_vals)
    
    # The structure of the remaining data:
    # 1. Energy bin boundaries (nebins + 1)
    # 2. For each detector:
    #    - signal (nebins)
    #    - error (nebins)
    
    expected_size = (nebins + 1) + ndet * 2 * nebins
    if len(all_vals) < expected_size:
        raise ValueError(f"Not enough data in .spe file. Expected {expected_size} values, got {len(all_vals)}")
        
    en = all_vals[:nebins + 1]
    
    # Remaining is S and ERR
    data_block = all_vals[nebins + 1 : expected_size]
    # Each detector block has size 2*nebins (signal, then error)
    data_block = data_block.reshape((ndet, 2, nebins))
    
    S = data_block[:, 0, :]
    ERR = data_block[:, 1, :]
    
    return {
        'en': en,
        'S': S,
        'ERR': ERR
    }
