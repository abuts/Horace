import h5py
import numpy as np
from typing import Dict, Any, List

class SQWFileReader:
    """
    Reader for Horace v4 HDF5 .sqw files.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def read_main_header(self) -> Dict[str, Any]:
        """Reads the main_header group."""
        res = {}
        with h5py.File(self.filepath, 'r') as f:
            if 'main_header' in f:
                mh = f['main_header']
                for k in mh.keys():
                    res[k] = mh[k][()]
        return res
        
    def read_experiment_info(self) -> List[Dict[str, Any]]:
        """Reads experiment metadata for each run."""
        runs = []
        with h5py.File(self.filepath, 'r') as f:
            if 'experiment_info' in f:
                ei = f['experiment_info']
                # Usually a cell array or group per run
                # Simplified representation:
                for k in sorted(ei.keys()):
                    run_group = ei[k]
                    run_info = {}
                    if isinstance(run_group, h5py.Group):
                        for sub_k in run_group.keys():
                            run_info[sub_k] = run_group[sub_k][()]
                    runs.append(run_info)
        return runs

    def read_dnd_data(self) -> Dict[str, np.ndarray]:
        """Reads the binned grid data (signal, err, npix, axes)."""
        res = {}
        with h5py.File(self.filepath, 'r') as f:
            if 'data' in f:
                d = f['data']
                res['s'] = d.get('s', d.get('signal', np.array([])))[()]
                res['e'] = d.get('e', d.get('error', np.array([])))[()]
                res['npix'] = d.get('npix', np.array([]))[()]
                
                # Bin boundaries
                res['p'] = []
                p_group = d.get('p')
                if isinstance(p_group, h5py.Group):
                    for i in range(len(p_group.keys())):
                        res['p'].append(p_group[f'p{i+1}'][()])
                elif p_group is not None:
                    # Might be an array of arrays if 1D
                    res['p'] = [p_group[()]]
        return res

    def read_pixels(self, start: int = 0, count: int = -1) -> np.ndarray:
        """Reads a chunk of pixel data from the file."""
        with h5py.File(self.filepath, 'r') as f:
            if 'pix' in f:
                pix = f['pix']
                if count == -1:
                    return pix[start:]
                else:
                    return pix[start:start+count]
            return np.array([])
