import os
from .par import read_par
from .spe import read_spe
from .nxspe import read_nxspe
from .sqw_file import SQWFileReader

def load_file(filepath: str):
    """
    Factory function to automatically detect and load supported neutron data files.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ('.par', '.phx'):
        return read_par(filepath)
    elif ext == '.spe':
        return read_spe(filepath)
    elif ext == '.nxspe':
        return read_nxspe(filepath)
    elif ext == '.sqw':
        reader = SQWFileReader(filepath)
        return {
            'main_header': reader.read_main_header(),
            'experiment_info': reader.read_experiment_info(),
            'data': reader.read_dnd_data()
        }
    else:
        # Try to read magic bytes for HDF5 (SQW v4 or NXSPE without extension)
        with open(filepath, 'rb') as f:
            magic = f.read(8)
            if magic == b'\x89HDF\x0d\x0a\x1a\x0a':
                # It's an HDF5 file, could be NXSPE or SQW
                try:
                    return read_nxspe(filepath)
                except Exception:
                    try:
                        reader = SQWFileReader(filepath)
                        return {
                            'main_header': reader.read_main_header(),
                            'experiment_info': reader.read_experiment_info(),
                            'data': reader.read_dnd_data()
                        }
                    except Exception:
                        pass

        raise ValueError(f"Unsupported file format or unknown extension: {ext}")
