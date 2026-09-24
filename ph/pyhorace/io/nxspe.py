import h5py
import numpy as np
from typing import Dict, Any

def read_nxspe(filepath: str) -> Dict[str, Any]:
    """
    Reads a NeXus NXSPE file (.nxspe) containing neutron scattering data.
    
    Returns:
        Dictionary with extracted data:
        - 'en': Energy bin boundaries
        - 'S': Signal array
        - 'ERR': Error array (computed from variance/error in file)
        - 'distance': Detector distances
        - 'theta': Scattering angles (polar)
        - 'phi': Azimuthal angles
        - 'det_width': Detector widths
        - 'det_height': Detector heights
        - 'Ei': Incident energy
        - 'psi': Goniometer angle
    """
    res = {}
    with h5py.File(filepath, 'r') as f:
        # NXSPE files usually have a root entry, e.g., 'entry', 'nxspe', etc.
        # Find the first group of type NXentry
        entry_name = None
        for key, val in f.items():
            if isinstance(val, h5py.Group) and val.attrs.get('NX_class', b'').decode('utf-8') == 'NXentry':
                entry_name = key
                break
        
        if not entry_name:
            # Fallback to default 'NXSPE' or 'entry'
            entry_name = list(f.keys())[0]
            
        entry = f[entry_name]
        
        # NXdata group
        data_group = None
        for key, val in entry.items():
            if isinstance(val, h5py.Group) and val.attrs.get('NX_class', b'').decode('utf-8') == 'NXdata':
                data_group = val
                break
                
        if not data_group:
            if 'data' in entry:
                data_group = entry['data']
            elif 'NXSPE_info' in entry:
                data_group = entry['NXSPE_info']
            else:
                raise ValueError("Could not find NXdata group in NXSPE file")

        # Read Signal and Error
        if 'data' in data_group:
            res['S'] = data_group['data'][()]
        
        if 'error' in data_group:
            res['ERR'] = data_group['error'][()]
        elif 'errors' in data_group:
            res['ERR'] = data_group['errors'][()]
        elif 'variance' in data_group:
            res['ERR'] = np.sqrt(data_group['variance'][()])
            
        # Read energy bins
        if 'energy' in data_group:
            res['en'] = data_group['energy'][()]
        elif 'en' in data_group:
            res['en'] = data_group['en'][()]
            
        # Instrument / Detector
        instr = entry.get('instrument', entry.get('NXSPE_info'))
        if instr:
            # Look for NXdetector
            for key, val in instr.items():
                if isinstance(val, h5py.Group) and val.attrs.get('NX_class', b'').decode('utf-8') == 'NXdetector':
                    det = val
                    if 'distance' in det: res['distance'] = det['distance'][()]
                    if 'polar_angle' in det: res['theta'] = det['polar_angle'][()]
                    if 'azimuthal_angle' in det: res['phi'] = det['azimuthal_angle'][()]
                    break
                    
        # Sample / Goniometer
        sample = entry.get('sample')
        if sample:
            if 'rotation_angle' in sample:
                res['psi'] = float(sample['rotation_angle'][()])
                
        # Incident energy
        if 'fermi' in instr: # commonly under chopper
            if 'energy' in instr['fermi']:
                res['Ei'] = float(instr['fermi']['energy'][()])
        elif 'Ei' in entry:
            res['Ei'] = float(entry['Ei'][()])

    return res
