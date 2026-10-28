"""Coordinate transformation, projection, and axes block classes for PyHorace."""

from pyhorace.coord_transf._dependencies import (
    HoraceError,
    crystal_alignment_info,
    fast_map,
    oriented_lattice,
    rotmat_to_rotvec,
    rotmat_to_rotvec_rad,
    rotvec_to_rotmat,
    rotvec_to_rotmat_rad,
)
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.axes_block_base import AxesBlockBase
from pyhorace.coord_transf.bmatrix import bmatrix
from pyhorace.coord_transf.calc_proj_matrix import calc_proj_matrix
from pyhorace.coord_transf.curve_proj_base import CurveProjBase
from pyhorace.coord_transf.cylinder_axes import cylinder_axes
from pyhorace.coord_transf.cylinder_proj import cylinder_proj
from pyhorace.coord_transf.expand_box import expand_box, get_geometry
from pyhorace.coord_transf.fudge_proj import fudge_proj
from pyhorace.coord_transf.instr_proj import instr_proj
from pyhorace.coord_transf.kf_sphere_proj import kf_sphere_proj
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.coord_transf.line_proj_base import LineProjBase
from pyhorace.coord_transf.sphere_axes import sphere_axes
from pyhorace.coord_transf.sphere_proj import sphere_proj
from pyhorace.coord_transf.symop import (
    Symop,
    SymopArray,
    SymopGeneral,
    SymopIdentity,
    SymopReflection,
    SymopRotation,
    SymopSetPlaneInterface,
    symop,
)
from pyhorace.coord_transf.ubmat_proj import ubmat_proj
from pyhorace.coord_transf.ubmatrix import ubmatrix

__all__ = [
    "AxesBlockBase",
    "CurveProjBase",
    "DnDBase",
    "HoraceError",
    "LineProjBase",
    "PixelDataBase",
    "PixelDataMemory",
    "Symop",
    "SymopIdentity",
    "SymopReflection",
    "SymopRotation",
    "bmatrix",
    "calc_proj_matrix",
    "crystal_alignment_info",
    "cylinder_axes",
    "cylinder_proj",
    "d0d",
    "d1d",
    "d2d",
    "d3d",
    "d4d",
    "expand_box",
    "fast_map",
    "fudge_proj",
    "get_geometry",
    "instr_proj",
    "kf_sphere_proj",
    "line_axes",
    "line_proj",
    "oriented_lattice",
    "rotmat_to_rotvec",
    "rotmat_to_rotvec_rad",
    "rotvec_to_rotmat",
    "rotvec_to_rotmat_rad",
    "sphere_axes",
    "sphere_proj",
    "symop",
    "ubmat_proj",
    "ubmatrix",
]


def __getattr__(name: str):
    if name in (
        "PixelDataBase",
        "PixelDataMemory",
        "PixelDataFileBacked",
        "PixelData",
    ):
        import importlib

        _pix = importlib.import_module("pyhorace.data.pixels")
        return getattr(_pix, name)
    if name in ("DnDBase", "SQWDnDBase", "d0d", "d1d", "d2d", "d3d", "d4d", "dnd", "dnd_metadata", "dnd_data", "data_sqw_dnd"):
        import importlib

        _sqw = importlib.import_module("pyhorace.sqw")
        return getattr(_sqw, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

