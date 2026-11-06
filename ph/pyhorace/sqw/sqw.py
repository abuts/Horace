"""Horace SQW (4D scattering function + pixel data) class.

Translated from ``horace_core/sqw/@sqw/sqw.m`` (properties, constructor, and destructor;
other methods stubbed for subsequent conversion).
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.serializable import Serializable, register_serializable


def warn_with_id(warn_id: str, message: str) -> None:
    """Issue a UserWarning prefixed with a MATLAB-style warning identifier."""
    warnings.warn(f"{warn_id}: {message}", UserWarning, stacklevel=2)
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data.experiment import Experiment
from pyhorace.data.instr_classes.ix_detector_array import IX_detector_array
from pyhorace.data.instr_classes.ix_experiment import IX_experiment
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.data.sigvar import sigvar
from pyhorace.io.main_header_cl import main_header_cl
from pyhorace.sqw.d0d import d0d
from pyhorace.sqw.d1d import d1d
from pyhorace.sqw.d2d import d2d
from pyhorace.sqw.d3d import d3d
from pyhorace.sqw.d4d import d4d
from pyhorace.sqw.data_sqw_dnd import data_sqw_dnd
from pyhorace.sqw.dnd_base import DnDBase
from pyhorace.sqw.sqw_dnd_base import SQWDnDBase
from pyhorace.sqw.sqw_plot_interface import sqw_plot_interface


class _CallableBool(int):
    """Boolean-like value that can be inspected as a property or called as a method."""

    def __new__(cls, val: bool) -> "_CallableBool":
        return super().__new__(cls, int(bool(val)))

    def __call__(self) -> bool:
        return bool(int(self))

    def __bool__(self) -> bool:
        return bool(int(self))

    def __repr__(self) -> str:
        return repr(bool(int(self)))

    def __str__(self) -> str:
        return str(bool(int(self)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (bool, np.bool_, int, np.integer)):
            return bool(int(self)) == bool(other)
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(bool(int(self)))


@register_serializable
class sqw(SQWDnDBase, sqw_plot_interface):
    """Horace 4D SQW dataset combining main header, experiment metadata, DnD grid data, and pixels.

    Translated from ``horace_core/sqw/@sqw/sqw.m``.
    Note: The MATLAB ``experiment_info`` property has been renamed to ``exper_info``
    (with ``experiment_info`` retained as a compatibility alias).
    Handle-simulation via ``tmp_file_holder_`` in MATLAB is replaced with Python's
    standard destructor (``__del__``).
    """

    _fields_to_save: ClassVar[Tuple[str, ...]] = (
        "main_header",
        "exper_info",
        "data",
        "pix",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) == 0 and len(kwargs) == 0:
            self._main_header: main_header_cl = main_header_cl()
            self._exper_info: Experiment = Experiment()
            self._data: DnDBase = d0d()
            self._pix: PixelDataBase = PixelDataBase.create()
            self._is_tmp_obj: bool = False
            self._tmp_file_path: str = ""
            return

        self.init(*args, **kwargs)

    # ------------------------------------------------------------------
    # Constructor & Initialization (init_.m, parse_sqw_args_.m, init_from_file_.m, make_sqw.m)
    # ------------------------------------------------------------------
    def init(self, *args: Any, **kwargs: Any) -> "sqw":
        """Initialize or re-initialize the ``sqw`` object (port of ``@sqw/private/init_.m``)."""
        if len(args) == 0 and len(kwargs) == 0:
            return self

        parsed = self.parse_sqw_args(*args, **kwargs)

        if parsed["sqw_obj"] is not None:
            src: sqw = parsed["sqw_obj"]
            self._copy_from(src)
            return self

        fb_opt: Optional[bool] = (
            parsed["file_backed"] if parsed["force_pix_location"] else None
        )

        if parsed["file"] is not None:
            return self.init_from_file(
                parsed["file"],
                pixel_page_size=parsed["pixel_page_size"],
                file_backed=fb_opt,
                force_mex=parsed["force_mex"],
            )

        if parsed["data_struct"] is not None:
            ds = parsed["data_struct"]
            if isinstance(ds, DnDBase):
                self._main_header = main_header_cl()
                self._exper_info = Experiment()
                self._data = ds.copy()
                self._pix = PixelDataMemory()
                return self

            if isinstance(ds, data_sqw_dnd):
                self._main_header = main_header_cl()
                self._exper_info = Experiment()
                self._data = DnDBase.dnd(ds)
                self._pix = PixelDataMemory()
                return self

            if hasattr(ds, "get_sqw") or hasattr(ds, "get_data") or hasattr(ds, "get_main_header"):
                return self._init_from_loader(
                    ds,
                    pixel_page_size=parsed["pixel_page_size"],
                    file_backed=fb_opt,
                )

            if isinstance(ds, dict):
                return self._init_from_dict(
                    ds,
                    pixel_page_size=parsed["pixel_page_size"],
                    file_backed=fb_opt,
                )

        raise HoraceError(
            "HORACE:sqw:invalid_argument",
            "Unrecognised constructor arguments for sqw object",
        )

    @staticmethod
    def parse_sqw_args(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Parse arguments passed to the ``sqw`` constructor (port of ``@sqw/private/parse_sqw_args_.m``)."""
        pos_args = list(args)
        kw = dict(kwargs)

        # Extract trailing MATLAB-style key-value pairs when first arg is a filename, loader, or dict
        if len(pos_args) >= 1 and isinstance(
            pos_args[0], (str, os.PathLike, dict)
        ) or (
            len(pos_args) >= 1
            and (
                hasattr(pos_args[0], "get_sqw")
                or hasattr(pos_args[0], "get_data")
                or hasattr(pos_args[0], "get_main_header")
            )
        ):
            rest = pos_args[1:]
            idx = 0
            while idx < len(rest):
                key = rest[idx]
                if isinstance(key, str):
                    k_clean = key.lower().lstrip("-")
                    if k_clean in (
                        "pixel_page_size",
                        "file_backed",
                        "filebacked",
                        "force_mex",
                    ):
                        if idx + 1 < len(rest):
                            kw[k_clean] = rest[idx + 1]
                            idx += 2
                            continue
                        else:
                            kw[k_clean] = True
                            idx += 1
                            continue
                break
            if idx == len(rest):
                pos_args = [pos_args[0]]

        # Check if positional/keyword arguments correspond to saveable fields
        save_fields = ("main_header", "exper_info", "experiment_info", "data", "pix", "detpar")
        if (len(pos_args) >= 2 and not isinstance(pos_args[0], (str, os.PathLike))) or any(
            k in kw for k in save_fields
        ):
            d_struct: Dict[str, Any] = {}
            field_order = ["main_header", "exper_info", "data", "pix"]
            for i, val in enumerate(pos_args):
                if i < len(field_order):
                    d_struct[field_order[i]] = val
            for k in list(kw.keys()):
                if k in save_fields:
                    d_struct[k] = kw.pop(k)
            pos_args = [d_struct]

        input_val = pos_args[0] if len(pos_args) > 0 else None

        force_pix_location = (
            "file_backed" in kw or "filebacked" in kw or "pixel_page_size" in kw
        )
        raw_fb = kw.pop("file_backed", kw.pop("filebacked", None))
        pixel_page_size = kw.pop("pixel_page_size", None)
        force_mex = kw.pop("force_mex", None)

        if len(kw) > 0:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Unrecognized init parameter(s): {list(kw.keys())}",
            )

        if pixel_page_size is not None:
            if (
                not isinstance(pixel_page_size, (int, float, np.number))
                or isinstance(pixel_page_size, bool)
                or pixel_page_size <= 0
            ):
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"pixel_page_size must be a positive scalar, got {pixel_page_size!r}",
                )
            pixel_page_size = int(pixel_page_size)
            if raw_fb is None and pixel_page_size < PixelDataBase.DEFAULT_PAGE_SIZE:
                raw_fb = True
        else:
            pixel_page_size = PixelDataBase.DEFAULT_PAGE_SIZE

        if raw_fb is not None:
            if not isinstance(raw_fb, (bool, np.bool_, int, float, np.number)):
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"file_backed must be logical or numeric, got {type(raw_fb).__name__}",
                )
            file_backed = bool(raw_fb)
        else:
            file_backed = False

        parsed: Dict[str, Any] = {
            "sqw_obj": None,
            "file": None,
            "filename": None,
            "data_struct": None,
            "pixel_page_size": pixel_page_size,
            "file_backed": file_backed,
            "force_pix_location": force_pix_location,
            "force_mex": force_mex,
        }

        if isinstance(input_val, sqw):
            parsed["sqw_obj"] = input_val
        elif isinstance(input_val, (str, os.PathLike)):
            parsed["file"] = str(input_val)
            parsed["filename"] = str(input_val)
        elif (
            hasattr(input_val, "get_sqw")
            or hasattr(input_val, "get_data")
            or hasattr(input_val, "get_main_header")
        ):
            parsed["file"] = input_val
            parsed["filename"] = input_val
        elif isinstance(input_val, (DnDBase, data_sqw_dnd, dict)):
            parsed["data_struct"] = input_val
        elif input_val is None:
            parsed["data_struct"] = sqw.make_sqw(False)
        else:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Cannot construct sqw from input of type {type(input_val).__name__}",
            )

        return parsed

    def init_from_file(
        self,
        filename: Any,
        pixel_page_size: Optional[int] = None,
        file_backed: Optional[bool] = None,
        force_mex: Optional[bool] = None,
    ) -> "sqw":
        """Populate ``sqw`` from a binary ``.sqw`` or ``.mat`` file, loader, or ``in_struc`` dict
        (port of ``@sqw/private/init_sqw_from_file_.m``).
        """
        if isinstance(filename, dict):
            in_struc = filename
            if "file" not in in_struc or in_struc["file"] is None or in_struc["file"] == "":
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    "init_from_file accepts only structure with field 'file' containing filename or initialized data loader class",
                )
            target = in_struc["file"]
            force_pix = bool(in_struc.get("force_pix_location", "file_backed" in in_struc))
            fb = bool(in_struc.get("file_backed", False)) if force_pix else None
            pg_sz = in_struc.get("pixel_page_size", pixel_page_size)
            return self.init_from_file(
                target,
                pixel_page_size=pg_sz,
                file_backed=fb,
                force_mex=force_mex,
            )

        if (
            hasattr(filename, "get_sqw")
            or hasattr(filename, "get_data")
            or hasattr(filename, "get_main_header")
        ):
            return self._init_from_loader(
                filename,
                pixel_page_size=pixel_page_size,
                file_backed=file_backed,
            )

        if not isinstance(filename, (str, os.PathLike)) or not str(filename):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "File name for sqw initialization must be a non-empty string",
            )
        path = str(filename)
        if not os.path.isfile(path):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"File does not exist: {path}",
            )

        ldr = None
        try:
            from pyhorace.io.sqw_formats_factory import sqw_formats_factory

            ldr = sqw_formats_factory.instance().get_loader(path)
        except Exception:
            ldr = None

        if ldr is not None:
            try:
                if not bool(ldr.sqw_type):
                    raise HoraceError(
                        "HORACE:sqw:invalid_argument",
                        f"Data file '{path}' does not contain an sqw-type object (contains dnd-type only)",
                    )
                return self._init_from_loader(
                    ldr,
                    pixel_page_size=pixel_page_size,
                    file_backed=file_backed,
                )
            finally:
                ldr.fclose()

        # Try loading as a MATLAB .mat file
        import scipy.io

        try:
            mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as exc:
            raise HoraceError(
                "HORACE:sqw:io_error",
                f"Could not read sqw file '{path}': {exc}",
            ) from exc

        for k, v in mat.items():
            if not k.startswith("__"):
                loaded = sqw.loadobj(v)
                if isinstance(loaded, sqw):
                    self._copy_from(loaded)
                    return self

        raise HoraceError(
            "HORACE:sqw:invalid_argument",
            f"No valid sqw object found in '{path}'",
        )

    def _init_from_loader(
        self,
        loader: Any,
        pixel_page_size: Optional[int] = None,
        file_backed: Optional[bool] = None,
    ) -> "sqw":
        """Initialize ``sqw`` from a ``horace_binfile_interface``-compatible loader object."""
        if hasattr(loader, "sqw_type") and not bool(getattr(loader, "sqw_type")):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Data file does not contain full sqw-type object",
            )

        if hasattr(loader, "get_sqw") and file_backed is None and pixel_page_size is None:
            res = loader.get_sqw()
            if isinstance(res, sqw):
                self._copy_from(res)
                self._trim_old_file_exper_info()
                self._main_header.nfiles = self._exper_info.n_runs
                return self
            if isinstance(res, dict):
                return self._init_from_dict(res, pixel_page_size=pixel_page_size, file_backed=file_backed)

        mh = loader.get_main_header() if hasattr(loader, "get_main_header") else main_header_cl()
        exp_info = loader.get_exp_info("-all") if hasattr(loader, "get_exp_info") else Experiment()
        detpar = loader.get_detpar() if hasattr(loader, "get_detpar") else None
        if detpar is not None and exp_info.n_runs > 0 and exp_info.detector_arrays.n_runs == 0:
            if isinstance(detpar, dict):
                det_arr = IX_detector_array(detpar)
                urc = unique_references_container("IX_detector_array")
                urc.add(det_arr)
                if exp_info.n_runs > 1:
                    urc = urc.replicate_runs(exp_info.n_runs)
                exp_info.detector_arrays = urc
            elif isinstance(detpar, IX_detector_array):
                urc = unique_references_container("IX_detector_array")
                urc.add(detpar)
                if exp_info.n_runs > 1:
                    urc = urc.replicate_runs(exp_info.n_runs)
                exp_info.detector_arrays = urc
            elif isinstance(detpar, unique_references_container):
                exp_info.detector_arrays = detpar

        dnd_data_obj = loader.get_dnd() if hasattr(loader, "get_dnd") else (
            loader.get_data() if hasattr(loader, "get_data") else d0d()
        )
        if not isinstance(dnd_data_obj, DnDBase):
            dnd_data_obj = DnDBase.dnd(dnd_data_obj)

        al_info = None
        if hasattr(loader, "_dnd_data_alignment") and hasattr(exp_info, "header_average"):
            al_info = loader._dnd_data_alignment(dnd_data_obj, exp_info.header_average())

        pix_obj = PixelDataBase.create(loader, filebacked=file_backed)
        if al_info is not None and hasattr(al_info, "rotmat") and pix_obj.num_pixels > 0:
            pix_obj.alignment_matr = al_info.rotmat
        if not mh.creation_date_defined:
            pix_obj.old_file_format = True
            if isinstance(pix_obj, PixelDataMemory) and pix_obj.num_pixels > 0:
                pix_obj._unique_run_id = np.unique(pix_obj.run_idx).astype(float)
        if (
            pixel_page_size is not None
            and pixel_page_size < PixelDataBase.DEFAULT_PAGE_SIZE
            and isinstance(pix_obj, PixelDataFileBacked)
        ):
            pix_obj._base_page_size = int(pixel_page_size)

        self._main_header = (
            mh
            if isinstance(mh, main_header_cl) or type(mh).__name__ == "main_header_cl"
            else main_header_cl(mh)
        )
        self._exper_info = (
            exp_info
            if isinstance(exp_info, Experiment)
            or (type(exp_info).__name__ == "Experiment" and hasattr(exp_info, "_expdata"))
            else Experiment(exp_info)
        )
        self._data = dnd_data_obj
        self._pix = pix_obj
        self._trim_old_file_exper_info()
        if (
            not self._main_header.creation_date_defined
            and self._pix.num_pixels > 0
            and self._exper_info.n_runs > 0
        ):
            from pyhorace.coord_transf._dependencies import fast_map

            u_ids = np.unique(self._pix.run_idx).astype(float)
            if u_ids.size == self._exper_info.n_runs:
                cur_keys = np.asarray(self._exper_info.runid_map.keys, dtype=float)
                if not np.array_equal(cur_keys, u_ids):
                    self._exper_info.runid_map = fast_map(
                        u_ids, np.arange(1, self._exper_info.n_runs + 1, dtype=int)
                    )
                    self._exper_info.runid_recalculated = False
        self._main_header.nfiles = self._exper_info.n_runs
        return self

    def _trim_old_file_exper_info(self) -> None:
        """Keep only experiments contributing into pixels for old file formats (port of ``init_from_loader_struct_.m``)."""
        if (
            self._pix is not None
            and self._pix.num_pixels > 0
            and getattr(self._pix, "old_file_format", False)
            and getattr(self._pix, "unique_run_id", np.array([])).size > 0
        ):
            self.exper_info = self._exper_info.get_subobj(self._pix.unique_run_id)
            self._pix.old_file_format = False

    def _init_from_dict(
        self,
        S: Dict[str, Any],
        pixel_page_size: Optional[int] = None,
        file_backed: Optional[bool] = None,
    ) -> "sqw":
        """Initialize ``sqw`` from a dictionary or legacy structure."""
        if "serial_name" in S or "version" in S:
            loaded = sqw.loadobj(S)
            if isinstance(loaded, sqw):
                self._copy_from(loaded)
                return self

        d = dict(S)
        # Convert legacy 'header' field to 'exper_info'
        if "header" in d and "exper_info" not in d and "experiment_info" not in d:
            hdr = d.pop("header")
            if hdr is not None and not (isinstance(hdr, (list, tuple, dict)) and len(hdr) == 0):
                d["exper_info"] = Experiment.build_from_binfile_headers(hdr)
            else:
                d["exper_info"] = Experiment()

        if "experiment_info" in d:
            exp_info_val = d.pop("experiment_info")
            if (
                "exper_info" not in d
                or d["exper_info"] is None
                or (isinstance(d["exper_info"], Experiment) and d["exper_info"].n_runs == 0)
            ):
                d["exper_info"] = exp_info_val

        detpar = d.pop("detpar", None)

        # Extract pix if nested inside data dict/object
        if "data" in d and isinstance(d["data"], dict) and "pix" in d["data"] and "pix" not in d:
            d_data = dict(d["data"])
            d["pix"] = d_data.pop("pix")
            d["data"] = d_data

        self._do_check_combo_arg = False
        try:
            if "main_header" in d:
                self.main_header = d["main_header"]
            if "exper_info" in d:
                self.exper_info = d["exper_info"]
            if detpar is not None and not (isinstance(detpar, (list, tuple, dict)) and len(detpar) == 0) and self._exper_info.detector_arrays.n_runs == 0:
                self.detpar = detpar
            if "data" in d:
                raw_data = d["data"]
                if isinstance(raw_data, DnDBase):
                    self.data = raw_data
                elif isinstance(raw_data, data_sqw_dnd):
                    self.data = DnDBase.dnd(raw_data)
                elif isinstance(raw_data, dict):
                    if "serial_name" in raw_data:
                        self.data = Serializable.loadobj(raw_data)
                    else:
                        self.data = DnDBase.dnd(data_sqw_dnd(raw_data))
                else:
                    self.data = raw_data
            if "pix" in d:
                raw_pix = d["pix"]
                from pyhorace.data.pixels.multipix_base import multipix_base
                if isinstance(raw_pix, multipix_base):
                    self.pix = raw_pix
                elif isinstance(raw_pix, PixelDataBase):
                    if file_backed is not None and raw_pix.is_filebacked != file_backed:
                        self.pix = PixelDataBase.create(raw_pix, filebacked=file_backed)
                    else:
                        self.pix = raw_pix
                else:
                    self.pix = PixelDataBase.create(raw_pix, filebacked=file_backed)
                if (
                    pixel_page_size is not None
                    and pixel_page_size < PixelDataBase.DEFAULT_PAGE_SIZE
                    and isinstance(self._pix, PixelDataFileBacked)
                ):
                    self._pix._base_page_size = int(pixel_page_size)
        finally:
            self._do_check_combo_arg = True

        self._trim_old_file_exper_info()
        self.check_combo_arg()
        return self

    @staticmethod
    def head_form(full: bool = False) -> List[str]:
        """Return list of field names for ``head`` summary (port of ``@sqw/private/head_form_.m``)."""
        if full:
            return [
                "filename",
                "filepath",
                "title",
                "alatt",
                "angdeg",
                "offset",
                "u_to_rlu",
                "ulen",
                "label",
                "iax",
                "iint",
                "pax",
                "p",
                "dax",
                "s",
                "e",
                "npix",
                "img_range",
                "npixels",
                "pix_range",
                "num_contrib_files",
                "data_range",
                "creation_date",
                "experiment_info",
                "detpar",
            ]
        return [
            "filename",
            "filepath",
            "title",
            "alatt",
            "angdeg",
            "offset",
            "u_to_rlu",
            "ulen",
            "label",
            "iax",
            "iint",
            "pax",
            "p",
            "dax",
            "img_range",
            "npixels",
            "num_contrib_files",
            "creation_date",
        ]

    @staticmethod
    def make_sqw(dnd_type: bool = False, *args: Any) -> Dict[str, Any]:
        """Create default ``sqw`` dictionary skeleton (port of ``@sqw/private/make_sqw.m``)."""
        if dnd_type:
            data_obj = d0d(*args) if len(args) == 0 else DnDBase.dnd(*args)
            return {
                "main_header": main_header_cl(),
                "exper_info": Experiment(),
                "data": data_obj,
                "pix": PixelDataMemory(),
            }
        data_obj = d0d() if len(args) == 0 else DnDBase.dnd(*args)
        return {
            "main_header": main_header_cl(),
            "exper_info": Experiment(),
            "data": data_obj,
            "pix": PixelDataMemory(),
        }

    def _copy_from(self, other: "sqw", exclude_pix: bool = False) -> None:
        """Internal deep-copy helper."""
        self._main_header = copy.deepcopy(other._main_header)
        self._exper_info = copy.deepcopy(other._exper_info)
        self._data = other._data.copy()
        if exclude_pix:
            self._pix = PixelDataMemory()
        else:
            self._pix = copy.deepcopy(other._pix)
        other_holder = getattr(other, "_tmp_file_holder", None)
        if not exclude_pix and other_holder is not None:
            self._tmp_file_holder = other_holder.acquire()
            self._is_tmp_obj = True
            self._tmp_file_path = other._tmp_file_path
            return
        self._tmp_file_holder = None
        self._is_tmp_obj = False
        self._tmp_file_path = ""

    def copy(self, *args: Any, exclude_pix: bool = False) -> "sqw":
        """Clone this ``sqw`` object, optionally excluding pixel data (port of ``@sqw/copy.m``)."""
        for a in args:
            if isinstance(a, str) and a.lower().lstrip("-") == "exclude_pix":
                exclude_pix = True
            elif isinstance(a, bool):
                exclude_pix = a

        new_obj = sqw()
        new_obj._copy_from(self, exclude_pix=exclude_pix)
        return new_obj

    def __copy__(self) -> "sqw":
        return self.copy()

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> "sqw":
        return self.copy()

    # ------------------------------------------------------------------
    # Destructor & Temporary File Lifecycle (delete.m, set_as_tmp_obj.m, activate.m, deactivate.m)
    # ------------------------------------------------------------------
    def set_as_tmp_obj(self, filename: Optional[str] = None) -> "sqw":
        """Mark this ``sqw`` object as a temporary file-backed object whose backing file
        will be deleted when the object is garbage-collected or ``delete()`` is called.

        Translated from ``@sqw/set_as_tmp_obj.m``, replacing ``tmp_file_holder_`` with
        Python's ``__del__`` destructor.
        """
        if not self.is_filebacked and filename is None:
            return self
        target = str(filename) if filename is not None else self._pix.full_filename
        if target:
            from pyhorace.data.pixels.pixel_data_file_backed import _TmpFileHolder

            cur_holder = getattr(self, "_tmp_file_holder", None)
            if cur_holder is None or cur_holder.file_name != target:
                if cur_holder is not None:
                    cur_holder.release()
                self._tmp_file_holder = _TmpFileHolder(target)
            self._is_tmp_obj = True
            self._tmp_file_path = target
        return self

    def deactivate(self) -> "sqw":
        """Close open file-backed pixel handles (port of ``@sqw/deactivate.m``)."""
        if self._pix is not None:
            self._pix.deactivate()
        cur_holder = getattr(self, "_tmp_file_holder", None)
        if cur_holder is not None:
            if cur_holder.ref_count > 1:
                cur_holder.ref_count -= 1
            self._tmp_file_holder = None
        self._is_tmp_obj = False
        self._tmp_file_path = ""
        return self

    def activate(self, new_file: Optional[str] = None) -> "sqw":
        """Re-open file-backed pixel handles (port of ``@sqw/activate.m``)."""
        if not self.is_filebacked and new_file is None:
            return self
        target = str(new_file) if new_file is not None else (self._pix.full_filename or self.full_filename)
        if self._pix is not None and target and self._pix.is_filebacked:
            try:
                self._pix.activate(target, no_tmp_file=True)
            except TypeError:
                self._pix.activate(target)
        if target:
            _, ext = os.path.splitext(target)
            if ext.lower() == ".tmp" or ext.lower().startswith(".tmp_"):
                self.set_as_tmp_obj(target)
            else:
                self.full_filename = self._pix.full_filename if (self._pix is not None and self._pix.full_filename) else target
        return self

    def delete(self) -> None:
        """Clean up temporary backing file if ``is_tmp_obj`` is True (port of ``@sqw/delete.m``)."""
        cur_holder = getattr(self, "_tmp_file_holder", None)
        if cur_holder is not None:
            if getattr(self, "_pix", None) is not None:
                try:
                    self._pix.deactivate()
                except Exception:
                    pass
            self._tmp_file_holder = None
            self._is_tmp_obj = False
            self._tmp_file_path = ""
            cur_holder.release()
            return
        if getattr(self, "_is_tmp_obj", False):
            tmp_path = getattr(self, "_tmp_file_path", "") or (
                self._pix.full_filename if getattr(self, "_pix", None) is not None else ""
            )
            if getattr(self, "_pix", None) is not None:
                try:
                    self._pix.deactivate()
                except Exception:
                    pass
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            self._is_tmp_obj = False
            self._tmp_file_path = ""
            return

    def __del__(self) -> None:
        """Standard Python destructor cleaning up any temporary backing file."""
        try:
            self.delete()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Properties (@sqw/sqw.m)
    # ------------------------------------------------------------------
    @property
    def main_header(self) -> main_header_cl:
        """Main header metadata (:class:`~pyhorace.io.main_header_cl`)."""
        return self._main_header

    @main_header.setter
    def main_header(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._main_header = main_header_cl()
        elif isinstance(val, main_header_cl) or (
            type(val).__name__ == "main_header_cl" and hasattr(val, "_filename")
        ):
            self._main_header = copy.deepcopy(val)
        elif isinstance(val, dict):
            if "serial_name" in val:
                loaded = Serializable.loadobj(val)
                if not (
                    isinstance(loaded, main_header_cl)
                    or type(loaded).__name__ == "main_header_cl"
                ):
                    raise HoraceError(
                        "HORACE:sqw:invalid_argument",
                        f"Main header has to be an instance of main_header_cl class or struct, got {type(loaded).__name__}",
                    )
                self._main_header = loaded
            else:
                self._main_header = main_header_cl(val)
        else:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Main header has to be an instance of main_header_cl class or struct, being in fact {type(val).__name__}",
            )

        if (
            self._do_check_combo_arg
            and self._exper_info.n_runs > 0
            and self._exper_info.n_runs != self._main_header.nfiles
        ):
            warn_with_id(
                "HORACE:inconsitent_sqw",
                f"Number of headers consistent with exper_info ({self._exper_info.n_runs}) "
                f"is not equal to nfiles in main_header ({self._main_header.nfiles})",
            )

    @property
    def exper_info(self) -> Experiment:
        """Experiment metadata (:class:`~pyhorace.data.experiment.Experiment`), renamed from ``experiment_info``."""
        return self._exper_info

    @exper_info.setter
    def exper_info(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._exper_info = Experiment()
        elif isinstance(val, Experiment) or (
            type(val).__name__ == "Experiment" and hasattr(val, "_expdata")
        ):
            self._exper_info = val
        elif (
            isinstance(val, tuple)
            and len(val) > 0
            and (
                isinstance(val[0], Experiment)
                or (type(val[0]).__name__ == "Experiment" and hasattr(val[0], "_expdata"))
            )
        ):
            val = val[0]
            self._exper_info = val
        elif isinstance(val, IX_experiment):
            self._exper_info = Experiment(val)
        elif isinstance(val, dict):
            if "serial_name" in val:
                loaded = Serializable.loadobj(val)
                if not (
                    isinstance(loaded, Experiment)
                    or type(loaded).__name__ == "Experiment"
                ):
                    raise HoraceError(
                        "HORACE:sqw:invalid_argument",
                        f"Experiment info can be only an instance of Experiment class, input is {type(loaded).__name__}",
                    )
                self._exper_info = loaded
            else:
                self._exper_info = Experiment(val)
        elif isinstance(val, (list, tuple)) and all(isinstance(x, (IX_experiment, dict)) for x in val):
            self._exper_info = Experiment(val)
        else:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Experiment info can be only an instance of Experiment class, input is {type(val).__name__}",
            )
        self._main_header.nfiles = self._exper_info.n_runs

    @property
    def experiment_info(self) -> Experiment:
        """Compatibility alias for :attr:`exper_info`."""
        return self.exper_info

    @experiment_info.setter
    def experiment_info(self, val: Any) -> None:
        self.exper_info = val

    @property
    def detpar(self) -> unique_references_container:
        """Detector arrays container from ``exper_info.detector_arrays``."""
        return self._exper_info.detector_arrays

    @detpar.setter
    def detpar(self, val: Any) -> None:
        if isinstance(val, dict) and "serial_name" not in val:
            if self._exper_info.detector_arrays.n_runs > 0:
                warn_with_id(
                    "HORACE:sqw:invalid_argument",
                    "Attempt to write single detpar struct directly via sqw to existing detector_array collection not implemented",
                )
            else:
                det_arr = IX_detector_array(val)
                urc = unique_references_container("IX_detector_array")
                urc.add(det_arr)
                if self._exper_info.n_runs > 1:
                    urc = urc.replicate_runs(self._exper_info.n_runs)
                self._exper_info.detector_arrays = urc
        else:
            self._exper_info.detector_arrays = val

    @property
    def data(self) -> DnDBase:
        """DnD image/grid data (:class:`~pyhorace.sqw.dnd_base.DnDBase`)."""
        return self._data

    @data.setter
    def data(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._data = d0d()
        elif isinstance(val, DnDBase):
            self._data = val
        elif isinstance(val, data_sqw_dnd):
            self._data = DnDBase.dnd(val)
        elif isinstance(val, dict) and "serial_name" in val:
            loaded = Serializable.loadobj(val)
            if not isinstance(loaded, DnDBase):
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"Data field has to be a DnDBase object, got {type(loaded).__name__}",
                )
            self._data = loaded
        else:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Data field has to be a DnDBase object, in fact it is: {type(val).__name__}",
            )

    @property
    def pix(self) -> PixelDataBase:
        """Pixel data (:class:`~pyhorace.data.pixels.pixel_data_base.PixelDataBase`)."""
        return self._pix

    @pix.setter
    def pix(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
            self._pix = PixelDataMemory()
        elif isinstance(val, PixelDataBase) or (hasattr(val, "run_label") and hasattr(val, "pix_position")):
            self._pix = val
        elif isinstance(val, dict) and "serial_name" in val:
            loaded = Serializable.loadobj(val)
            if not isinstance(loaded, PixelDataBase):
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"pix has to be a PixelDataBase object, got {type(loaded).__name__}",
                )
            self._pix = loaded
        else:
            try:
                self._pix = PixelDataBase.create(val)
            except Exception as exc:
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"pix has to be an instance of PixelDataBase class or convertible to it, got {type(val).__name__}: {exc}",
                ) from exc

    @property
    def creation_date(self) -> str:
        """Creation date string from ``main_header``."""
        return self._main_header.creation_date

    @creation_date.setter
    def creation_date(self, val: Any) -> None:
        self._main_header.creation_date = val
        self._data.creation_date = val

    @property
    def full_filename(self) -> str:
        """Full filename of the ``sqw`` object (returns temporary file path if ``is_tmp_obj`` is True)."""
        if self._is_tmp_obj and self._tmp_file_path:
            return self._tmp_file_path
        return self._main_header.full_filename

    @full_filename.setter
    def full_filename(self, val: Any) -> None:
        if val is None:
            s_val = ""
        elif isinstance(val, (str, os.PathLike)):
            s_val = str(val)
        else:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"full_filename must be a string, got {type(val).__name__}",
            )
        self._main_header.full_filename = s_val
        self._data.full_filename = s_val
        if isinstance(self._pix, PixelDataBase):
            self._pix.full_filename = s_val

    @property
    def is_tmp_obj(self) -> bool:
        """True if this ``sqw`` object owns a temporary backing file to be deleted on destruction."""
        return bool(getattr(self, "_is_tmp_obj", False))

    @property
    def npixels(self) -> int:
        """Total number of pixels in ``pix``."""
        return int(self._pix.num_pixels)

    @property
    def num_pixels(self) -> int:
        """Total number of pixels in ``pix``."""
        return int(self._pix.num_pixels)

    @property
    def runid_map(self) -> Any:
        """Map connecting ``run_id`` to 1-based experiment index in ``exper_info.expdata``."""
        if self._exper_info is None or self._exper_info.n_runs == 0:
            return self._exper_info.runid_map if self._exper_info is not None else []
        return self._exper_info.runid_map

    @property
    def header(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Legacy binary-file header structure(s) without ``instrument`` and ``sample`` fields."""
        if self._exper_info is None or self._exper_info.n_runs == 0:
            return []
        hdrs = self._exper_info.convert_to_old_headers()
        cleaned: List[Dict[str, Any]] = []
        for h in hdrs:
            hc = dict(h)
            hc.pop("instrument", None)
            hc.pop("sample", None)
            cleaned.append(hc)
        if len(cleaned) == 1:
            return cleaned[0]
        return cleaned

    @property
    def dnd_type(self) -> _CallableBool:
        """True if ``pix`` contains 0 pixels (usable both as a property and as a 0-arg method)."""
        return _CallableBool(self._pix.num_pixels == 0)

    @property
    def has_pixels(self) -> _CallableBool:
        """Return True if ``pix`` has at least one pixel (usable as property or 0-arg method)."""
        return _CallableBool(self._pix.num_pixels > 0)

    def _has_pixels_legacy(self) -> bool:
        """Return True if ``pix`` has at least one pixel."""
        return bool(self._pix.num_pixels > 0)

    @property
    def is_filebacked(self) -> bool:
        """True if ``pix`` has pixels and is file-backed."""
        return bool(self.has_pixels() and self._pix.is_filebacked)

    # ------------------------------------------------------------------
    # Delegated DnD / metadata properties (used via SQWDnDBase)
    # ------------------------------------------------------------------
    @property
    def axes(self) -> Any:
        return self._data.axes

    @axes.setter
    def axes(self, val: Any) -> None:
        self._data.axes = val

    @property
    def proj(self) -> Any:
        return self._data.proj

    @proj.setter
    def proj(self, val: Any) -> None:
        self._data.proj = val

    @property
    def dimensions(self) -> Any:
        return self._data.dimensions

    @property
    def s(self) -> np.ndarray:
        return self._data.s

    @s.setter
    def s(self, val: Any) -> None:
        self._data.s = val

    @property
    def e(self) -> np.ndarray:
        return self._data.e

    @e.setter
    def e(self, val: Any) -> None:
        self._data.e = val

    @property
    def npix(self) -> np.ndarray:
        return self._data.npix

    @npix.setter
    def npix(self, val: Any) -> None:
        self._data.npix = val

    @property
    def filename(self) -> str:
        return self._main_header.filename or self._data.filename

    @filename.setter
    def filename(self, val: str) -> None:
        self._main_header.filename = val
        self._data.filename = val

    @property
    def filepath(self) -> str:
        return self._main_header.filepath or self._data.filepath

    @filepath.setter
    def filepath(self, val: str) -> None:
        self._main_header.filepath = val
        self._data.filepath = val

    @property
    def title(self) -> str:
        return self._data.title or self._main_header.title

    @title.setter
    def title(self, val: str) -> None:
        self._main_header.title = val
        self._data.title = val

    @property
    def alatt(self) -> np.ndarray:
        return self._data.alatt

    @alatt.setter
    def alatt(self, val: Any) -> None:
        self._data.alatt = val

    @property
    def angdeg(self) -> np.ndarray:
        return self._data.angdeg

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        self._data.angdeg = val

    @property
    def offset(self) -> np.ndarray:
        return self._data.offset

    @offset.setter
    def offset(self, val: Any) -> None:
        self._data.offset = val

    @property
    def label(self) -> List[str]:
        return self._data.label

    @label.setter
    def label(self, val: Any) -> None:
        self._data.label = val

    @property
    def iax(self) -> np.ndarray:
        return self._data.iax

    @property
    def iint(self) -> np.ndarray:
        return self._data.iint

    @property
    def pax(self) -> np.ndarray:
        return self._data.pax

    @property
    def p(self) -> List[np.ndarray]:
        return self._data.p

    @property
    def dax(self) -> np.ndarray:
        return self._data.dax

    @dax.setter
    def dax(self, val: Any) -> None:
        self._data.dax = val

    @property
    def img_db_range(self) -> np.ndarray:
        return self._data.img_db_range

    @property
    def nbins(self) -> int:
        return self._data.nbins

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def classVersion(self) -> int:
        return 6

    def class_version(self) -> int:
        return self.classVersion()

    def saveableFields(self) -> List[str]:
        return list(self._fields_to_save)

    def saveable_fields(self) -> List[str]:
        return self.saveableFields()

    def check_combo_arg(self) -> "sqw":
        """Validate interdependent properties of ``sqw``."""
        if self._data is not None and hasattr(self._data, "check_combo_arg"):
            self._data.do_check_combo_arg = True
            self._data = self._data.check_combo_arg()
        if (
            self._exper_info is not None
            and self._exper_info.n_runs > 0
            and self._main_header.nfiles != self._exper_info.n_runs
        ):
            self._main_header.nfiles = self._exper_info.n_runs
        return self

    def convert_old_struct(
        self, inputs: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], "sqw"]:
        """Convert older ``sqw`` struct versions (v1..v5) to v6 format (port of ``@sqw/from_old_struct.m``)."""
        out = dict(inputs)
        if "array_dat" in out and "main_header" not in out:
            ad = out["array_dat"]
            if isinstance(ad, dict):
                out = dict(ad)
        out.pop("serial_name", None)
        out.pop("version", None)

        if "header" in out and "exper_info" not in out and "experiment_info" not in out:
            hdr = out.pop("header")
            if hdr is not None and not (isinstance(hdr, (list, tuple, dict)) and len(hdr) == 0):
                out["exper_info"] = Experiment.build_from_binfile_headers(hdr)
            else:
                out["exper_info"] = Experiment()

        if "experiment_info" in out and "exper_info" not in out:
            out["exper_info"] = out.pop("experiment_info")

        if "data" in out and isinstance(out["data"], dict) and "pix" in out["data"] and "pix" not in out:
            d_copy = dict(out["data"])
            out["pix"] = d_copy.pop("pix")
            out["data"] = d_copy

        detpar = out.pop("detpar", None)
        if detpar is not None and isinstance(detpar, dict) and "exper_info" in out:
            exp_val = out["exper_info"]
            if isinstance(exp_val, dict):
                exp_val = Experiment.loadobj(exp_val)
            if isinstance(exp_val, Experiment) and exp_val.n_runs > 0 and exp_val.detector_arrays.n_runs == 0:
                det_arr = IX_detector_array(detpar)
                urc = unique_references_container("IX_detector_array")
                urc.add(det_arr)
                if exp_val.n_runs > 1:
                    urc = urc.replicate_runs(exp_val.n_runs)
                exp_val.detector_arrays = urc
                out["exper_info"] = exp_val

        return out, self

    def from_old_struct(self, inputs: Dict[str, Any], ver: Optional[float] = None) -> "sqw":
        """Restore ``sqw`` from an older structure format (port of ``@sqw/from_old_struct.m``)."""
        converted, _ = self.convert_old_struct(inputs, ver)
        self._init_from_dict(converted)
        # Align run_ids with pixel run_idx if needed for legacy files without creation_date
        if (
            not self._main_header.creation_date_defined
            and self._pix.num_pixels > 0
            and self._exper_info.n_runs > 0
        ):
            from pyhorace.coord_transf._dependencies import fast_map

            u_ids = np.unique(self._pix.run_idx).astype(float)
            if u_ids.size == self._exper_info.n_runs:
                cur_keys = np.asarray(self._exper_info.runid_map.keys, dtype=float)
                if not np.array_equal(cur_keys, u_ids):
                    self._exper_info.runid_map = fast_map(
                        u_ids, np.arange(1, self._exper_info.n_runs + 1, dtype=int)
                    )
                    self._exper_info.runid_recalculated = False
        return self

    def saveobj(self) -> Dict[str, Any]:
        """Return dictionary representation for saving (port of ``@sqw/sqw.m:saveobj``)."""
        if not self._main_header.creation_date_defined:
            from datetime import datetime

            self._main_header.creation_date = datetime.now()
        return super().saveobj()

    @staticmethod
    def _mat_to_dict(val: Any) -> Any:
        """Recursively convert ``scipy.io.matlab.mat_struct`` objects to ``dict``."""
        if hasattr(val, "_fieldnames"):
            return {fn: sqw._mat_to_dict(getattr(val, fn)) for fn in val._fieldnames}
        if isinstance(val, dict):
            return {k: sqw._mat_to_dict(v) for k, v in val.items()}
        if isinstance(val, np.ndarray) and val.dtype == object:
            return [sqw._mat_to_dict(x) for x in val.flat]
        if isinstance(val, (list, tuple)):
            return [sqw._mat_to_dict(x) for x in val]
        return val

    @classmethod
    def loadobj(cls, S: Any) -> Any:
        """Load an ``sqw`` instance from a saved struct or legacy MATLAB object."""
        if isinstance(S, sqw):
            return S.copy()
        S_conv = cls._mat_to_dict(S)
        if isinstance(S_conv, dict):
            d = dict(S_conv)
            if "experiment_info" in d and "exper_info" not in d:
                d["exper_info"] = d.pop("experiment_info")
            if (
                "array_dat" in d
                and isinstance(d["array_dat"], dict)
                and "experiment_info" in d["array_dat"]
                and "exper_info" not in d["array_dat"]
            ):
                ad = dict(d["array_dat"])
                ad["exper_info"] = ad.pop("experiment_info")
                d["array_dat"] = ad
            obj = cls()
            return obj.from_old_struct(d)
        obj = cls()
        return Serializable.loadobj(S_conv, obj)

    # ------------------------------------------------------------------
    # Basic DnD / SQWDnDBase delegations
    # ------------------------------------------------------------------
    def dnd(self) -> DnDBase:
        """Extract the underlying DnD object (port of ``@sqw/dnd.m``)."""
        return self._data.copy()

    def d0d(self) -> d0d:
        """Convert 0D SQW to ``d0d`` (port of ``@sqw/d0d.m``)."""
        return d0d(self._data)

    def d1d(self) -> d1d:
        """Convert 1D SQW to ``d1d`` (port of ``@sqw/d1d.m``)."""
        return d1d(self._data)

    def d2d(self) -> d2d:
        """Convert 2D SQW to ``d2d`` (port of ``@sqw/d2d.m``)."""
        return d2d(self._data)

    def d3d(self) -> d3d:
        """Convert 3D SQW to ``d3d`` (port of ``@sqw/d3d.m``)."""
        return d3d(self._data)

    def d4d(self) -> d4d:
        """Convert 4D SQW to ``d4d`` (port of ``@sqw/d4d.m``)."""
        return d4d(self._data)

    def get_NUM_DIMS(self) -> int:
        return int(self._data.dimensions)

    def get_img_size(self) -> Tuple[int, ...]:
        return self._data.sigvar_size()

    def sigvar(self) -> sigvar:
        return self._data.sigvar()

    def sigvar_get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._data.sigvar_get()

    def sigvar_set(self, sigvar_obj: Any) -> "sqw":
        self._data.sigvar_set(sigvar_obj)
        return self

    def sigvar_size(self) -> Tuple[int, ...]:
        return self._data.sigvar_size()

    def calculate_q_bins(self, nargout: Optional[int] = None) -> Any:
        return self._data.calculate_q_bins(nargout=nargout)

    def calculate_qw_bins(
        self,
        *args: Any,
        boundaries: bool = False,
        edges: bool = False,
        nargout: Optional[int] = None,
    ) -> Any:
        return self._data.calculate_qw_bins(
            *args, boundaries=boundaries, edges=edges, nargout=nargout
        )

    def header_average(self, nargout: Optional[int] = None) -> Any:
        """Return average header over contributing runs (port of ``@sqw/header_average.m``)."""
        return self._exper_info.header_average(nargout=nargout)

    def my_detpar(self) -> Dict[str, Any]:
        """Return legacy detpar structure from ``exper_info.detector_arrays`` (port of ``@sqw/my_detpar.m``)."""
        if self._exper_info.detector_arrays.n_runs == 0:
            return {}
        return self._exper_info.detector_arrays[0].get_detpar_representation()

    # ------------------------------------------------------------------
    # Method Stubs for unported @sqw methods
    # ------------------------------------------------------------------
    def aclims(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/aclims.m``."""
        # TODO(port-dependency): Port @sqw/aclims.m
        raise NotImplementedError("sqw.aclims is not yet implemented.")

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/apply.m``."""
        # TODO(port-dependency): Port @sqw/apply.m
        raise NotImplementedError("sqw.apply is not yet implemented.")

    def calc_sqw_urange(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/calc_sqw_urange.m``."""
        # TODO(port-dependency): Port @sqw/calc_sqw_urange.m
        raise NotImplementedError("sqw.calc_sqw_urange is not yet implemented.")

    def calculate_qw_pixels(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/calculate_qw_pixels.m``."""
        # TODO(port-dependency): Port @sqw/calculate_qw_pixels.m
        raise NotImplementedError("sqw.calculate_qw_pixels is not yet implemented.")

    def calculate_qw_pixels2(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/calculate_qw_pixels2.m``."""
        # TODO(port-dependency): Port @sqw/calculate_qw_pixels2.m
        raise NotImplementedError("sqw.calculate_qw_pixels2 is not yet implemented.")

    def calculate_uproj_pixels(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/calculate_uproj_pixels.m``."""
        # TODO(port-dependency): Port @sqw/calculate_uproj_pixels.m
        raise NotImplementedError("sqw.calculate_uproj_pixels is not yet implemented.")

    def change_crystal(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/change_crystal.m``."""
        # TODO(port-dependency): Port @sqw/change_crystal.m
        raise NotImplementedError("sqw.change_crystal is not yet implemented.")

    def combine_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/combine_sqw.m``."""
        # TODO(port-dependency): Port @sqw/combine_sqw.m
        raise NotImplementedError("sqw.combine_sqw is not yet implemented.")

    def compact(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/compact.m``."""
        # TODO(port-dependency): Port @sqw/compact.m
        raise NotImplementedError("sqw.compact is not yet implemented.")

    def cut(self, *args: Any, **kwargs: Any) -> Any:
        """Take a cut from an ``sqw`` object along existing or new projection axes.

        Ported from ``@sqw/cut.m``, ``@sqw/private/cut_single_.m``, and
        ``@sqw/private/cut_accumulate_data_.m``.
        """
        from pyhorace.sqw._sqw_methods import cut_sqw_method

        return cut_sqw_method(self, *args, **kwargs)

    def cut_dnd(self, *args: Any, **kwargs: Any) -> Any:
        """Take a DND cut (no pixels retained) from an ``sqw`` object (port of ``@sqw/cut_dnd.m``)."""
        from pyhorace.sqw._sqw_methods import cut_dnd_method

        return cut_dnd_method(self, *args, **kwargs)

    def cut_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Take an SQW cut from an ``sqw`` object (port of ``@sqw/cut_sqw.m``)."""
        from pyhorace.sqw._sqw_methods import cut_sqw_only_method

        return cut_sqw_only_method(self, *args, **kwargs)

    def cut_sym(self, *args: Any, **kwargs: Any) -> Any:
        """Take a symmetry-enabled cut from an ``sqw`` object (port of ``@sqw/cut_sym.m``)."""
        from pyhorace.sqw._sqw_methods import cut_sym_method

        return cut_sym_method(self, *args, **kwargs)

    def define_target_axes_block(
        self,
        targ_proj: Any,
        pbin: Sequence[Any],
        sym: Any = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Build target AxesBlockBase and transformed projections (delegates to ``data.define_target_axes_block``)."""
        from pyhorace.coord_transf.bmatrix import _infer_nargout

        if nargout is None:
            nargout = _infer_nargout(default=1)
        return self._data.define_target_axes_block(
            targ_proj, pbin, sym=sym, nargout=nargout
        )

    def get_targ_range(self, targ_proj: Any = None, *args: Any) -> Any:
        """Compute range of this dataset's axes in a target projection."""
        return self._data.get_targ_range(targ_proj, *args)

    def get_proj_and_pbin(self, nargout: Optional[int] = None) -> Any:
        """Retrieve the projection and binning descriptors of this SQW dataset."""
        from pyhorace.coord_transf.bmatrix import _infer_nargout

        if nargout is None:
            nargout = _infer_nargout(default=2)
        return self._data.get_proj_and_pbin(nargout=nargout)

    def _cut_single(
        self,
        targ_proj: Any,
        targ_axes: Any,
        opt: Dict[str, Any],
        log_level: int,
        sym: Any,
        return_cut: bool = True,
    ) -> Optional[Union["sqw", DnDBase]]:
        """Execute a single cut on this ``sqw`` object (port of ``@sqw/private/cut_single_.m``)."""
        from pyhorace.sqw._sqw_methods import _cut_single as _cut_single_impl

        return _cut_single_impl(
            self, targ_proj, targ_axes, opt, log_level, sym, return_cut=return_cut
        )

    def _cut_accumulate_data(
        self,
        targ_proj: Any,
        targ_axes: Any,
        keep_pixels: bool,
        log_level: int,
        return_cut: bool,
        sym: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PixelDataBase, np.ndarray]:
        """Accumulate binned signal, error, npix, and pixels for a cut (port of ``@sqw/private/cut_accumulate_data_.m``)."""
        from pyhorace.sqw._sqw_methods import (
            _cut_accumulate_data as _cut_accumulate_data_impl,
        )

        return _cut_accumulate_data_impl(
            self, targ_proj, targ_axes, keep_pixels, log_level, return_cut, sym
        )

    @staticmethod
    def _merge_ranges(
        starts_cell: Sequence[np.ndarray], sizes_cell: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge overlapping or adjacent 1-based pixel block ranges across symmetries."""
        from pyhorace.sqw._sqw_methods import _merge_ranges as _merge_ranges_impl

        return _merge_ranges_impl(starts_cell, sizes_cell)

    @staticmethod
    def _split_data_blocks(
        block_starts: np.ndarray, block_sizes: np.ndarray, buf_size: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split pixel block ranges into chunks of at most ``buf_size`` pixels."""
        from pyhorace.sqw._sqw_methods import (
            _split_data_blocks as _split_data_blocks_impl,
        )

        return _split_data_blocks_impl(block_starts, block_sizes, buf_size)

    @staticmethod
    def _sort_pix(
        pix_retained: Sequence[PixelDataBase],
        pix_ix_retained: Sequence[np.ndarray],
        npix: np.ndarray,
    ) -> PixelDataBase:
        """Sort accumulated pixel chunks by 1-based bin index ``pix_ix_retained``."""
        from pyhorace.sqw._sqw_methods import _sort_pix as _sort_pix_impl

        return _sort_pix_impl(pix_retained, pix_ix_retained, npix)

    def fake_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/fake_sqw.m``."""
        # TODO(port-dependency): Port @sqw/fake_sqw.m
        raise NotImplementedError("sqw.fake_sqw is not yet implemented.")

    '''
        def gen_sqw(self, *args: Any, **kwargs: Any) -> Any:
            """Stub for ``@sqw/gen_sqw.m``."""
            # TODO(port-dependency): Port @sqw/gen_sqw.m
            raise NotImplementedError("sqw.gen_sqw is not yet implemented.")
    '''

    def get_efix(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/get_efix.m``."""
        # TODO(port-dependency): Port @sqw/get_efix.m
        raise NotImplementedError("sqw.get_efix is not yet implemented.")

    def get_inst_class(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/get_inst_class.m``."""
        # TODO(port-dependency): Port @sqw/get_inst_class.m
        raise NotImplementedError("sqw.get_inst_class is not yet implemented.")

    def get_nearest_pixels(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/get_nearest_pixels.m``."""
        # TODO(port-dependency): Port @sqw/get_nearest_pixels.m
        raise NotImplementedError("sqw.get_nearest_pixels is not yet implemented.")

    def get_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/get_sqw.m``."""
        # TODO(port-dependency): Port @sqw/get_sqw.m
        raise NotImplementedError("sqw.get_sqw is not yet implemented.")

    def IX_dataset_1d(self) -> Any:
        """Convert 1D sqw object into IX_dataset_1d (port of ``@sqw/IX_dataset_1d.m``)."""
        from pyhorace.data.ix_dataset_1d import IX_dataset_1d as _IX1D

        if int(self.dimensions()) != 1:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Not all elements in the array of sqw objects are one dimensional",
            )
        res = _IX1D(self.data)
        res._obj_holder = getattr(self, "_obj_holder", None) or self
        return res

    def IX_dataset_2d(self) -> Any:
        """Convert 2D sqw object into IX_dataset_2d (port of ``@sqw/IX_dataset_2d.m``)."""
        from pyhorace.data.ix_dataset_2d import IX_dataset_2d as _IX2D

        if int(self.dimensions()) != 2:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "sqw object is not two dimensional",
            )
        res = _IX2D(self.data)
        res._obj_holder = getattr(self, "_obj_holder", None) or self
        return res

    def IX_dataset_3d(self) -> Any:
        """Convert 3D sqw object into IX_dataset_3d (port of ``@sqw/IX_dataset_3d.m``)."""
        from pyhorace.data.ix_dataset_3d import IX_dataset_3d as _IX3D

        if int(self.dimensions()) != 3:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Not all elements in the array of sqw objects are three dimensional",
            )
        res = _IX3D(self.data)
        res._obj_holder = getattr(self, "_obj_holder", None) or self
        return res

    def join(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/join.m``."""
        # TODO(port-dependency): Port @sqw/join.m
        raise NotImplementedError("sqw.join is not yet implemented.")

    def mask(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/mask.m``."""
        # TODO(port-dependency): Port @sqw/mask.m
        raise NotImplementedError("sqw.mask is not yet implemented.")

    def mask_detectors(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/mask_detectors.m``."""
        # TODO(port-dependency): Port @sqw/mask_detectors.m
        raise NotImplementedError("sqw.mask_detectors is not yet implemented.")

    def mask_pixels(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/mask_pixels.m``."""
        # TODO(port-dependency): Port @sqw/mask_pixels.m
        raise NotImplementedError("sqw.mask_pixels is not yet implemented.")

    def mask_random_fraction_pixels(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/mask_random_fraction_pixels.m``."""
        # TODO(port-dependency): Port @sqw/mask_random_fraction_pixels.m
        raise NotImplementedError("sqw.mask_random_fraction_pixels is not yet implemented.")

    def mask_random_pixels(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/mask_random_pixels.m``."""
        # TODO(port-dependency): Port @sqw/mask_random_pixels.m
        raise NotImplementedError("sqw.mask_random_pixels is not yet implemented.")

    def mask_runs(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/mask_runs.m``."""
        # TODO(port-dependency): Port @sqw/mask_runs.m
        raise NotImplementedError("sqw.mask_runs is not yet implemented.")

    def multifit_func(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/multifit_func.m``."""
        # TODO(port-dependency): Port @sqw/multifit_func.m
        raise NotImplementedError("sqw.multifit_func is not yet implemented.")

    def multifit_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/multifit_sqw.m``."""
        # TODO(port-dependency): Port @sqw/multifit_sqw.m
        raise NotImplementedError("sqw.multifit_sqw is not yet implemented.")

    def multifit_sqw_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/multifit_sqw_sqw.m``."""
        # TODO(port-dependency): Port @sqw/multifit_sqw_sqw.m
        raise NotImplementedError("sqw.multifit_sqw_sqw is not yet implemented.")

    def noisify(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/noisify.m``."""
        # TODO(port-dependency): Port @sqw/noisify.m
        raise NotImplementedError("sqw.noisify is not yet implemented.")

    def page_op(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/page_op.m``."""
        # TODO(port-dependency): Port @sqw/page_op.m
        return self.pix.page_op(*args, **kwargs)

    def apply_op(self, *args: Any) -> Any:
        """Apply a :class:`~pyhorace.coord_transf.page_operations.PageOpBase` operation to this ``sqw`` object.

        Ported from ``@sqw/sqw.m`` (``apply_op``).
        Can be called as ``wout = w.apply_op(op)`` or ``wout = sqw.apply_op(w, op)``.
        """
        operation = args[-1]
        target = args[0] if len(args) > 1 else self
        if isinstance(target, (list, tuple, np.ndarray)):
            if len(target) == 0:
                return target
            out = [item.pix.apply_op(item, operation) for item in target]
            return DataOpArray(out) if isinstance(target, DataOpArray) else out
        return target.pix.apply_op(target, operation)

    def get_write_handle(self, outfile: str = "", outfile_defined: bool = False) -> Any:
        """Initialize a :class:`~pyhorace.data.pixels.pix_write_handle.pix_write_handle` for paged output.

        Ported from ``horace_core/sqw/@sqw/get_write_handle.m``.
        """
        from pyhorace.data.pixels.pix_write_handle import pix_write_handle
        from pyhorace.io.sqw_formats_factory import sqw_formats_factory

        if not self.pix.is_filebacked:
            return None
        if outfile_defined:
            op_outfile = str(outfile)
            move_to_original = False
        else:
            op_outfile, move_to_original = PixelDataBase.build_op_filename(
                self.pix.full_filename, outfile
            )
        orig_mh = copy.deepcopy(self._main_header)
        orig_dmeta = copy.deepcopy(self._data.metadata)
        ldr = sqw_formats_factory.instance().get_pref_access(self)
        ldr = ldr.init(self, op_outfile)
        ldr = ldr.put_sqw("-hold_pix_place")
        self._main_header = orig_mh
        self._main_header = orig_mh
        old_chk = self._data.do_check_combo_arg
        self._data.do_check_combo_arg = False
        try:
            self._data.metadata = orig_dmeta
        finally:
            self._data.do_check_combo_arg = old_chk

        wh = pix_write_handle(ldr)
        wh.move_to_original = move_to_original
        return wh

    def finish_dump(self, page_op: Any) -> "sqw":
        """Complete writing a file-backed ``sqw`` object after ``PageOp`` execution.

        Ported from ``horace_core/sqw/@sqw/finish_dump.m``.
        """
        import shutil

        wh = page_op.write_handle
        if wh is None:
            return self
        if not self.pix.is_filebacked:
            raise HoraceError(
                "HORACE:sqw:runtime_error",
                "Some pixel dump completed in write_handle without sqw pixels been filebacked",
            )
        if page_op.outfile:
            self.full_filename = page_op.outfile
            out_blocks = ["bl__main_header", "bl_data_metadata", "bl_pix_metadata"]
        else:
            self.pix.full_filename = wh.write_file_name
            out_blocks = ["bl_pix_metadata"]

        pix_meta = wh.finish_pix_dump(self.pix, False)
        if page_op.outfile:
            pix_meta.full_filename = page_op.outfile
        self.pix.metadata = pix_meta

        sqw_ldr = wh.release_pixinit_info(True)
        if page_op.exp_modified:
            out_blocks.extend(
                [
                    "bl__main_header",
                    "bl_experiment_info_instruments",
                    "bl_experiment_info_samples",
                    "bl_experiment_info_expdata",
                ]
            )
        if not page_op.changes_pix_only:
            out_blocks.append("bl_data_nd_data")
        unique_blocks: List[str] = []
        for b in out_blocks:
            if b not in unique_blocks:
                unique_blocks.append(b)
        sqw_ldr = sqw_ldr.put_new_blocks_values(self, "update", unique_blocks)

        set_as_tmp_obj = False
        if wh.move_to_original:
            self.deactivate()
            page_op.pix.deactivate()
            sqw_ldr.deactivate()
            shutil.move(wh.write_file_name, page_op.outfile)
            if page_op.init_filebacked_output:
                sqw_ldr.full_filename = page_op.outfile
                sqw_ldr.activate()
        elif wh.is_tmp_file and page_op.init_filebacked_output:
            set_as_tmp_obj = True
        else:
            self.tmp_file_holder_ = None
            self._is_tmp_obj = False
            self._tmp_file_path = ""
            self.full_filename = sqw_ldr.full_filename

        if page_op.init_filebacked_output:
            self.pix = self.pix.init(sqw_ldr)
            if set_as_tmp_obj:
                self.set_as_tmp_obj(wh.write_file_name)
        else:
            self.pix = PixelDataMemory()
        sqw_ldr.delete()
        return self

    def finalize_alignment(self, *args: Any) -> "sqw":
        """Apply pixel alignment matrix to pixels and recompute bin data if needed.

        Ported from ``horace_core/sqw/@sqw/finalize_alignment.m``.
        """
        nopix = any(isinstance(a, str) and a.lower() == "-nopix" for a in args)
        if not self.pix.is_corrected or nopix:
            return self
        return self.recompute_bin_data()

    @staticmethod
    def generate_cube_sqw(shape: Any, *args: Any) -> "sqw":
        """Generate an instance of ``sqw`` data with given shape for testing.

        Ported from ``horace_core/sqw/@sqw/generate_cube_sqw.m``.
        """
        from pyhorace.coord_transf.a_projection_base import aProjectionBase
        from pyhorace.coord_transf.axes_block_base import AxesBlockBase
        from pyhorace.coord_transf.line_axes import line_axes
        from pyhorace.coord_transf.line_proj import line_proj
        from pyhorace.data.instr_classes.ix_detector_array import IX_detector_array
        from pyhorace.data.instr_classes.ix_experiment import IX_experiment
        from pyhorace.data.instr_classes.ix_null_inst import IX_null_inst
        from pyhorace.data.instr_classes.ix_sample import IX_sample

        out = sqw()
        alatt = np.array([2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=np.float64)
        angdeg = np.array([90.0, 90.0, 90.0], dtype=np.float64)

        if isinstance(shape, (int, float, np.number)) and not isinstance(shape, bool):
            sh = int(shape)
            if sh % 2 == 0:
                minloc = -sh / 2.0
                maxloc = sh / 2.0
            else:
                minloc = -(sh - 1) / 2.0
                maxloc = (sh - 1) / 2.0
            img_range = np.vstack([np.ones(4) * minloc, np.ones(4) * maxloc])
            sqw_axes = line_axes(img_range=img_range, nbins_all_dims=np.ones(4, dtype=int) * sh)
        elif isinstance(shape, AxesBlockBase):
            sqw_axes = shape
            img_range = np.asarray(sqw_axes.img_range, dtype=np.float64)
        else:
            raise HoraceError(
                "HORACE:sqw:not_implemented",
                f"Currently unsupported input type: {type(shape).__name__}",
            )

        coord = sqw_axes.get_bin_nodes("-bin_centre")
        npix = int(coord.shape[1])

        if len(args) > 0 and isinstance(args[0], aProjectionBase):
            proj = copy.deepcopy(args[0])
            argi = list(args[1:])
            if proj.alatt_defined:
                alatt = np.asarray(proj.alatt, dtype=np.float64)
            else:
                proj.alatt = alatt
            if proj.angdeg_defined:
                angdeg = np.asarray(proj.angdeg, dtype=np.float64)
            else:
                proj.angdeg = angdeg
        else:
            argi = list(args)
            proj = line_proj([1, 0, 0], [0, 1, 0], alatt=alatt, angdeg=angdeg, type="ppr")

        coord = proj.transform_img_to_pix(coord)
        idx_1_to_npix = np.arange(1, npix + 1, dtype=np.float64)
        pix_data_arr = np.vstack(
            [
                coord,
                np.ones((1, npix), dtype=np.float64),
                np.tile(idx_1_to_npix, (4, 1)),
            ]
        )
        pix = PixelDataMemory(pix_data_arr)

        if int(sqw_axes.nbins_all_dims[3]) == 1:
            en = img_range[:, 3].ravel()
            efix = float(en[1]) * 1.01
            if efix <= 0:
                efix = 1.0
        else:
            en = np.asarray(sqw_axes.p[-1], dtype=np.float64).ravel()
            efix = float(np.max(en)) * 1.01

        samp = IX_sample(alatt, angdeg)
        expdata_struct = {
            "filename": "fake",
            "filepath": "/fake",
            "efix": efix,
            "emode": 1,
            "cu": [1, 0, 0],
            "cv": [0, 1, 0],
            "psi": 1,
            "omega": 1,
            "dpsi": 1,
            "gl": 1,
            "gs": 1,
            "en": en,
            "uoffset": [0, 0, 0],
            "run_id": 1,
        }
        detpar_struct = IX_detector_array().get_detpar_representation()
        detpar_struct["filename"] = "fake"
        detpar_struct["filepath"] = "/fake"

        samples = samp
        instruments = IX_null_inst()
        detector_arrays = IX_detector_array(detpar_struct)
        expdata = IX_experiment(expdata_struct)
        expinfo = Experiment(detector_arrays, instruments, samples, expdata)

        ax0 = line_axes(img_range=img_range, nbins_all_dims=[1, 1, 1, 1])
        ax0 = proj.copy_proj_defined_properties_to_axes(ax0)
        dnd_0 = DnDBase.dnd(
            ax0,
            proj,
            np.array(float(npix)),
            np.array(float(npix)),
            np.array(float(npix)),
        )

        out._do_check_combo_arg = False
        out.pix = pix
        out.experiment_info = expinfo
        out.data = dnd_0
        out._do_check_combo_arg = True
        out.check_combo_arg()

        if len(argi) > 0 and callable(argi[0]):
            if len(argi) > 1:
                out = out.sqw_eval(*argi)
            else:
                out = out.sqw_eval(argi[0], [])
            out.pix.variance = np.ones((1, out.pix.num_pixels), dtype=np.float64)
            out.data.e = np.zeros_like(out.data.s)

        cut_range = sqw_axes.get_cut_range("-full_range")
        out = out.cut(proj, *cut_range)
        return out

    def parse_pixel_indices(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/parse_pixel_indices.m``."""
        # TODO(port-dependency): Port @sqw/parse_pixel_indices.m
        raise NotImplementedError("sqw.parse_pixel_indices is not yet implemented.")

    def recompute_bin_data(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/recompute_bin_data.m``."""
        # TODO(port-dependency): Port @sqw/recompute_bin_data.m
        from pyhorace.coord_transf.page_operations.page_op_recompute_bins import (
            PageOp_recompute_bins,
        )

        outfile = kwargs.get("outfile", args[0] if len(args) > 0 else "")
        if self.num_pixels == 0:
            return self
        op = PageOp_recompute_bins()
        if outfile:
            op.outfile = str(outfile)
        op = op.init(self)
        return self.apply_op(op)

    def refine_crystal(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/refine_crystal.m``."""
        # TODO(port-dependency): Port @sqw/refine_crystal.m
        raise NotImplementedError("sqw.refine_crystal is not yet implemented.")

    def refine_crystal_alter_moments(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/refine_crystal_alter_moments.m``."""
        # TODO(port-dependency): Port @sqw/refine_crystal_alter_moments.m
        raise NotImplementedError("sqw.refine_crystal_alter_moments is not yet implemented.")

    def replicate(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/replicate.m``."""
        # TODO(port-dependency): Port @sqw/replicate.m
        raise NotImplementedError("sqw.replicate is not yet implemented.")

    def resolution_plot(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/resolution_plot.m``."""
        # TODO(port-dependency): Port @sqw/resolution_plot.m
        raise NotImplementedError("sqw.resolution_plot is not yet implemented.")

    def section(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/section.m``."""
        # TODO(port-dependency): Port @sqw/section.m
        raise NotImplementedError("sqw.section is not yet implemented.")

    def set_efix(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/set_efix.m``."""
        # TODO(port-dependency): Port @sqw/set_efix.m
        raise NotImplementedError("sqw.set_efix is not yet implemented.")

    def set_instrument(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/set_instrument.m``."""
        # TODO(port-dependency): Port @sqw/set_instrument.m
        raise NotImplementedError("sqw.set_instrument is not yet implemented.")

    def set_mod_pulse(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/set_mod_pulse.m``."""
        # TODO(port-dependency): Port @sqw/set_mod_pulse.m
        raise NotImplementedError("sqw.set_mod_pulse is not yet implemented.")

    def set_sample(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/set_sample.m``."""
        # TODO(port-dependency): Port @sqw/set_sample.m
        raise NotImplementedError("sqw.set_sample is not yet implemented.")

    def shift_energy_bins(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/shift_energy_bins.m``."""
        # TODO(port-dependency): Port @sqw/shift_energy_bins.m
        raise NotImplementedError("sqw.shift_energy_bins is not yet implemented.")

    def shift_pixels(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/shift_pixels.m``."""
        # TODO(port-dependency): Port @sqw/shift_pixels.m
        raise NotImplementedError("sqw.shift_pixels is not yet implemented.")

    def signal(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/signal.m``."""
        # TODO(port-dependency): Port @sqw/signal.m
        raise NotImplementedError("sqw.signal is not yet implemented.")

    def slim(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/slim.m``."""
        # TODO(port-dependency): Port @sqw/slim.m
        raise NotImplementedError("sqw.slim is not yet implemented.")

    def smooth(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/smooth.m``."""
        # TODO(port-dependency): Port @sqw/smooth.m
        raise NotImplementedError("sqw.smooth is not yet implemented.")

    def smooth_units(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/smooth_units.m``."""
        # TODO(port-dependency): Port @sqw/smooth_units.m
        raise NotImplementedError("sqw.smooth_units is not yet implemented.")

    def split(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/split.m``."""
        # TODO(port-dependency): Port @sqw/split.m
        raise NotImplementedError("sqw.split is not yet implemented.")

    def symmetrise_sqw(self, *args: Any, **kwargs: Any) -> "sqw":
        """Symmetrise ``sqw`` dataset according to ``Symop`` (port of ``@sqw/symmetrise_sqw.m``)."""
        from pyhorace.coord_transf.a_projection_base import aProjectionBase
        from pyhorace.coord_transf.axes_block_base import normalize_signal
        from pyhorace.coord_transf.expand_box import expand_box, get_geometry
        from pyhorace.coord_transf.symop import (
            Symop,
            SymopArray,
            SymopReflection,
        )

        if not self.has_pixels():
            raise HoraceError(
                "HORACE:symmetrise_sqw:invalid_argument",
                "input object must be sqw type with detector pixel information",
            )

        varargin = list(args)
        if len(varargin) > 0 and isinstance(varargin[-1], aProjectionBase):
            transf_proj = varargin.pop()
            if getattr(transf_proj, "nonorthogonal", False):
                raise HoraceError(
                    "HORACE:symmetrise_sqw:invalid_argument",
                    "Cannot symmetrise to non-orthogonal projection",
                )

        if len(varargin) == 1 and (
            isinstance(varargin[0], (Symop, SymopArray))
            or (
                isinstance(varargin[0], (list, tuple, np.ndarray))
                and len(varargin[0]) > 0
                and all(isinstance(x, (Symop, SymopArray)) for x in varargin[0])
            )
        ):
            sym_in = varargin[0]
        elif len(varargin) == 3:
            warn_with_id(
                "HORACE:symmetrise_sqw:deprecated",
                'Passing vectors to symmetrise_sqw is deprecated, please use "Symop"',
            )
            sym_in = SymopReflection(*varargin)
        else:
            raise HoraceError(
                "HORACE:symmetrise_sqw:invalid_argument",
                "Call as:\nwout = symmetrise_sqw(win, sym)\nDEPRECATED wout = symmetrise_sqw(win,v1,v2,v3)",
            )

        wout = self.copy()
        if wout.pix.is_filebacked:
            wout.pix = PixelDataMemory(wout.pix)
        else:
            wout.pix = copy.deepcopy(wout.pix)

        proj = wout.data.proj
        sym_validated, _ = Symop.validate_and_generate_sym(sym_in, proj, nargout=2)

        if isinstance(sym_validated, list) and not isinstance(sym_validated, SymopArray):
            for op in sym_validated:
                wout.pix = op.transform_pix(wout.pix)
            sym_arr: List[Symop] = []
            for item in sym_validated:
                if isinstance(item, SymopArray):
                    sym_arr.extend(item)
                else:
                    sym_arr.append(item)
        else:
            wout.pix = sym_validated.transform_pix(wout.pix)
            if isinstance(sym_validated, SymopArray):
                sym_arr = list(sym_validated)
            else:
                sym_arr = [sym_validated]
        wout.pix.recalc_data_range("all")

        existing_range = np.asarray(wout.data.axes.img_range, dtype=float).copy()
        exp_range = expand_box(existing_range[0, :3], existing_range[1, :3], nargout=1)
        cc_ranges = proj.transform_img_to_pix(exp_range)

        cc_exist_range = np.asarray(cc_ranges, dtype=float).copy()
        _, edges_ind = get_geometry(3)
        for op in reversed(sym_arr):
            if isinstance(op, SymopReflection):
                off = np.asarray(op.offset, dtype=float).reshape(3, 1)
                u_vec = np.asarray(op.u, dtype=float).reshape(3, 1)
                v_vec = np.asarray(op.v, dtype=float).reshape(3, 1)
                plane_points = off + np.hstack([np.zeros((3, 1), dtype=float), u_vec, v_vec])
                plane_points_cc = proj.transform_hkl_to_pix(plane_points)
                cross_points = sqw._box_intersect_3d(cc_ranges, plane_points_cc, edges_ind)
                if cross_points.size > 0:
                    cc_exist_range = np.hstack([cc_exist_range, cross_points])
            else:
                rot_center = proj.transform_hkl_to_pix(
                    np.asarray(op.offset, dtype=float).reshape(3, 1)
                )
                cc_exist_range = np.hstack([cc_exist_range, rot_center])

        for op in reversed(sym_arr):
            cc_exist_range = op.transform_pix(cc_exist_range, 1e-7)

        img_box_points = proj.transform_pix_to_img(cc_exist_range)
        img_db_range_minmax = np.vstack(
            [np.min(img_box_points, axis=1), np.max(img_box_points, axis=1)]
        )
        all_sym_range = np.hstack([img_db_range_minmax, existing_range[:, 3:4]])

        dat = wout.data.copy()
        dat.axes.img_range = all_sym_range
        targ_dims = tuple(int(x) for x in dat.axes.dims_as_ssize)
        s = np.zeros(targ_dims, dtype=np.float64, order="F")
        e = np.zeros(targ_dims, dtype=np.float64, order="F")
        npix = np.zeros(targ_dims, dtype=np.float64, order="F")

        npix, s, e, pix_binned, _ = proj.bin_pixels(
            dat.axes, wout.pix, npix, s, e, "-nomex", nargout=5
        )
        s, e = normalize_signal(s, e, npix)
        dat.s = s
        dat.e = e
        dat.npix = npix
        wout.data = dat
        wout.pix = pix_binned
        return wout

    @staticmethod
    def _box_intersect_3d(
        box_points: np.ndarray, plane_points: np.ndarray, edges_ind: np.ndarray
    ) -> np.ndarray:
        """Compute intersection points of a 3D box's 12 edges with a plane defined by 3 points.

        Ported from ``herbert_core/utilities/geometry/box_intersect.m``.
        """
        p0 = np.asarray(plane_points[:, 0], dtype=float)
        u = np.asarray(plane_points[:, 1], dtype=float) - p0
        v = np.asarray(plane_points[:, 2], dtype=float) - p0
        n = np.cross(u, v)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            return np.zeros((3, 0), dtype=float)
        n = n / n_norm

        cross_pts: List[np.ndarray] = []
        for e_idx in range(edges_ind.shape[1]):
            i1 = int(edges_ind[0, e_idx]) - 1
            i2 = int(edges_ind[1, e_idx]) - 1
            p1 = box_points[:, i1]
            p2 = box_points[:, i2]
            d1 = float(np.dot(p1 - p0, n))
            d2 = float(np.dot(p2 - p0, n))
            if abs(d1) <= 1e-9 and abs(d2) <= 1e-9:
                cross_pts.append(p1.copy())
                cross_pts.append(p2.copy())
            elif abs(d1) <= 1e-9:
                cross_pts.append(p1.copy())
            elif abs(d2) <= 1e-9:
                cross_pts.append(p2.copy())
            elif d1 * d2 < 0.0:
                t = d1 / (d1 - d2)
                cross_pts.append(p1 + t * (p2 - p1))

        if not cross_pts:
            return np.zeros((3, 0), dtype=float)
        arr = np.column_stack(cross_pts)
        _, uniq_idx = np.unique(np.round(arr, decimals=9), axis=1, return_index=True)
        return arr[:, np.sort(uniq_idx)]

    def testgateway(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/testgateway.m``."""
        # TODO(port-dependency): Port @sqw/testgateway.m
        raise NotImplementedError("sqw.testgateway is not yet implemented.")

    def tobyfit(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/tobyfit.m``."""
        # TODO(port-dependency): Port @sqw/tobyfit.m
        raise NotImplementedError("sqw.tobyfit is not yet implemented.")

    def to_cut(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/to_cut.m``."""
        # TODO(port-dependency): Port @sqw/to_cut.m
        raise NotImplementedError("sqw.to_cut is not yet implemented.")

    def value(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/value.m``."""
        # TODO(port-dependency): Port @sqw/value.m
        raise NotImplementedError("sqw.value is not yet implemented.")

    def write_nsqw_to_nsqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/write_nsqw_to_nsqw.m``."""
        # TODO(port-dependency): Port @sqw/write_nsqw_to_nsqw.m
        raise NotImplementedError("sqw.write_nsqw_to_nsqw is not yet implemented.")

    def write_qspec_to_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/write_qspec_to_sqw.m``."""
        # TODO(port-dependency): Port @sqw/write_qspec_to_sqw.m
        raise NotImplementedError("sqw.write_qspec_to_sqw is not yet implemented.")

    def xydata(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/xydata.m``."""
        # TODO(port-dependency): Port @sqw/xydata.m
        raise NotImplementedError("sqw.xydata is not yet implemented.")


    def apply(
        self,
        func_handle: Any,
        args: Any = None,
        recompute_bins: bool = True,
        compute_variance: bool = False,
    ) -> "sqw":
        """Apply a function (``func_handle``) to pixels of this ``sqw`` object and recompute the DnD image if requested.

        Ported from ``horace_core/sqw/@sqw/apply.m``.
        """
        from pyhorace.coord_transf.page_operations.page_op_apply import PageOp_apply

        if args is None or (isinstance(args, (list, tuple)) and len(args) == 0):
            args = [[]]
        page_op = PageOp_apply()
        page_op = page_op.init(
            self, func_handle, args, bool(compute_variance), bool(recompute_bins)
        )
        return self.apply_op(page_op)

    def func_eval(
        self,
        func_handle: Callable[..., Any],
        pars: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Evaluate a function at the plotting bin centres of an ``sqw`` object.

        Ported from ``horace_core/sqw/@sqw/func_eval.m``.
        """
        from pyhorace.sqw._sqw_methods import func_eval_sqw

        return func_eval_sqw(self, func_handle, pars, *args, **kwargs)

    def sqw_eval(
        self,
        sqwfunc: Callable[..., Any],
        pars: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Calculate sqw for a model scattering function over input sqw object.

        Ported from ``horace_core/sqw/@SQWDnDBase/sqw_eval.m``.
        """
        from pyhorace.sqw._sqw_methods import sqw_eval_method

        return sqw_eval_method(self, sqwfunc, pars, *args, **kwargs)

    def sqw_eval_pix(
        self,
        sqwfunc: Callable[..., Any],
        pars: Any,
        options: Union[Dict[str, Any], bool],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Helper function for ``sqw_eval`` executed on a full ``sqw`` object containing pixels.

        Ported from ``horace_core/sqw/@sqw/sqw_eval_pix.m``.
        """
        from pyhorace.sqw._sqw_methods import sqw_eval_pix_method

        return sqw_eval_pix_method(self, sqwfunc, pars, options, *args, **kwargs)

    def disp2sqw_eval(
        self,
        dispreln: Callable[..., Any],
        pars: Any,
        fwhh: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Calculate sqw for a model dispersion relation.

        Ported from ``horace_core/sqw/@SQWDnDBase/disp2sqw_eval.m``.
        """
        from pyhorace.sqw._sqw_methods import disp2sqw_eval_method

        return disp2sqw_eval_method(self, dispreln, pars, fwhh, *args, **kwargs)

    def equal_to_tol(
        self,
        other: Any,
        tol: Any = (0.0, 0.0),
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """Test equality of two ``sqw`` objects within tolerance.

        Ported from ``horace_core/sqw/@sqw/equal_to_tol_single.m``.
        """
        from pyhorace.core.serializable import compare_values

        name_a = str(kwargs.get("name_a", "input_1"))
        name_b = str(kwargs.get("name_b", "input_2"))
        nan_equal = bool(kwargs.get("nan_equal", True))
        ignore_str = bool(kwargs.get("ignore_str", False))
        ignore_date = bool(kwargs.get("ignore_date", False))
        reorder = bool(kwargs.get("reorder", True))
        fraction = float(kwargs.get("fraction", 1.0))

        if isinstance(tol, str):
            args = (tol, *args)
            tol = (0.0, 0.0)

        i = 0
        while i < len(args):
            a = args[i]
            if (
                isinstance(a, (int, float, np.number, list, tuple, np.ndarray))
                and not isinstance(a, bool)
                and i == 0
                and (tol == 0.0 or tol == (0.0, 0.0))
            ):
                tol = a
                i += 1
                continue
            if isinstance(a, str):
                low = a.lower().lstrip("-")
                if low == "ignore_date":
                    if i + 1 < len(args) and isinstance(args[i + 1], (bool, np.bool_)):
                        ignore_date = bool(args[i + 1])
                        i += 2
                    else:
                        ignore_date = True
                        i += 1
                    continue
                if low == "ignore_str":
                    if i + 1 < len(args) and isinstance(args[i + 1], (bool, np.bool_)):
                        ignore_str = bool(args[i + 1])
                        i += 2
                    else:
                        ignore_str = True
                        i += 1
                    continue
                if low == "nan_equal" and i + 1 < len(args):
                    nan_equal = bool(args[i + 1])
                    i += 2
                    continue
                if low == "reorder" and i + 1 < len(args):
                    reorder = bool(args[i + 1])
                    i += 2
                    continue
                if low == "fraction" and i + 1 < len(args):
                    fraction = float(args[i + 1])
                    i += 2
                    continue
                if low == "name_a" and i + 1 < len(args):
                    name_a = str(args[i + 1])
                    i += 2
                    continue
                if low == "name_b" and i + 1 < len(args):
                    name_b = str(args[i + 1])
                    i += 2
                    continue
            i += 1

        if type(self) is not type(other):
            return (
                False,
                f'Objects have different types. "{name_a}" has class: "{type(self).__name__}" and "{name_b}" class: "{type(other).__name__}"',
            )

        for field_name in ("main_header", "detpar", "experiment_info", "data"):
            tmp1 = getattr(self, field_name, None)
            tmp2 = getattr(other, field_name, None)
            if ignore_date:
                if field_name == "main_header" and tmp1 is not None and tmp2 is not None:
                    tmp1 = copy.deepcopy(tmp1)
                    tmp1.creation_date = tmp2.creation_date
                    tmp1.creation_date_defined_privately = (
                        tmp2.creation_date_defined_privately
                    )
                elif field_name == "data" and tmp1 is not None and tmp2 is not None:
                    tmp1 = tmp1.copy()
                    tmp1.creation_date = tmp2.creation_date
            ok, mess = compare_values(
                tmp1,
                tmp2,
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=f"{name_a}.{field_name}",
                name_b=f"{name_b}.{field_name}",
            )
            if not ok:
                return False, mess

        npix_arg = self.data.npix.ravel(order="F") if reorder else None
        return self.pix.equal_to_tol(
            other.pix,
            tol=tol,
            nan_equal=nan_equal,
            ignore_str=ignore_str,
            reorder=reorder,
            npix=npix_arg,
            fraction=fraction,
            name_a=f"{name_a}.pix",
            name_b=f"{name_b}.pix",
        )

    def sigvar_set(self, sigvar_obj: Any) -> "sqw":
        """Set output object signal and variance fields from input ``sigvar`` object.

        Ported from ``horace_core/sqw/@sqw/sigvar_set.m``.
        """
        w = self.copy()
        w.data = self.data.copy().sigvar_set(sigvar_obj)
        if w.has_pixels():
            from pyhorace.coord_transf.page_operations.page_op_sigvar_set import (
                PageOp_sigvar_set,
            )

            page_op = PageOp_sigvar_set()
            page_op = page_op.init(w)
            w = sqw.apply_op(w, page_op)
        return w

    def noisify(self, *args: Any, **kwargs: Any) -> "sqw":
        """Add random noise to the signal(s) of an ``sqw`` object.

        Ported from ``horace_core/sqw/@sqw/noisify.m``.
        """
        wout = self.copy()
        if wout.has_pixels():
            from pyhorace.coord_transf.page_operations.page_op_noisify import (
                PageOp_noisify,
            )

            page_op = PageOp_noisify()
            page_op = page_op.init(wout, *args, **kwargs)
            if not page_op.noisify_par["is_poisson"]:
                if not wout.pix.is_range_valid("signal"):
                    is_old_file = bool(getattr(wout.pix, "old_file_format", False))
                    page_op.print_range_warning(wout.full_filename, is_old_file)
                    wout = wout.recompute_bin_data()
                sig_col = int(PixelDataBase.field_index("signal")[0]) - 1
                max_sig = float(wout.pix.data_range[1, sig_col])
                page_op.noisify_par["ymax"] = max_sig
            wout = sqw.apply_op(wout, page_op)
        else:
            wout.data = self.data.noisify(*args, **kwargs)
        return wout

    def mask(self, keep_array: Any = None) -> "sqw":
        """Remove the bins indicated by the mask array.

        Ported from ``horace_core/sqw/@sqw/mask.m``.
        """
        if keep_array is None or (
            isinstance(keep_array, (list, tuple, np.ndarray))
            and np.asarray(keep_array).size == 0
        ):
            return self.copy()
        if self.has_pixels():
            return self.mask_pixels(keep_array, True)
        wout = self.copy()
        wout.data = wout.data.mask(keep_array)
        return wout

    def mask_pixels(
        self, keep_info: Any = None, mask_by_bins: bool = False
    ) -> "sqw":
        """Remove the pixels indicated by the mask array or count.

        Ported from ``horace_core/sqw/@sqw/mask_pixels.m``.
        """
        if not self.has_pixels():
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Can mask pixels only in an sqw-type object",
            )
        if keep_info is None or (
            isinstance(keep_info, (list, tuple, np.ndarray))
            and np.asarray(keep_info).size == 0
        ):
            return self.copy()

        from pyhorace.coord_transf.page_operations.page_op_mask import PageOp_mask
        from pyhorace.sqw.sqw_dnd_base import SQWDnDBase

        if isinstance(keep_info, SQWDnDBase):
            if not keep_info.has_pixels():
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    "If the mask object is a Horace object if must be sqw-type i.e. contain pixel information",
                )
            if self.pix.num_pixels != keep_info.pix.num_pixels:
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    "Number of pixels in input and mask object must be the same",
                )
        elif isinstance(
            keep_info, (int, float, np.number, bool, np.bool_, list, tuple, np.ndarray)
        ) and not isinstance(keep_info, (str, bytes)):
            ki_arr = np.asarray(keep_info)
            is_scalar_num = (
                ki_arr.size == 1
                and not isinstance(keep_info, (bool, np.bool_))
                and not np.issubdtype(ki_arr.dtype, np.bool_)
            )
            num_bins = int(np.asarray(self.data.npix).size)
            if ki_arr.size == num_bins and not is_scalar_num:
                keep_info = ki_arr.astype(bool)
                mask_by_bins = True
            elif ki_arr.size == self.num_pixels or is_scalar_num:
                if is_scalar_num:
                    num_keep = int(ki_arr.item())
                else:
                    num_keep = int(np.sum(ki_arr.astype(bool)))
                if num_keep <= 0:
                    raise HoraceError(
                        "HORACE:sqw:invalid_argument",
                        f"Can not mask all pixels. Num pixels to keep should be positive number. Provided: {num_keep}",
                    )
                if num_keep > self.num_pixels:
                    raise HoraceError(
                        "HORACE:sqw:invalid_argument",
                        f"Can not keep more pixels then available. Num pixels to keep should be smaller or equal to num_pixels ({self.num_pixels}). Provided: {num_keep}",
                    )
                if mask_by_bins:
                    raise HoraceError(
                        "HORACE:sqw:invalid_argument",
                        f"Mask by bins requested but the size of mask ({ki_arr.size}) is equal to size of pixel array instead of npix array ({num_bins})",
                    )
            else:
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"Mask array should define either pix mask with numel {self.num_pixels} or bin mask with numel {num_bins}. Its numel : {ki_arr.size}",
                )
            if not is_scalar_num:
                keep_info = np.asarray(keep_info, dtype=bool)
        else:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Mask must provide a numeric or logical array with same number of elements as the data or an SQW with the same projection and axes",
            )

        page_op = PageOp_mask()
        page_op = page_op.init(self, keep_info)
        if mask_by_bins:
            page_op.mask_by_bins = True
        return sqw.apply_op(self, page_op)

    def mask_random_pixels(self, npix: Any = None) -> Any:
        """Reduce the number of pixels randomly in an ``sqw`` object or sequence of ``sqw`` objects.

        Ported from ``horace_core/sqw/@sqw/mask_random_pixels.m``.
        """
        if isinstance(self, (list, tuple, np.ndarray)):
            objs = list(self)
            is_seq = True
        else:
            objs = [self]
            is_seq = False

        if not all(o.has_pixels() for o in objs):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Can not mask random pixels on sqw object not containing any pixels",
            )

        sz = len(objs)
        npix_arr = np.floor(np.asarray(npix, dtype=np.float64) + 0.5).astype(np.int64).ravel()
        if npix_arr.size == 1 and sz > 1:
            npix_arr = np.repeat(npix_arr, sz)

        if npix_arr.size != sz:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "npix must either be scalar or an array of the same size as input sqw object",
            )

        if np.any(npix_arr <= 0):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Cannot mask every pixel",
            )

        if any(objs[i].num_pixels < int(npix_arr[i]) for i in range(sz)):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Cannot retain greater number of pixels than data contains",
            )

        wout = [objs[i].mask_pixels(int(npix_arr[i])) for i in range(sz)]
        if not is_seq:
            return wout[0]
        return DataOpArray(wout) if isinstance(self, DataOpArray) else wout

    def mask_random_fraction_pixels(self, npix_frac: Any = None) -> Any:
        """Reduce the number of pixels randomly to a fraction ``npix_frac`` in an ``sqw`` object.

        Ported from ``horace_core/sqw/@sqw/mask_random_fraction_pixels.m``.
        """
        if isinstance(self, (list, tuple, np.ndarray)):
            objs = list(self)
            is_seq = True
        else:
            objs = [self]
            is_seq = False

        sz = len(objs)
        frac_arr = np.asarray(npix_frac, dtype=np.float64).ravel()
        if frac_arr.size == 1 and sz > 1:
            frac_arr = np.repeat(frac_arr, sz)

        if frac_arr.size != sz:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "npix must either be scalar or an array of the same size as input sqw object",
            )

        npix_keep = np.zeros(sz, dtype=np.int64)
        for i in range(sz):
            fval = float(frac_arr[i])
            if fval <= 0.0:
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    "Cannot mask every pixel, or have negative fraction of pixels retained",
                )
            if fval > 1.0:
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    "Can not have pixel fraction larger then 1",
                )
            n_keep = int(np.floor(objs[i].num_pixels * fval + 0.5))
            if n_keep < 1:
                n_keep = 1
            npix_keep[i] = n_keep

        return sqw.mask_random_pixels(self, npix_keep if is_seq else int(npix_keep[0]))

    def slim(self, reduce: Any = None) -> Any:
        """Slim an ``sqw`` object by removing random pixels by factor ``reduce``.

        Ported from ``horace_core/sqw/@sqw/slim.m``.
        """
        red_arr = np.asarray(reduce, dtype=np.float64)
        if np.any(red_arr < 1.0):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"reduce factor has to be greater than or equal to 1. It is {reduce}",
            )
        return sqw.mask_random_fraction_pixels(self, 1.0 / red_arr)

    def coordinates_calc(self, name: Any = None) -> Any:
        """Replace the ``sqw`` signal and variance data with coordinate values.

        Ported from ``horace_core/sqw/@sqw/coordinates_calc.m``.
        """
        from pyhorace.coord_transf.page_operations.page_op_coord_calc import (
            PageOp_coord_calc,
        )

        if name is None:
            raise HoraceError(
                "HORACE:coordinates_calc:invalid_argument",
                "this function requires two input arguments",
            )
        if not isinstance(name, str) or len(name) == 0:
            raise HoraceError(
                "HORACE:coordinates_calc:invalid_argument",
                f"Second argument of this function should be char string selected from input list. "
                f"Its type is {type(name).__name__}",
            )

        page_op = PageOp_coord_calc()
        if name not in page_op.xname:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Input parameter must be one or group of parameters from list {page_op.xname}\n"
                f". It is: {name}",
            )
        ind = page_op.xname.index(name) + 1

        is_seq = isinstance(self, (list, tuple, np.ndarray))
        in_objs = list(np.asarray(self, dtype=object).ravel(order="F")) if is_seq else [self]
        wout = [o.copy() for o in in_objs]

        invalid_dim = [ind <= 4 and ind > w.NUM_DIMS for w in wout]
        if any(invalid_dim):
            invalid_nums = [idx + 1 for idx, inv in enumerate(invalid_dim) if inv]
            invalid_dims = [w.NUM_DIMS for w, inv in zip(wout, invalid_dim) if inv]
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Coordinate name {page_op.xname[ind - 1]} for object(s) N:{invalid_nums} "
                f"request axes numbers higher then the dimensionality of sqw object(s): {invalid_dims}",
            )

        for i in range(len(wout)):
            page_op = page_op.init(wout[i], ind)
            wout[i] = sqw.apply_op(wout[i], page_op)

        if not is_seq:
            return wout[0]
        if isinstance(self, np.ndarray):
            out = np.empty(len(wout), dtype=object)
            for idx, item in enumerate(wout):
                out[idx] = item
            return out.reshape(self.shape, order="F")
        return wout

    def signal(self, xname: Any = None) -> Any:
        """Set the signal of an ``sqw`` object to the values for the specified coordinate.

        Ported from ``horace_core/sqw/@sqw/signal.m``.
        """
        return sqw.coordinates_calc(self, xname)

    def compact(self) -> Any:
        """Squeeze the data range in the dnd image of an ``sqw`` object to eliminate empty border bins.

        Ported from ``horace_core/sqw/@sqw/compact.m``.
        """
        from pyhorace.core.data_op_interface import DataOpArray

        is_seq = isinstance(self, (list, tuple, np.ndarray, DataOpArray))
        in_objs = list(np.asarray(self, dtype=object).ravel(order="F")) if is_seq else [self]
        wout = [o.copy() for o in in_objs]
        for n in range(len(in_objs)):
            wout[n]._data = in_objs[n]._data.compact()
        if not is_seq:
            return wout[0]
        if isinstance(self, np.ndarray):
            out = np.empty(len(wout), dtype=object)
            for idx, item in enumerate(wout):
                out[idx] = item
            return out.reshape(self.shape, order="F")
        if isinstance(self, DataOpArray):
            return DataOpArray(wout)
        return wout

    def section(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Take a section out of an ``sqw`` object along its existing plot axes.

        Ported from ``horace_core/sqw/@sqw/section.m``.
        """
        from pyhorace.coord_transf.bmatrix import _infer_nargout
        from pyhorace.coord_transf.page_operations.page_op_section import PageOp_section
        from pyhorace.core.data_op_interface import DataOpArray

        if nargout is None:
            nargout = _infer_nargout(default=1)

        is_seq = isinstance(self, (list, tuple, np.ndarray, DataOpArray))
        in_objs = list(np.asarray(self, dtype=object).ravel(order="F")) if is_seq else [self]
        wout = [o.copy() for o in in_objs]

        if len(args) == 0:
            res = wout[0] if not is_seq else (DataOpArray(wout) if isinstance(self, DataOpArray) else wout)
            return (res, np.empty((0, 0), dtype=int)) if nargout == 2 else res

        ndim = int(in_objs[0].dimensions)
        if ndim == 0:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Cannot section a zero dimensional object",
            )
        if len(in_objs) > 1 and any(int(o.dimensions) != ndim for o in in_objs[1:]):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "All objects must have same dimensionality for sectioning to work",
            )
        if len(args) != ndim:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Check number of arguments",
            )

        page_op = PageOp_section()
        irange = np.empty((0, 0), dtype=int)
        for n in range(len(in_objs)):
            img, irange = wout[n].data.section(*args, nargout=2)
            if bool(in_objs[n].has_pixels):
                page_op = page_op.init(wout[n], img, irange)
                wout[n] = sqw.apply_op(wout[n], page_op)
            else:
                wout[n].data = img

        if not is_seq:
            res = wout[0]
        elif isinstance(self, np.ndarray):
            out = np.empty(len(wout), dtype=object)
            for idx, item in enumerate(wout):
                out[idx] = item
            res = out.reshape(self.shape, order="F")
        elif isinstance(self, DataOpArray):
            res = DataOpArray(wout)
        else:
            res = wout

        if nargout == 2:
            return res, irange
        return res

    def replicate(self, wref: Any, *args: Any) -> Any:
        """Make a higher-dimensional dataset from a lower-dimensional dataset by replicating along extra axes.

        Ported from ``horace_core/sqw/@sqw/replicate.m``.
        """
        import warnings
        from pyhorace.coord_transf.page_operations.page_op_sigvar_set import (
            PageOp_sigvar_set,
        )
        from pyhorace.core.data_op_interface import DataOpArray

        is_seq = isinstance(self, (list, tuple, np.ndarray, DataOpArray))
        in_objs = list(np.asarray(self, dtype=object).ravel(order="F")) if is_seq else [self]

        ndim = int(in_objs[0].dimensions)
        for o in in_objs[1:]:
            if int(o.dimensions) != ndim:
                raise HoraceError(
                    "HORACE:replicate:invalid_argument",
                    "Check first input argument - an array of sqw objects must all have the same dimensionality",
                )

        set_pix = False
        for a in args:
            if isinstance(a, str) and a.lower().startswith("-set_pix"):
                set_pix = True
            else:
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"Unrecognized option to replicate: {a!r}",
                )

        if not isinstance(wref, (sqw, DnDBase)):
            raise HoraceError(
                "HORACE:replicate:invalid_argument",
                f"The second argument must be a scalar dnd or sqw object, actually its type is {type(wref).__name__}",
            )

        wref_dnd_type = wref.data if isinstance(wref, sqw) else wref
        wref_has_pix = bool(isinstance(wref, sqw) and bool(wref.has_pixels))

        wout: List[Any] = []
        for i in range(len(in_objs)):
            dnd_obj = in_objs[i].dnd() if isinstance(in_objs[i], sqw) else in_objs[i]
            rep_obj = dnd_obj.replicate(wref_dnd_type)
            if set_pix:
                if wref_has_pix:
                    w_item = wref.copy()
                    w_item.data.s = rep_obj.s
                    w_item.data.e = rep_obj.e
                    page_op = PageOp_sigvar_set()
                    page_op.in_replicate = True
                    page_op = page_op.init(w_item)
                    w_item = sqw.apply_op(w_item, page_op)
                    wout.append(w_item)
                else:
                    if isinstance(wref, DnDBase):
                        explanation = "sqw_ref_object is a dnd object"
                    else:
                        explanation = "sqw_ref_object is sqw object without pixels"
                    warnings.warn(
                        f"-set_pix key is provided as input of replicate(sqw_rep,sqw_ref_object,\"-set_pix\"), but {explanation}",
                        UserWarning,
                        stacklevel=2,
                    )
                    wout.append(rep_obj)
            else:
                wout.append(rep_obj)

        if not is_seq:
            return wout[0]
        if isinstance(self, np.ndarray):
            out = np.empty(len(wout), dtype=object)
            for idx, item in enumerate(wout):
                out[idx] = item
            return out.reshape(self.shape, order="F")
        if isinstance(self, DataOpArray):
            return DataOpArray(wout)
        return wout

    @property
    def img_size_bytes(self) -> int:
        """Return memory footprint (in bytes) of the s, e, npix image arrays."""
        return self._data.img_size_bytes

    def get_se_npix(
        self, start_pos: int = 1, end_pos: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return slices of s, e, npix arrays (1-based inclusive indices)."""
        return self._data.get_se_npix(start_pos, end_pos)

    def get_npix_block(
        self, start_pos: int = 1, end_pos: Optional[int] = None
    ) -> np.ndarray:
        """Return slice of npix array (1-based inclusive indices)."""
        return self._data.get_npix_block(start_pos, end_pos)

    def get_dnd_metadata(self) -> Any:
        """Return dnd_metadata from the underlying DnD object."""
        return self._data.metadata

    def split(self, *args: Any, **kwargs: Any) -> Any:
        """Split an ``sqw`` object into an array of ``sqw`` objects (or files), one per contributing run.

        Ported from ``horace_core/sqw/@sqw/split.m``.
        """
        from pyhorace.coord_transf.page_operations.page_op_split_sqw import (
            PageOp_split_sqw,
        )
        from pyhorace.core.configurations.hpc_config import hpc_config
        from pyhorace.core.data_op_interface import DataOpArray

        if isinstance(self, (list, tuple, np.ndarray, DataOpArray)):
            if len(self) > 1:
                raise HoraceError(
                    "HORACE:split:not_implemented",
                    "split currently works with only one sqw object",
                )
            self = self[0]

        return_files = False
        split_filebacked = False
        argi: List[Any] = []
        for a in args:
            if isinstance(a, str) and a.startswith("-"):
                a_low = a.lower()
                if "-files".startswith(a_low) and len(a_low) >= 5 and a_low != "-filebacked"[:len(a_low)]:
                    return_files = True
                elif a_low == "-files":
                    return_files = True
                elif "-filebacked".startswith(a_low):
                    split_filebacked = True
                else:
                    raise HoraceError(
                        "HORACE:split:invalid_arguments",
                        f"Unrecognised option: {a}",
                    )
            else:
                argi.append(a)

        if len(argi) > 0:
            folder_for_parts = str(argi[0])
            if not os.path.isdir(folder_for_parts):
                try:
                    os.makedirs(folder_for_parts, exist_ok=True)
                except Exception as exc:
                    raise HoraceError(
                        "HORACE:split:invalid_argument",
                        f"folder {folder_for_parts} does not exist and can not be created. Reason: {exc}",
                    ) from exc
        else:
            folder_for_parts = ""

        nfiles = int(self.main_header.nfiles)
        if nfiles == 1:
            return self.copy()

        if self.pix.is_filebacked and (return_files or split_filebacked):
            self.pix.keep_precision = True

        split_img_size = self.img_size_bytes
        split_pix_size = int(self.pix.num_pixels) * int(self.pix.pix_byte_size)
        total_size = split_img_size * nfiles + split_pix_size

        mem_available = hpc_config().phys_mem_available

        page_op = PageOp_split_sqw()
        pix_filebacked = False
        img_filebacked = False
        if total_size > mem_available or split_filebacked or return_files:
            pix_filebacked = True
            img_filebacked = (split_img_size * nfiles > mem_available) or return_files

        page_op.outfile = folder_for_parts
        page_op = page_op.init(self, pix_filebacked, img_filebacked)
        page_op.init_filebacked_output = (
            bool(page_op.init_filebacked_output) or split_filebacked
        )
        if img_filebacked:
            page_op.init_filebacked_output = False
        wout = sqw.apply_op(self, page_op)
        return wout

    def run_inspector(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], int, Any]:
        """Inspect individual contributing runs of a 1D or 2D ``sqw`` object.

        Ported from ``horace_core/sqw/@sqw/run_inspector.m``.
        """
        from pyhorace.core.data_op_interface import DataOpArray

        if isinstance(self, (list, tuple, np.ndarray, DataOpArray)):
            if len(self) != 1:
                raise HoraceError(
                    "HORACE:run_inspector:invalid_argument",
                    "Input dataset must be a single sqw object",
                )
            self = self[0]
        if not isinstance(self, sqw):
            raise HoraceError(
                "HORACE:run_inspector:invalid_argument",
                "Input dataset must be a single sqw object",
            )

        all_args: List[Any] = list(args)
        for k, v in kwargs.items():
            all_args.extend([k, v])

        pr: Dict[str, Any] = {
            "obj": self,
            "col": np.empty(0, dtype=np.float64),
            "ax": np.empty(0, dtype=np.float64),
            "test_parser": False,
            "test_videofig": False,
        }

        short_par = ("ax", "col", "test_parser", "test_videofig")
        idx = 0
        while idx < len(all_args):
            raw_k = all_args[idx]
            if not isinstance(raw_k, str):
                raise HoraceError(
                    "HORACE:run_inspector:invalid_argument",
                    f"Expected parameter name string, got {type(raw_k).__name__}",
                )
            k_low = raw_k.lower()
            matched_k = None
            for sp in short_par:
                if k_low.startswith(sp):
                    matched_k = sp
                    break
            if matched_k is None:
                raise HoraceError(
                    "HORACE:run_inspector:invalid_argument",
                    f"Unrecognized parameter name: {raw_k}",
                )
            if idx + 1 >= len(all_args):
                raise HoraceError(
                    "HORACE:run_inspector:invalid_argument",
                    f"Parameter '{raw_k}' is missing a value",
                )
            val = all_args[idx + 1]
            idx += 2

            if matched_k == "col":
                val_arr = np.asarray(val, dtype=np.float64).ravel()
                if val_arr.size != 2 or not (val_arr[0] < val_arr[1]):
                    raise HoraceError(
                        "HORACE:run_inspector:invalid_argument",
                        "Invalid 'col' parameter: must be [c_lo, c_hi] with c_lo < c_hi",
                    )
                pr["col"] = val_arr
            elif matched_k == "ax":
                val_arr = np.asarray(val, dtype=np.float64).ravel()
                if (
                    val_arr.size != 4
                    or not (val_arr[0] < val_arr[1])
                    or not (val_arr[2] < val_arr[3])
                ):
                    raise HoraceError(
                        "HORACE:run_inspector:invalid_argument",
                        "Invalid 'ax' parameter: must be [x_lo, x_hi, y_lo, y_hi] with x_lo < x_hi and y_lo < y_hi",
                    )
                pr["ax"] = val_arr
            elif matched_k == "test_parser":
                if not isinstance(val, (bool, np.bool_)):
                    raise HoraceError(
                        "HORACE:run_inspector:invalid_argument",
                        "Invalid 'test_parser' parameter: must be logical",
                    )
                pr["test_parser"] = bool(val)
            elif matched_k == "test_videofig":
                if not isinstance(val, (bool, np.bool_)):
                    raise HoraceError(
                        "HORACE:run_inspector:invalid_argument",
                        "Invalid 'test_videofig' parameter: must be logical",
                    )
                pr["test_videofig"] = bool(val)

        nd = int(self.dimensions()[0]) if isinstance(self.dimensions(), tuple) else int(self.dimensions())
        if nd < 1 or nd > 2:
            raise HoraceError(
                "HORACE:run_inspector:invalid_argument",
                f"***  Input dataset must be an sqw object with 1 or 2-dimensions image.\n"
                f"     The image of the provided object has: {nd}-dimensions.",
            )

        if pr["test_parser"]:
            return pr, nd, []

        split_array = self.split()
        return pr, nd, split_array

    def join(self, *args: Any) -> Any:
        """Join an array or sequence of ``sqw`` objects or ``sqw`` files into a single ``sqw`` object.

        Ported from ``horace_core/sqw/@sqw/join.m``.
        """
        from pyhorace.algorithms.join_sqw import join_sqw

        return join_sqw(self, *args)

    def combine_sqw(self, *args: Any) -> Any:
        """Combine two or more ``sqw`` objects in order to improve statistics.

        Ported from ``horace_core/sqw/@sqw/combine_sqw.m``.
        """
        from pyhorace.coord_transf._dependencies import expand_box, range_add_border
        from pyhorace.coord_transf.line_axes import line_axes
        from pyhorace.core.configurations.hor_config import hor_config
        from pyhorace.core.data_op_interface import DataOpArray
        from pyhorace.sqw.d0d import d0d

        if len(args) == 0:
            if isinstance(self, (list, tuple, np.ndarray, DataOpArray)):
                inputs = list(np.asarray(self, dtype=object).ravel(order="F"))
            else:
                inputs = [self]
        elif isinstance(args[0], (list, tuple, DataOpArray)) or (
            isinstance(args[0], np.ndarray) and args[0].dtype == object
        ):
            if len(args) > 1:
                raise HoraceError(
                    "HORACE:combine_sqw:invalid_argument",
                    "if second argument is a cellarray of sqw objects, combine_sqw can only accept 2 arguments",
                )
            base_list = (
                list(np.asarray(self, dtype=object).ravel(order="F"))
                if isinstance(self, (list, tuple, np.ndarray, DataOpArray))
                else [self]
            )
            inputs = base_list + list(np.asarray(args[0], dtype=object).ravel(order="F"))
        else:
            base_list = (
                list(np.asarray(self, dtype=object).ravel(order="F"))
                if isinstance(self, (list, tuple, np.ndarray, DataOpArray))
                else [self]
            )
            inputs = base_list + list(args)

        right_type = [isinstance(x, sqw) for x in inputs]
        if not all(right_type):
            n_empty = len(right_type) - sum(right_type)
            raise HoraceError(
                "HORACE:combine_sqw:invalid_argument",
                f"Input objects must be sqw type with detector pixel information. "
                f"Input contains {n_empty} non-sqw objects",
            )

        inputs = [x for x in inputs if x.pix.num_pixels > 0]
        if len(inputs) == 0:
            return self
        if len(inputs) == 1:
            return inputs[0].copy()

        img_ranges = [np.asarray(x.data.img_range, dtype=np.float64) for x in inputs]
        full_img_rng = [expand_box(box[0, :], box[1, :]) for box in img_ranges]
        exper = [sq.experiment_info for sq in inputs]
        all_proj_block = [ws.data.proj for ws in inputs]
        full_pix_rng = [
            proj.transform_img_to_pix(data)
            for proj, data in zip(all_proj_block, full_img_rng)
        ]

        proj1 = all_proj_block[0]
        full_img_rng_1 = np.hstack(
            [proj1.transform_pix_to_img(data) for data in full_pix_rng]
        )
        combine_range = np.vstack(
            [np.min(full_img_rng_1, axis=1), np.max(full_img_rng_1, axis=1)]
        )
        combine_range = range_add_border(combine_range)

        new_range_arg: List[Any] = []
        nbins = np.asarray(inputs[0].data.axes.nbins_all_dims, dtype=int).ravel()
        paxis = nbins > 1
        for i in range(4):
            rng = combine_range[:, i].copy()
            if paxis[i]:
                np_i = int(nbins[i])
                dist = float(rng[1] - rng[0])
                step = dist / (np_i - 1) if np_i > 1 else dist
                new_range_arg.append([float(rng[0]), step, float(rng[1])])
            else:
                new_range_arg.append([float(rng[0]), float(rng[1])])

        wout = inputs[0].copy()
        pix_list = [x.pix for x in inputs]

        hc = hor_config()
        info_level = hc.log_level
        try:
            hc.log_level = -1
            wout.pix = PixelDataBase.cat(*pix_list)
        finally:
            hc.log_level = info_level

        wout.experiment_info = exper[0].combine_experiments(exper[1:], True, True)

        ax = line_axes(nbins_all_dims=np.ones(4, dtype=int), img_range=combine_range)
        wout.data = d0d(ax, proj1)
        wout.data.npix = np.full(
            wout.data.npix.shape, wout.pix.num_pixels, dtype=np.int64
        )
        wout = wout.cut(proj1, *new_range_arg)
        return wout


def symmetrise_sqw(win: Any, *args: Any, **kwargs: Any) -> sqw:
    """Symmetrise an ``sqw`` object or file according to ``Symop`` (port of ``symmetrise_sqw.m``)."""
    if isinstance(win, (str, os.PathLike)):
        obj = sqw(win, file_backed=False)
        return obj.symmetrise_sqw(*args, **kwargs)
    if isinstance(win, sqw):
        return win.symmetrise_sqw(*args, **kwargs)
    if isinstance(win, (list, tuple, np.ndarray)) or hasattr(win, "__len__"):
        if len(win) != 1:
            raise HoraceError(
                "HORACE:symmetrise_sqw:invalid_argument",
                "symmetrisation only implemented for single sqw object, not arrays of objects. Use a for-loop to deal with arrays",
            )
        first = win[0]
        if isinstance(first, sqw):
            return first.symmetrise_sqw(*args, **kwargs)
    raise HoraceError(
        "HORACE:symmetrise_sqw:invalid_argument",
        "input object must be sqw type with detector pixel information",
    )


def _init_sqw_obj_from_file_for_sqw_op(
    win: Any,
) -> Tuple[int, List[Any], List[bool]]:
    """Parse input sequence of files or ``sqw`` objects for ``sqw_op`` / ``sqw_op_bin_pixels``.

    Ported from ``horace_core/algorithms/private/init_sqw_obj_from_file_for_sqw_op_.m``.
    """
    from pyhorace.core.data_op_interface import DataOpArray
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory

    if isinstance(win, (str, os.PathLike, sqw)):
        win_list = [win]
    elif isinstance(win, (np.ndarray, DataOpArray)):
        win_list = list(np.asarray(win, dtype=object).ravel(order="F"))
    elif isinstance(win, (list, tuple)):
        win_list = list(win)
    else:
        win_list = [win]

    n_inputs = len(win_list)
    ldrs: List[Any] = []
    is_sqw_obj: List[bool] = []
    for i, item in enumerate(win_list):
        if isinstance(item, (str, os.PathLike)):
            ldr = sqw_formats_factory.instance().get_loader(str(item))
            if not bool(getattr(ldr, "sqw_type", False)):
                ldr.fclose()
                raise HoraceError(
                    "HORACE:algorithms:invalid_argument",
                    f"input file N:{i + 1} with name {item} is not an sqw-type file",
                )
            ldrs.append(ldr)
            is_sqw_obj.append(False)
        elif isinstance(item, sqw):
            ldrs.append(item)
            is_sqw_obj.append(True)
        else:
            raise HoraceError(
                "HORACE:algorithms:invalid_argument",
                f"Input object N:{i + 1} is not an sqw object. Its class is: {type(item).__name__}",
            )
    return n_inputs, ldrs, is_sqw_obj


def sqw_op(
    win: Any,
    sqwop_func: Any,
    pars: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Perform a paged operation over pixels defined by user-provided ``sqwop_func``.

    Ported from ``horace_core/sqw/@sqw/sqw_op.m``, ``horace_core/sqw/@sqw/private/sqw_op_single_.m``,
    and ``horace_core/algorithms/sqw_op.m``.
    """
    from pyhorace.coord_transf.page_operations.page_op_sqw_op import PageOp_sqw_op
    from pyhorace.core.data_op_interface import DataOpArray

    n_inputs, ldrs, is_sqw_obj = _init_sqw_obj_from_file_for_sqw_op(win)
    objs: List[sqw] = []
    for i in range(n_inputs):
        if is_sqw_obj[i]:
            objs.append(ldrs[i])
        else:
            obj_i = sqw(ldrs[i])
            if not obj_i.pix.is_filebacked and hasattr(ldrs[i], "fclose"):
                ldrs[i].fclose()
            objs.append(obj_i)

    sqwop_func_parsed, pars_cell, opts = objs[0].parse_eval_args(
        sqwop_func, pars, *args, **kwargs
    )
    if bool(opts.get("average", False)):
        raise HoraceError(
            "HORACE:PageOp_sqw_op:not_implemented",
            '"-average" option is not currently implemented for sqw_op.'
            "Contact HoraceHelp@stfc.ac.uk if you need it with the description of its meaning in your case",
        )
    if bool(opts.get("test_input_parsing", False)):
        return opts

    wout: List[Any] = []
    for i, obj_i in enumerate(objs):
        if opts.get("pageop_processor") is None:
            eval_op: Any = PageOp_sqw_op()
        else:
            eval_op = opts["pageop_processor"]
        outfiles = opts.get("outfile") or []
        outf_i = (
            outfiles[i]
            if i < len(outfiles)
            else (outfiles[0] if len(outfiles) > 0 else "")
        )
        if outf_i:
            eval_op.outfile = outf_i
        if bool(opts.get("filebacked", False)):
            eval_op.init_filebacked_output = True
        eval_op = eval_op.init(obj_i, sqwop_func_parsed, pars_cell, opts)
        res_i = sqw.apply_op(obj_i, eval_op)
        if not is_sqw_obj[i] and hasattr(ldrs[i], "fclose"):
            ldrs[i].fclose()
        wout.append(res_i)

    return wout[0] if len(wout) == 1 else DataOpArray(wout)


def sqw_op_bin_pixels(
    win: Any,
    sqwop_func: Any,
    pars: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Perform a paged operation over pixels and rebin into target projection/axes.

    Ported from ``horace_core/sqw/@sqw/sqw_op_bin_pixels.m``,
    ``horace_core/sqw/@sqw/private/sqw_op_bin_pix_single_.m``,
    and ``horace_core/algorithms/sqw_op_bin_pixels.m``.
    """
    from pyhorace.coord_transf.page_operations.page_op_sqw_binning import (
        PageOp_sqw_binning,
    )
    from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval
    from pyhorace.coord_transf.symop import SymopIdentity
    from pyhorace.core.data_op_interface import DataOpArray
    from pyhorace.data.pixels.pixobj_combine_info import pixobj_combine_info

    n_inputs, ldrs, is_sqw_obj = _init_sqw_obj_from_file_for_sqw_op(win)

    cut_argi: List[Any] = []
    eval_argi: List[Any] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if isinstance(arg, PageOp_sqw_eval):
            eval_argi.append(arg)
            idx += 1
        elif isinstance(arg, str):
            if arg == "":
                cut_argi.append([])
                idx += 1
            elif arg.startswith("-"):
                eval_argi.append(arg)
                idx += 1
            else:
                eval_argi.append(arg)
                if idx + 1 < len(args):
                    eval_argi.append(args[idx + 1])
                    idx += 2
                else:
                    idx += 1
        else:
            cut_argi.append(arg)
            idx += 1

    first_obj = ldrs[0] if is_sqw_obj[0] else sqw(ldrs[0])
    num_dims = int(first_obj.data.NUM_DIMS)
    if len(cut_argi) < num_dims:
        cut_argi = [[] for _ in range(num_dims)] + cut_argi

    targ_proj, pbin_list, sym, opt = SQWDnDBase.process_and_validate_cut_inputs(
        first_obj, True, *cut_argi, nargout=4
    )
    if len(sym) > 1 or not isinstance(sym[0], SymopIdentity):
        raise HoraceError(
            "HORACE:sqw_op_bin_pixels:not_implemented",
            "sqw_op_bin_pixels does not yet accepts Symop parameters.\n"
            "Add necessary SymOp to custom operation yourself",
        )

    targ_ax_block, targ_proj = first_obj.define_target_axes_block(
        targ_proj, pbin_list[0], sym, nargout=2
    )

    sqwop_func_parsed, pars_cell, opts = first_obj.parse_eval_args(
        sqwop_func, pars, *eval_argi, **kwargs
    )
    if not bool(opt.get("keep_pix", True)):
        opts["nopix"] = True
    for k, v in opt.items():
        if k not in ("keep_pix", "outfile"):
            opts[k] = v

    if bool(opts.get("average", False)):
        raise HoraceError(
            "HORACE:sqw_op_bin_pixels:not_implemented",
            '"-average" option is not currently implemented for sqw_op_bin_pixels.'
            "Contact HoraceHelp@stfc.ac.uk if you really need it implemented which the explanation of its meaning in your case",
        )

    objs: List[sqw] = [first_obj]
    for i in range(1, n_inputs):
        if is_sqw_obj[i]:
            objs.append(ldrs[i])
        else:
            objs.append(sqw(ldrs[i]))

    if bool(opts.get("combine", False)):
        w_comb = objs[0].copy() if is_sqw_obj[0] else objs[0]
        if n_inputs > 1:
            pix_list = [w_comb.pix] + [o.pix for o in objs[1:]]
            npix_arr = np.array([p.num_pixels for p in pix_list], dtype=np.int64)
            exper_list = [w_comb.experiment_info] + [o.experiment_info for o in objs[1:]]
            pix_all = pixobj_combine_info(pix_list, [int(x) for x in npix_arr])
            w_comb.data.do_check_combo_arg = False
            w_comb.data.npix = npix_arr.reshape(1, -1)
            w_comb.data.s = np.zeros((1, n_inputs), dtype=np.float64)
            w_comb.data.e = np.zeros((1, n_inputs), dtype=np.float64)
            w_comb.experiment_info = exper_list[0].combine_experiments(
                exper_list[1:], True, True
            )
            w_comb._do_check_combo_arg = False
            w_comb.pix = pix_all
            w_comb._do_check_combo_arg = True
            w_comb.data.do_check_combo_arg = True
        win_to_process = [w_comb]
    else:
        win_to_process = [
            objs[i].copy() if is_sqw_obj[i] else objs[i] for i in range(n_inputs)
        ]

    if bool(opts.get("test_input_parsing", False)):
        opts["targ_proj"] = targ_proj
        opts["targ_ax_block"] = targ_ax_block
        opts["input_objects"] = (
            win_to_process[0] if len(win_to_process) == 1 else win_to_process
        )
        return opts

    wout: List[Any] = []
    for i, w_i in enumerate(win_to_process):
        if opts.get("pageop_processor") is None:
            rebin_op: Any = PageOp_sqw_binning()
        else:
            rebin_op = opts["pageop_processor"]
        outfiles = opts.get("outfile") or []
        outf_i = (
            outfiles[i]
            if i < len(outfiles)
            else (outfiles[0] if len(outfiles) > 0 else "")
        )
        if outf_i:
            rebin_op.outfile = outf_i
        if bool(opts.get("filebacked", False)) and not bool(opts.get("nopix", False)):
            rebin_op.init_filebacked_output = True
        rebin_op = rebin_op.init(
            w_i, sqwop_func_parsed, pars_cell, targ_ax_block, targ_proj, opts
        )
        res_i = sqw.apply_op(w_i, rebin_op)
        if isinstance(w_i.pix, pixobj_combine_info):
            w_i.pix.close_faccessors()
        wout.append(res_i)

    for i in range(n_inputs):
        if not is_sqw_obj[i] and hasattr(ldrs[i], "fclose"):
            ldrs[i].fclose()

    return wout[0] if len(wout) == 1 else DataOpArray(wout)


sqw.sqw_op = sqw_op  # type: ignore[attr-defined]
sqw.sqw_op_bin_pixels = sqw_op_bin_pixels  # type: ignore[attr-defined]
SQW = sqw

