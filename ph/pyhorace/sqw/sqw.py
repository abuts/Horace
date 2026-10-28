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
class sqw(SQWDnDBase):
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
        Serializable.__init__(self)
        self._main_header: main_header_cl = main_header_cl()
        self._exper_info: Experiment = Experiment()
        self._data: DnDBase = d0d()
        self._pix: PixelDataBase = PixelDataBase.create()
        self._is_tmp_obj: bool = False
        self._tmp_file_path: str = ""
        if len(args) == 0 and len(kwargs) == 0:
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
            res = loader.get_sqw("-verbatim")
            if isinstance(res, sqw):
                self._copy_from(res)
                self._trim_old_file_exper_info()
                self._main_header.nfiles = self._exper_info.n_runs
                return self
            if isinstance(res, dict):
                return self._init_from_dict(res, pixel_page_size=pixel_page_size, file_backed=file_backed)

        mh = loader.get_main_header("-keep_original") if hasattr(loader, "get_main_header") else main_header_cl()
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

        dnd_data_obj = loader.get_dnd("-verbatim") if hasattr(loader, "get_dnd") else (
            loader.get_data("-verbatim") if hasattr(loader, "get_data") else d0d()
        )
        if not isinstance(dnd_data_obj, DnDBase):
            dnd_data_obj = DnDBase.dnd(dnd_data_obj)

        pix_obj = PixelDataBase.create(loader, filebacked=file_backed)
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

        self._main_header = mh if isinstance(mh, main_header_cl) else main_header_cl(mh)
        self._exper_info = exp_info if isinstance(exp_info, Experiment) else Experiment(exp_info)
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
            if detpar is not None and not (isinstance(detpar, (list, tuple, dict)) and len(detpar) == 0):
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
                if isinstance(raw_pix, PixelDataBase):
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
        # Do not duplicate temp-file deletion ownership onto copies by default
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
            self._is_tmp_obj = True
            self._tmp_file_path = target
        return self

    def deactivate(self) -> "sqw":
        """Close open file-backed pixel handles (port of ``@sqw/deactivate.m``)."""
        if self._pix is not None:
            self._pix.deactivate()
        self._is_tmp_obj = False
        self._tmp_file_path = ""
        return self

    def activate(self, new_file: Optional[str] = None) -> "sqw":
        """Re-open file-backed pixel handles (port of ``@sqw/activate.m``)."""
        target = str(new_file) if new_file is not None else self.full_filename
        if self._pix is not None and target:
            self._pix.activate(target)
        if target:
            self.full_filename = target
            _, ext = os.path.splitext(target)
            if ext.lower() == ".tmp":
                self.set_as_tmp_obj(target)
        return self

    def delete(self) -> None:
        """Clean up temporary backing file if ``is_tmp_obj`` is True (port of ``@sqw/delete.m``)."""
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
        elif isinstance(val, main_header_cl):
            self._main_header = copy.deepcopy(val)
        elif isinstance(val, dict):
            if "serial_name" in val:
                loaded = Serializable.loadobj(val)
                if not isinstance(loaded, main_header_cl):
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
        elif isinstance(val, Experiment):
            self._exper_info = val
        elif isinstance(val, IX_experiment):
            self._exper_info = Experiment(val)
        elif isinstance(val, dict):
            if "serial_name" in val:
                loaded = Serializable.loadobj(val)
                if not isinstance(loaded, Experiment):
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
        elif isinstance(val, PixelDataBase):
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

    def has_pixels(self) -> bool:
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
    def dimensions(self) -> int:
        return int(self._data.dimensions)

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
        """Stub for ``@sqw/cut.m``."""
        # TODO(port-dependency): Port @sqw/cut.m
        raise NotImplementedError("sqw.cut is not yet implemented.")

    def cut_dnd(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/cut_dnd.m``."""
        # TODO(port-dependency): Port @sqw/cut_dnd.m
        raise NotImplementedError("sqw.cut_dnd is not yet implemented.")

    def cut_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/cut_sqw.m``."""
        # TODO(port-dependency): Port @sqw/cut_sqw.m
        raise NotImplementedError("sqw.cut_sqw is not yet implemented.")

    def cut_sym(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/cut_sym.m``."""
        # TODO(port-dependency): Port @sqw/cut_sym.m
        raise NotImplementedError("sqw.cut_sym is not yet implemented.")

    def dimensions_match(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/dimensions_match.m``."""
        # TODO(port-dependency): Port @sqw/dimensions_match.m
        raise NotImplementedError("sqw.dimensions_match is not yet implemented.")

    def disp2sqw_eval(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/disp2sqw_eval.m``."""
        # TODO(port-dependency): Port @sqw/disp2sqw_eval.m
        raise NotImplementedError("sqw.disp2sqw_eval is not yet implemented.")

    def fake_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/fake_sqw.m``."""
        # TODO(port-dependency): Port @sqw/fake_sqw.m
        raise NotImplementedError("sqw.fake_sqw is not yet implemented.")

    def func_eval(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/func_eval.m``."""
        # TODO(port-dependency): Port @sqw/func_eval.m
        raise NotImplementedError("sqw.func_eval is not yet implemented.")

    def gen_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/gen_sqw.m``."""
        # TODO(port-dependency): Port @sqw/gen_sqw.m
        raise NotImplementedError("sqw.gen_sqw is not yet implemented.")

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

    def get_proj_and_pbin(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/get_proj_and_pbin.m``."""
        # TODO(port-dependency): Port @sqw/get_proj_and_pbin.m
        raise NotImplementedError("sqw.get_proj_and_pbin is not yet implemented.")

    def get_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/get_sqw.m``."""
        # TODO(port-dependency): Port @sqw/get_sqw.m
        raise NotImplementedError("sqw.get_sqw is not yet implemented.")

    def IX_dataset_1d(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/IX_dataset_1d.m``."""
        # TODO(port-dependency): Port @sqw/IX_dataset_1d.m
        raise NotImplementedError("sqw.IX_dataset_1d is not yet implemented.")

    def IX_dataset_2d(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/IX_dataset_2d.m``."""
        # TODO(port-dependency): Port @sqw/IX_dataset_2d.m
        raise NotImplementedError("sqw.IX_dataset_2d is not yet implemented.")

    def IX_dataset_3d(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/IX_dataset_3d.m``."""
        # TODO(port-dependency): Port @sqw/IX_dataset_3d.m
        raise NotImplementedError("sqw.IX_dataset_3d is not yet implemented.")

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
        raise NotImplementedError("sqw.page_op is not yet implemented.")

    def parse_pixel_indices(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/parse_pixel_indices.m``."""
        # TODO(port-dependency): Port @sqw/parse_pixel_indices.m
        raise NotImplementedError("sqw.parse_pixel_indices is not yet implemented.")

    def recompute_bin_data(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/recompute_bin_data.m``."""
        # TODO(port-dependency): Port @sqw/recompute_bin_data.m
        raise NotImplementedError("sqw.recompute_bin_data is not yet implemented.")

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

    def sqw_eval(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/sqw_eval.m``."""
        # TODO(port-dependency): Port @sqw/sqw_eval.m
        raise NotImplementedError("sqw.sqw_eval is not yet implemented.")

    def sqw_eval_pix(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/sqw_eval_pix.m``."""
        # TODO(port-dependency): Port @sqw/sqw_eval_pix.m
        raise NotImplementedError("sqw.sqw_eval_pix is not yet implemented.")

    def symmetrise_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Stub for ``@sqw/symmetrise_sqw.m``."""
        # TODO(port-dependency): Port @sqw/symmetrise_sqw.m
        raise NotImplementedError("sqw.symmetrise_sqw is not yet implemented.")

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


SQW = sqw

