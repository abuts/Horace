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
        """Take a cut from an ``sqw`` object along existing or new projection axes.

        Ported from ``@sqw/cut.m``, ``@sqw/private/cut_single_.m``, and
        ``@sqw/private/cut_accumulate_data_.m``.
        """
        from pyhorace.coord_transf.bmatrix import _infer_nargout
        from pyhorace.core.configurations.hor_config import hor_config

        nargout = kwargs.pop("nargout", None)
        if nargout is None:
            nargout = _infer_nargout(default=1)
        return_cut = bool(nargout > 0)

        if bool(self.dnd_type):
            return self._data.cut(*args, nargout=nargout, **kwargs)

        log_level = int(hor_config().log_level)
        targ_proj, pbin_list, sym, opt = self.process_and_validate_cut_inputs(
            return_cut, *args, nargout=4
        )

        n_cuts = len(pbin_list)
        if n_cuts > 1 and opt.get("outfile"):
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Guard against single file output with multiple cuts",
            )

        results: List[Any] = []
        for pbin in pbin_list:
            targ_ax_block, cur_proj, cur_sym = self.define_target_axes_block(
                targ_proj, pbin, sym, nargout=3
            )
            wout = self._cut_single(
                cur_proj,
                targ_ax_block,
                opt,
                log_level,
                cur_sym,
                return_cut=return_cut,
            )
            if return_cut:
                results.append(wout)

        if not return_cut:
            return None
        return results[0] if len(results) == 1 else DataOpArray(results)

    def cut_dnd(self, *args: Any, **kwargs: Any) -> Any:
        """Take a DND cut (no pixels retained) from an ``sqw`` object (port of ``@sqw/cut_dnd.m``)."""
        from pyhorace.coord_transf.bmatrix import _infer_nargout

        nargout = kwargs.pop("nargout", None)
        if nargout is None:
            nargout = _infer_nargout(default=1)
        return self.cut(*args, "-nopix", nargout=nargout, **kwargs)

    def cut_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Take an SQW cut from an ``sqw`` object (port of ``@sqw/cut_sqw.m``)."""
        from pyhorace.coord_transf.bmatrix import _infer_nargout

        nargout = kwargs.pop("nargout", None)
        if nargout is None:
            nargout = _infer_nargout(default=1)
        if bool(self.dnd_type):
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Cannot call cut_sqw on an sqw object of dnd_type (without pixels)",
            )
        return self.cut(*args, nargout=nargout, **kwargs)

    def cut_sym(self, *args: Any, **kwargs: Any) -> Any:
        """Take a symmetry-enabled cut from an ``sqw`` object (port of ``@sqw/cut_sym.m``)."""
        from pyhorace.coord_transf.bmatrix import _infer_nargout

        nargout = kwargs.pop("nargout", None)
        if nargout is None:
            nargout = _infer_nargout(default=1)
        return self.cut(*args, nargout=nargout, **kwargs)

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
        keep_pix = bool(opt.get("keep_pix", True))
        outfile = str(opt.get("outfile", "") or "")

        s, e, npix, pix_out, runid_contributed = self._cut_accumulate_data(
            targ_proj, targ_axes, keep_pix, log_level, return_cut, sym
        )

        proj_0 = (
            targ_proj[0]
            if isinstance(targ_proj, (list, tuple, np.ndarray))
            else targ_proj
        )
        data_dnd = DnDBase.dnd(targ_axes, proj_0, s, e, npix)
        data_dnd.title = self.title

        if not keep_pix:
            wout: Union[sqw, DnDBase] = data_dnd
        else:
            main_header = copy.deepcopy(self._main_header)
            exp_info = copy.deepcopy(self._exper_info)

            if (
                not main_header.creation_date_defined
                and exp_info.n_runs > 0
                and len(runid_contributed) > 0
            ):
                cur_keys = set(int(k) for k in np.asarray(exp_info.runid_map.keys).ravel())
                contrib_set = set(int(r) for r in np.asarray(runid_contributed).ravel())
                if not contrib_set.issubset(cur_keys) and self._pix.num_pixels > 0:
                    from pyhorace.coord_transf._dependencies import fast_map

                    u_ids = np.unique(self._pix.run_idx).astype(float)
                    if u_ids.size == exp_info.n_runs:
                        exp_info.runid_map = fast_map(
                            u_ids, np.arange(1, exp_info.n_runs + 1, dtype=int)
                        )
                        exp_info.runid_recalculated = False

            if len(runid_contributed) > 0 and exp_info.n_runs > 0:
                exp_sub = exp_info.get_subobj(
                    np.asarray(runid_contributed, dtype=int).ravel()
                )
            else:
                exp_sub = Experiment()

            main_header.nfiles = exp_sub.n_runs
            wout_sqw = sqw()
            wout_sqw._do_check_combo_arg = False
            wout_sqw._main_header = main_header
            wout_sqw._exper_info = exp_sub
            wout_sqw._data = data_dnd
            wout_sqw._pix = pix_out
            wout_sqw._do_check_combo_arg = True
            wout_sqw.check_combo_arg()
            wout = wout_sqw

        if outfile:
            wout.save(outfile)
            wout.full_filename = outfile

        if not return_cut:
            return None
        return wout

    def _cut_accumulate_data(
        self,
        targ_proj: Any,
        targ_axes: Any,
        keep_pixels: bool,
        log_level: int,
        return_cut: bool,
        sym: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PixelDataBase, np.ndarray]:
        """Accumulate binned signal, error, npix, and pixels for a cut.

        Ported from ``@sqw/private/cut_accumulate_data_.m``.
        """
        from pyhorace.coord_transf.axes_block_base import normalize_signal
        from pyhorace.core.configurations.hor_config import hor_config

        targ_proj_list: List[Any] = (
            list(targ_proj)
            if isinstance(targ_proj, (list, tuple, np.ndarray))
            else [targ_proj]
        )
        sym_list: List[Any] = (
            list(sym) if isinstance(sym, (list, tuple, np.ndarray)) else [sym]
        )
        n_sym = len(targ_proj_list)
        apply_symmetries = n_sym > 1

        sproj = copy.deepcopy(self._data.proj)
        saxes = self._data.axes
        data_npix = self._data.npix

        starts_per_sym: List[np.ndarray] = []
        sizes_per_sym: List[np.ndarray] = []
        for proj in targ_proj_list:
            b_starts, b_sizes = sproj.get_nrange(data_npix, saxes, targ_axes, proj)
            starts_per_sym.append(np.asarray(b_starts, dtype=np.int64).ravel())
            sizes_per_sym.append(np.asarray(b_sizes, dtype=np.int64).ravel())

        if apply_symmetries:
            block_starts, block_sizes = sqw._merge_ranges(
                starts_per_sym, sizes_per_sym
            )
        else:
            block_starts = starts_per_sym[0]
            block_sizes = sizes_per_sym[0]

        targ_dims = tuple(int(x) for x in targ_axes.dims_as_ssize)
        s = np.zeros(targ_dims, dtype=np.float64, order="F")
        e = np.zeros(targ_dims, dtype=np.float64, order="F")
        npix = np.zeros(targ_dims, dtype=np.float64, order="F")

        buf_size = int(hor_config().mem_chunk_size)
        chunks_starts, chunks_sizes = sqw._split_data_blocks(
            block_starts, block_sizes, buf_size
        )

        pix_retained: List[PixelDataBase] = []
        pix_ix_retained: List[np.ndarray] = []
        unique_runid = np.array([], dtype=np.int64)

        num_chunks = len(chunks_starts)
        is_fb = bool(self._pix.is_filebacked)
        access_type = "Read" if is_fb else "Access"
        pixel_contrib_name = "retained" if keep_pixels else "contributed"

        import time

        t_start = time.perf_counter()
        t_read_total = 0.0
        t_bin_total = 0.0
        t_sort_total = 0.0

        for iter_idx, (ch_starts, ch_sizes) in enumerate(
            zip(chunks_starts, chunks_sizes), start=1
        ):
            t0 = time.perf_counter()
            candidate_pix = self._pix.get_pix_in_ranges(ch_starts, ch_sizes, False, True)
            t_read_total += time.perf_counter() - t0
            if candidate_pix.num_pixels == 0:
                continue

            if log_level >= 1:
                print(
                    f"*** Step {iter_idx:3d} of {num_chunks:3d}; Read data for {candidate_pix.num_pixels:8d} pixels -- processing data...",
                    end="",
                )

            t1 = time.perf_counter()
            npix_before = float(np.sum(npix))
            if keep_pixels:
                for i in range(n_sym):
                    proj = targ_proj_list[i]
                    if apply_symmetries:
                        (
                            npix,
                            s,
                            e,
                            pix_ok,
                            unique_runid,
                            pix_indx,
                            selected,
                        ) = proj.bin_pixels(
                            targ_axes,
                            candidate_pix,
                            npix,
                            s,
                            e,
                            unique_runid,
                            "-nomex",
                            nargout=7,
                        )
                        if pix_ok.num_pixels > 0:
                            pix_ok.q_coordinates = sym_list[i].transform_vec(
                                pix_ok.q_coordinates
                            )
                        candidate_pix = candidate_pix.tag(selected)
                    else:
                        (
                            npix,
                            s,
                            e,
                            pix_ok,
                            unique_runid,
                            pix_indx,
                        ) = proj.bin_pixels(
                            targ_axes,
                            candidate_pix,
                            npix,
                            s,
                            e,
                            unique_runid,
                            "-nomex",
                            nargout=6,
                        )
                    if pix_ok.num_pixels > 0:
                        pix_retained.append(pix_ok)
                        pix_ix_retained.append(np.asarray(pix_indx, dtype=np.int64).ravel())
            else:
                for i in range(n_sym):
                    proj = targ_proj_list[i]
                    if apply_symmetries:
                        npix, s, e, selected = proj.bin_pixels(
                            targ_axes,
                            candidate_pix,
                            npix,
                            s,
                            e,
                            "-selected_only",
                            "-nomex",
                            nargout=4,
                        )
                        candidate_pix = candidate_pix.tag(selected)
                    else:
                        npix, s, e = proj.bin_pixels(
                            targ_axes,
                            candidate_pix,
                            npix,
                            s,
                            e,
                            "-nomex",
                            nargout=3,
                        )
            t_bin_total += time.perf_counter() - t1
            npix_step = int(round(float(np.sum(npix)) - npix_before))
            if log_level >= 1:
                print(f" ----->  {pixel_contrib_name}  {npix_step:8d} pixels")

        s, e = normalize_signal(s, e, npix)

        t2 = time.perf_counter()
        if keep_pixels:
            pix_out = sqw._sort_pix(pix_retained, pix_ix_retained, npix)
        else:
            pix_out = PixelDataMemory()
        t_sort_total += time.perf_counter() - t2

        if log_level >= 2:
            t_total = max(time.perf_counter() - t_start, 1e-9)
            t_read_safe = max(t_read_total, 1e-9)
            n_read_bytes = float(np.sum(block_sizes)) * 9.0 * 4.0
            speed_mbs = (n_read_bytes / (1024.0 * 1024.0)) / t_read_safe
            prep_pct = (100.0 * t_sort_total / t_total) if keep_pixels else 0.0
            print(
                f"*** Cut timings: Total={t_total:.3f}s | {access_type} speed: {speed_mbs:.2f} MB/sec\n"
                f"    Resulting pix preparation: {prep_pct:5.1f}%"
            )

        return s, e, npix, pix_out, unique_runid

    @staticmethod
    def _merge_ranges(
        starts_cell: Sequence[np.ndarray], sizes_cell: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge overlapping or adjacent 1-based pixel block ranges across symmetries.

        Ported from ``merge_ranges`` in ``@sqw/private/cut_accumulate_data_.m``.
        """
        non_empty_starts = []
        non_empty_ends = []
        for st, sz in zip(starts_cell, sizes_cell):
            st_arr = np.asarray(st, dtype=np.int64).ravel()
            sz_arr = np.asarray(sz, dtype=np.int64).ravel()
            valid = sz_arr > 0
            if np.any(valid):
                s_v = st_arr[valid]
                e_v = s_v + sz_arr[valid] - 1
                non_empty_starts.append(s_v)
                non_empty_ends.append(e_v)

        if not non_empty_starts:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        all_starts = np.concatenate(non_empty_starts)
        all_ends = np.concatenate(non_empty_ends)

        order = np.lexsort((all_ends, all_starts))
        all_starts = all_starts[order]
        all_ends = all_ends[order]

        merged_starts = [int(all_starts[0])]
        merged_ends = [int(all_ends[0])]
        for s_cur, e_cur in zip(all_starts[1:], all_ends[1:]):
            if int(s_cur) <= merged_ends[-1] + 1:
                if int(e_cur) > merged_ends[-1]:
                    merged_ends[-1] = int(e_cur)
            else:
                merged_starts.append(int(s_cur))
                merged_ends.append(int(e_cur))

        ms = np.asarray(merged_starts, dtype=np.int64)
        me = np.asarray(merged_ends, dtype=np.int64)
        return ms, me - ms + 1

    @staticmethod
    def _split_data_blocks(
        block_starts: np.ndarray, block_sizes: np.ndarray, buf_size: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split pixel block ranges into chunks of at most ``buf_size`` pixels.

        Ported from ``split_data_blocks.m``.
        """
        starts = np.asarray(block_starts, dtype=np.int64).ravel()
        sizes = np.asarray(block_sizes, dtype=np.int64).ravel()
        valid = sizes > 0
        starts = starts[valid]
        sizes = sizes[valid]

        if starts.size == 0:
            return [], []

        buf_sz = max(int(buf_size), 1)
        chunks_starts: List[np.ndarray] = []
        chunks_sizes: List[np.ndarray] = []

        cur_starts: List[int] = []
        cur_sizes: List[int] = []
        cur_sum = 0

        for s_val, sz_val in zip(starts, sizes):
            s = int(s_val)
            sz = int(sz_val)
            while sz > 0:
                space = buf_sz - cur_sum
                if sz <= space:
                    cur_starts.append(s)
                    cur_sizes.append(sz)
                    cur_sum += sz
                    sz = 0
                    if cur_sum >= buf_sz:
                        chunks_starts.append(np.asarray(cur_starts, dtype=np.int64))
                        chunks_sizes.append(np.asarray(cur_sizes, dtype=np.int64))
                        cur_starts = []
                        cur_sizes = []
                        cur_sum = 0
                else:
                    cur_starts.append(s)
                    cur_sizes.append(space)
                    chunks_starts.append(np.asarray(cur_starts, dtype=np.int64))
                    chunks_sizes.append(np.asarray(cur_sizes, dtype=np.int64))
                    cur_starts = []
                    cur_sizes = []
                    cur_sum = 0
                    s += space
                    sz -= space

        if cur_starts:
            chunks_starts.append(np.asarray(cur_starts, dtype=np.int64))
            chunks_sizes.append(np.asarray(cur_sizes, dtype=np.int64))

        return chunks_starts, chunks_sizes

    @staticmethod
    def _sort_pix(
        pix_retained: Sequence[PixelDataBase],
        pix_ix_retained: Sequence[np.ndarray],
        npix: np.ndarray,
    ) -> PixelDataBase:
        """Sort accumulated pixel chunks by 1-based bin index ``pix_ix_retained``.

        Ported from ``sort_pix.m``.
        """
        non_empty_pix = []
        non_empty_ix = []
        for p, ix in zip(pix_retained, pix_ix_retained):
            if p is not None and p.num_pixels > 0:
                non_empty_pix.append(p.get_pixels(np.arange(1, p.num_pixels + 1, dtype=np.intp), "-raw_data"))
                non_empty_ix.append(np.asarray(ix, dtype=np.int64).ravel())

        if not non_empty_pix:
            return PixelDataMemory()

        all_raw = np.hstack(non_empty_pix)
        all_ix = np.concatenate(non_empty_ix)
        if all_ix.size > 1:
            order = np.argsort(all_ix, kind="mergesort")
            all_raw = all_raw[:, order]

        out = PixelDataMemory(np.asarray(all_raw, dtype=np.float64))
        out.recalc_data_range("all")
        return out

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


SQW = sqw

