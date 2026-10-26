"""IX_experiment class holding single-run experimental metadata and goniometer parameters."""

from __future__ import annotations

import copy
import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf import bmatrix, ubmatrix
from pyhorace.coord_transf._dependencies import fast_map
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.hashable import HashableArray, build_hash, isequal
from pyhorace.data.instr_classes.goniometer import Goniometer


class IX_experiment(Goniometer):
    """Experimental metadata for a single neutron scattering run.

    Inherits crystal lattice and orientation angles from :class:`Goniometer`.
    Corresponds to MATLAB ``IX_experiment`` in Herbert/Horace.
    """

    _STRING_POS_FIELDS: ClassVar[Tuple[str, ...]] = ("filename", "filepath")

    _LEGACY_CONSTR_FIELDS: ClassVar[Tuple[str, ...]] = (
        "filename",
        "filepath",
        "efix",
        "emode",
        "cu",
        "cv",
        "psi",
        "omega",
        "dpsi",
        "gl",
        "gs",
        "en",
        "uoffset",
        "u_to_rlu",
        "ulen",
        "ulabel",
        "run_id",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._filename: str = ""
        self._filepath: str = ""
        self._run_id: float = float("nan")
        self._en: np.ndarray = np.zeros(0, dtype=float)
        self._u_to_rlu: Optional[np.ndarray] = None
        self._uoffset: np.ndarray = np.zeros(4, dtype=float)
        self._ulen: np.ndarray = np.ones(4, dtype=float)
        self._ulabel: List[str] = ["", "", "", ""]
        if len(args) == 0 and len(kwargs) == 0:
            return
        self.init(*args, **kwargs)

    # ------------------------------------------------------------------
    # Construction & initialization
    # ------------------------------------------------------------------
    def init(self, *args: Any, **kwargs: Any) -> "IX_experiment":
        """Initialize or re-initialize the IX_experiment instance."""
        if len(args) == 0 and len(kwargs) == 0:
            return self

        if len(args) == 1 and len(kwargs) == 0:
            arg0 = args[0]
            if isinstance(arg0, IX_experiment):
                self._copy_experiment_from(arg0)
                return self
            if isinstance(arg0, Goniometer):
                self._copy_goniometer_from(arg0)
                return self
            if isinstance(arg0, dict):
                return self._init_from_dict(arg0)

        arg_list = list(args)
        base_dict: Optional[Dict[str, Any]] = None
        if len(arg_list) >= 1 and isinstance(arg_list[0], (dict, Goniometer)):
            first = arg_list.pop(0)
            if isinstance(first, IX_experiment):
                self._copy_experiment_from(first)
            elif isinstance(first, Goniometer):
                self._copy_goniometer_from(first)
            else:
                base_dict = dict(first)

        # Extract angular units first
        arg_list, kwargs, unit_val = self.check_angular_units_present(arg_list, kwargs)

        self._do_check_combo_arg = False
        try:
            if base_dict is not None:
                self._apply_struct_dict(base_dict, override_units=(unit_val is None))

            if unit_val is not None:
                self.angular_units = unit_val

            pos_args, kv_dict = self._split_positional_and_kv(arg_list)
            for k, v in kwargs.items():
                kv_dict[k] = v

            # Extract any standalone Goniometer in pos_args
            filtered_pos: List[Any] = []
            for p in pos_args:
                if isinstance(p, Goniometer):
                    self.goniometer = p
                else:
                    filtered_pos.append(p)
            pos_args = filtered_pos

            n_pos = len(pos_args)
            if n_pos > 0:
                pos_fields = self._select_positional_fields(pos_args)
                if n_pos > len(pos_fields):
                    raise ValueError(
                        f"HERBERT:IX_experiment:invalid_argument: "
                        f"Too many positional arguments ({n_pos})."
                    )
                for fname, fval in zip(pos_fields[:n_pos], pos_args):
                    self._set_named_field(fname, fval)

            for fname, fval in kv_dict.items():
                self._set_named_field(fname, fval)
        finally:
            self._do_check_combo_arg = True

        self.check_combo_arg()
        return self

    def _copy_experiment_from(self, other: "IX_experiment") -> None:
        self._copy_goniometer_from(other)
        self._filename = other._filename
        self._filepath = other._filepath
        self._run_id = float(other._run_id)
        self._en = other._en.copy()
        self._u_to_rlu = None if other._u_to_rlu is None else other._u_to_rlu.copy()
        self._uoffset = other._uoffset.copy()
        self._ulen = other._ulen.copy()
        self._ulabel = list(other._ulabel)
        self.clear_hash()

    def _select_positional_fields(self, pos_args: List[Any]) -> Tuple[str, ...]:
        n_pos = len(pos_args)
        # Distinguish saveable_fields order vs legacy init_.m order:
        # saveable_fields: ('filename', 'filepath', 'run_id', 'en', 'efix', 'emode', 'alatt', 'angdeg', 'u', 'v', ...)
        # legacy: ('filename', 'filepath', 'efix', 'emode', 'cu', 'cv', 'psi', 'omega', 'dpsi', 'gl', 'gs', 'en', ...)
        if n_pos >= 7:
            arg6_size = np.size(pos_args[6]) if not isinstance(pos_args[6], str) else -1
            arg4_size = np.size(pos_args[4]) if not isinstance(pos_args[4], str) else -1
            if arg6_size == 3 and arg4_size == 1:
                return self.saveable_fields()
        return self._LEGACY_CONSTR_FIELDS

    def _resolve_field_name(self, raw_name: str) -> str:
        if not isinstance(raw_name, str):
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"Field name must be a string, got {type(raw_name).__name__}"
            )
        name = raw_name.strip()
        if name.startswith("-"):
            warnings.warn(
                f"HERBERT:Serializable:deprecated: Keyword '{raw_name}' with leading '-' is deprecated; "
                f"use '{name.lstrip('-').strip()}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            name = name.lstrip("-").strip()

        if name == "cu":
            return "u"
        if name == "cv":
            return "v"

        valid_fields = set(self.construction_fields()) | {
            "u",
            "v",
            "cu",
            "cv",
            "u_to_rlu",
            "uoffset",
            "ulen",
            "ulabel",
            "goniometer",
            "angular_is_degree",
            "degrees",
            "radians",
        }
        if name in valid_fields:
            return name

        raise ValueError(
            f"HERBERT:IX_experiment:invalid_argument: "
            f"Unrecognized property or keyword '{raw_name}'."
        )

    # ------------------------------------------------------------------
    # Serializable & hashable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 3

    def saveable_fields(self) -> Tuple[str, ...]:
        return (
            "filename",
            "filepath",
            "run_id",
            "en",
            "efix",
            "emode",
            "alatt",
            "angdeg",
            "u",
            "v",
            "psi",
            "omega",
            "dpsi",
            "gl",
            "gs",
            "angular_units",
        )

    def construction_fields(self) -> Tuple[str, ...]:
        return self.saveable_fields()

    def hashable_fields(self) -> Tuple[str, ...]:
        return (
            "filename",
            "filepath",
            "efix",
            "emode",
            "alatt",
            "angdeg",
            "u",
            "v",
            "psi",
            "omega",
            "dpsi",
            "gl",
            "gs",
            "angular_units",
        )

    def check_combo_arg(self) -> "IX_experiment":
        """Validate interdependent properties (collinearity of u, v and efix/emode compatibility)."""
        super().check_combo_arg()
        if self._efix == 0.0 and self._emode != 0:
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"efix (incident energy) can be 0 in elastic mode only (emode=0). Got emode={self._emode}."
            )
        self.clear_hash()
        return self

    def to_bare_struct(self, recursive_bare: bool = False) -> Dict[str, Any]:
        S = super().to_bare_struct(recursive_bare=recursive_bare)
        S.pop("angular_is_degree", None)
        if self._u_to_rlu is not None:
            S["u_to_rlu"] = self.u_to_rlu
        return S

    def from_bare_struct(self, S: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        if isinstance(S, list):
            return super().from_bare_struct(S)
        s_copy = dict(S)
        self._do_check_combo_arg = False
        try:
            if "goniometer" in s_copy:
                self.goniometer = s_copy.pop("goniometer")
            if "u_to_rlu" in s_copy:
                self.u_to_rlu = s_copy.pop("u_to_rlu")
            super().from_bare_struct(s_copy)
        finally:
            self._do_check_combo_arg = True
        self.check_combo_arg()
        return self

    def convert_old_struct(self, S: Dict[str, Any], ver: float = 1) -> Dict[str, Any]:
        """Convert older version structs (ver 1 or 2) to version 3 format."""
        out = dict(S)
        if ver == 1:
            out["run_id"] = float("nan")
        if ver <= 2:
            out = self._recover_from_v2_structure(out)
        return out

    @staticmethod
    def _recover_from_v2_structure(S: Dict[str, Any]) -> Dict[str, Any]:
        """Port of recover_from_v2_structure_.m."""
        out = dict(S)
        if "goniometer" in out:
            gon = out.pop("goniometer")
            if isinstance(gon, dict):
                if "serial_name" in gon:
                    gon_obj = Goniometer.loadobj(gon)
                else:
                    gon_obj = Goniometer(gon)
            elif isinstance(gon, Goniometer):
                gon_obj = copy.deepcopy(gon)
            else:
                gon_obj = Goniometer()
            gon_obj.angular_is_degree = False
            for fld in ("efix", "emode", "alatt", "angdeg", "u", "v", "psi", "omega", "dpsi", "gl", "gs", "angular_units"):
                if fld not in out or out[fld] is None:
                    out[fld] = getattr(gon_obj, fld)
        else:
            # Old versions stored angles in radians by default
            if "angular_units" not in out and "angular_is_degree" not in out:
                out["angular_units"] = "rad"

        if "cu" in out and "u" not in out:
            out["u"] = out.pop("cu")
        else:
            out.pop("cu", None)
        if "cv" in out and "v" not in out:
            out["v"] = out.pop("cv")
        else:
            out.pop("cv", None)

        return out

    # ------------------------------------------------------------------
    # Properties specific to IX_experiment
    # ------------------------------------------------------------------
    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, val: Any) -> None:
        if val is None:
            self._filename = ""
        elif isinstance(val, str):
            self._filename = val
        else:
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"filename must be a string, got {type(val).__name__}."
            )
        self.clear_hash()

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, val: Any) -> None:
        if val is None:
            self._filepath = ""
        elif isinstance(val, str):
            self._filepath = val
        else:
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"filepath must be a string, got {type(val).__name__}."
            )
        self.clear_hash()

    @property
    def run_id(self) -> float:
        return self._run_id

    @run_id.setter
    def run_id(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.size(val) == 0):
            self._run_id = float("nan")
            return
        if isinstance(val, (bool, np.bool_, str)):
            raise ValueError(
                "HERBERT:IX_experiment:invalid_argument: "
                "run_id must be a numeric scalar or NaN."
            )
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HERBERT:IX_experiment:invalid_argument: "
                "run_id must be a numeric scalar."
            ) from exc
        if arr.size != 1:
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"run_id must be a scalar, got size {arr.size}."
            )
        self._run_id = float(arr.item())

    @property
    def en(self) -> np.ndarray:
        return self._en.copy()

    @en.setter
    def en(self, val: Any) -> None:
        if val is None:
            self._en = np.zeros(0, dtype=float)
            return
        if isinstance(val, (bool, np.bool_, str)):
            raise ValueError(
                "HERBERT:IX_experiment:invalid_argument: "
                "en must be a numeric vector."
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HERBERT:IX_experiment:invalid_argument: "
                "en must be a numeric vector."
            ) from exc
        self._en = arr

    @property
    def goniometer(self) -> Goniometer:
        gon = Goniometer()
        gon._copy_goniometer_from(self)
        return gon

    @goniometer.setter
    def goniometer(self, val: Any) -> None:
        if isinstance(val, Goniometer):
            self._copy_goniometer_from(val)
        elif isinstance(val, dict):
            gon = Goniometer(val)
            self._copy_goniometer_from(gon)
        else:
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"goniometer must be a Goniometer instance or dict, got {type(val).__name__}."
            )
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def u_to_rlu(self) -> np.ndarray:
        if self._u_to_rlu is None:
            return np.zeros(0, dtype=float)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = self._u_to_rlu
        return mat

    @u_to_rlu.setter
    def u_to_rlu(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.size(val) == 0):
            self._u_to_rlu = None
            return
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HERBERT:IX_experiment:invalid_argument: "
                "u_to_rlu must be a 3x3 or 4x4 numeric matrix."
            ) from exc
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] not in (3, 4):
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"u_to_rlu must be a 3x3 or 4x4 matrix, got shape {arr.shape}."
            )
        if self._subdiag_elements(arr):
            self._u_to_rlu = arr[:3, :3].copy()
        else:
            self._u_to_rlu = None

    @staticmethod
    def _subdiag_elements(mat: np.ndarray) -> bool:
        n = mat.shape[0]
        off_diag = mat[~np.eye(n, dtype=bool)]
        return bool(np.any(np.abs(off_diag) > 4.0e-7))

    @property
    def uoffset(self) -> np.ndarray:
        return self._uoffset.copy()

    @uoffset.setter
    def uoffset(self, val: Any) -> None:
        if val is None or np.size(val) == 0:
            self._uoffset = np.zeros(4, dtype=float)
            return
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 3:
            arr = np.concatenate([arr, [0.0]])
        elif arr.size != 4:
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"uoffset must have 3 or 4 elements, got {arr.size}."
            )
        self._uoffset = arr

    @property
    def ulen(self) -> np.ndarray:
        return self._ulen.copy()

    @ulen.setter
    def ulen(self, val: Any) -> None:
        if val is None or np.size(val) == 0:
            self._ulen = np.ones(4, dtype=float)
            return
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 3:
            arr = np.concatenate([arr, [1.0]])
        elif arr.size != 4:
            raise ValueError(
                f"HERBERT:IX_experiment:invalid_argument: "
                f"ulen must have 3 or 4 elements, got {arr.size}."
            )
        self._ulen = arr

    @property
    def ulabel(self) -> List[str]:
        return list(self._ulabel)

    @ulabel.setter
    def ulabel(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._ulabel = ["", "", "", ""]
            return
        if isinstance(val, (list, tuple, np.ndarray)):
            labels = [str(x) for x in val]
            if len(labels) == 3:
                labels.append("E")
            elif len(labels) != 4:
                raise ValueError(
                    f"HERBERT:IX_experiment:invalid_argument: "
                    f"ulabel must have 3 or 4 strings, got {len(labels)}."
                )
            self._ulabel = labels
        else:
            raise ValueError(
                "HERBERT:IX_experiment:invalid_argument: "
                "ulabel must be a sequence of strings."
            )

    # ------------------------------------------------------------------
    # Methods on IX_experiment
    # ------------------------------------------------------------------
    def convert_to_bin_boundaries(self, en: Optional[Sequence[float]] = None) -> np.ndarray:
        """Convert energy bin centres to bin boundaries.

        If ``en`` is None, converts ``self.en`` in-place and returns the new boundaries.
        """
        in_place = en is None
        arr = self._en if in_place else np.asarray(en, dtype=float).ravel()
        if arr.size == 0:
            res = np.zeros(0, dtype=float)
        elif arr.size == 1:
            res = np.array([arr[0] - 0.5, arr[0] + 0.5], dtype=float)
        else:
            de = 0.5 * np.diff(arr)
            res = np.concatenate(
                [[arr[0] - de[0]], 0.5 * (arr[:-1] + arr[1:]), [arr[-1] + de[-1]]]
            )
        if in_place:
            self._en = res
        return res.copy()

    def get_en(self, bin_centers: bool = False) -> np.ndarray:
        """Return energy array (either bin boundaries or bin centers)."""
        if not bin_centers or self._en.size < 2:
            return self._en.copy()
        return 0.5 * (self._en[:-1] + self._en[1:])

    def get_en_transfer(
        self,
        bin_centre: bool = True,
        get_lidx: bool = False,
        nargout: Optional[int] = None,
    ) -> Any:
        """Return energy transfer array(s) (and optionally unique indices)."""
        en_val = self.get_en(bin_centers=bin_centre)
        nout = _infer_nargout(default=2 if get_lidx else 1, explicit_nargout=nargout)
        if nout >= 2 or get_lidx:
            return [en_val], np.array([1], dtype=int)
        return en_val

    def get_efix(self, extract_unique: bool = False) -> float:
        """Return incident/fixed energy efix."""
        return float(self._efix)

    def get_emode(self) -> int:
        """Return spectrometer energy mode emode."""
        return int(self._emode)

    def get_run_ids(self) -> float:
        """Return run_id for this experiment."""
        return float(self._run_id)

    def set_run_ids(self, ids: Any) -> "IX_experiment":
        """Set run_id for this experiment."""
        self.run_id = ids
        return self

    def get_runid_map(self) -> fast_map:
        """Return a fast_map mapping this experiment's run_id to 1."""
        return fast_map([self._run_id], [1])

    def get_lattice_parameters(self, nargout: Optional[int] = None) -> Any:
        """Return lattice parameters (alatt, angdeg)."""
        nout = _infer_nargout(default=2, explicit_nargout=nargout)
        if nout == 1:
            return self.alatt
        return self.alatt, self.angdeg

    def check_SameBins(
        self,
        other: Union["IX_experiment", Sequence["IX_experiment"], np.ndarray],
        tol: float = 1e-8,
    ) -> bool:
        """Check if this experiment and ``other`` have identical energy bins within ``tol``."""
        ref_en = self._en
        if isinstance(other, IX_experiment):
            others = [other]
        elif isinstance(other, (list, tuple)):
            others = list(other)
        else:
            other_en = np.asarray(other, dtype=float).ravel()
            if ref_en.size != other_en.size:
                return False
            return bool(np.all(np.abs(ref_en - other_en) <= tol))

        for item in others:
            item_en = item._en if isinstance(item, IX_experiment) else np.asarray(item, dtype=float).ravel()
            if ref_en.size != item_en.size:
                return False
            if not np.all(np.abs(ref_en - item_en) <= tol):
                return False
        return True

    # ------------------------------------------------------------------
    # Binary file header conversion (convert_to_binfile_header_.m & build_from_binfile_header)
    # ------------------------------------------------------------------
    def convert_to_binfile_header(
        self,
        mode: str = "-alatt_angdeg",
        arg1: Any = None,
        arg2: Any = None,
        nomangle: bool = False,
    ) -> Dict[str, Any]:
        """Convert IX_experiment into a legacy Horace binary file header dictionary."""
        from pyhorace.data.instr_classes.ix_null_inst import IX_null_inst
        from pyhorace.data.instr_classes.ix_null_sample import IX_null_sample

        alatt = self._alatt.copy()
        angdeg = self._angdeg.copy()
        inst: Any = None
        samp: Any = None

        if mode == "-inst_samp":
            inst = arg1 if arg1 is not None else IX_null_inst()
            samp = arg2 if arg2 is not None else IX_null_sample()
            if hasattr(samp, "alatt") and np.size(samp.alatt) > 0:
                alatt = np.asarray(samp.alatt, dtype=float).ravel()
            if hasattr(samp, "angdeg") and np.size(samp.angdeg) > 0:
                angdeg = np.asarray(samp.angdeg, dtype=float).ravel()
        elif mode == "-alatt_angdeg":
            if arg1 is not None and not isinstance(arg1, str):
                alatt = np.asarray(arg1, dtype=float).ravel()
            if arg2 is not None and not isinstance(arg2, str):
                angdeg = np.asarray(arg2, dtype=float).ravel()
            if isinstance(arg1, str) and arg1 == "-nomangle":
                nomangle = True
            if isinstance(arg2, str) and arg2 == "-nomangle":
                nomangle = True
            inst = IX_null_inst()
            samp = IX_null_sample("", alatt, angdeg)
        elif mode == "-nomangle":
            nomangle = True
            inst = IX_null_inst()
            samp = IX_null_sample("", alatt, angdeg)
        else:
            raise ValueError(
                f"HORACE:IX_experiment:invalid_argument: Unknown convert_to_binfile_header mode '{mode}'."
            )

        header: Dict[str, Any] = {}
        if not nomangle and not np.isnan(self._run_id):
            base, ext = os.path.splitext(self._filename)
            header["filename"] = f"{base}$id${int(self._run_id)}{ext}"
        else:
            header["filename"] = self._filename

        header["filepath"] = self._filepath
        header["efix"] = float(self._efix)
        header["emode"] = int(self._emode)
        header["alatt"] = alatt
        header["angdeg"] = angdeg
        header["cu"] = self._u.copy()
        header["cv"] = self._v.copy()
        # Legacy binary header always stores angles in radians
        psi_deg = 0.0 if np.isnan(self._psi) else self._psi
        header["psi"] = float(np.deg2rad(psi_deg))
        header["omega"] = float(np.deg2rad(self._omega))
        header["dpsi"] = float(np.deg2rad(self._dpsi))
        header["gl"] = float(np.deg2rad(self._gl))
        header["gs"] = float(np.deg2rad(self._gs))
        header["en"] = self._en.copy()
        header["uoffset"] = self._uoffset.copy()

        if self._u_to_rlu is not None:
            header["u_to_rlu"] = self.u_to_rlu
        else:
            b_mat = bmatrix(alatt, angdeg, nargout=1)
            ub_mat = ubmatrix(self._u, self._v, b_mat, nargout=1)
            u_to_rlu = np.eye(4, dtype=float)
            u_to_rlu[:3, :3] = np.linalg.inv(ub_mat)
            header["u_to_rlu"] = u_to_rlu

        header["ulen"] = self._ulen.copy()
        header["ulabel"] = list(self._ulabel)
        header["instrument"] = inst
        header["sample"] = samp
        return header

    @staticmethod
    def build_from_binfile_header(
        inputs: Dict[str, Any],
        nargout: Optional[int] = None,
    ) -> Union["IX_experiment", Tuple["IX_experiment", np.ndarray, np.ndarray]]:
        """Construct an IX_experiment (and optionally alatt, angdeg) from a legacy binfile header dict."""
        nout = _infer_nargout(default=1, explicit_nargout=nargout)
        hdr = dict(inputs)
        alatt = np.asarray(hdr.pop("alatt", [2.0 * np.pi] * 3), dtype=float).ravel()
        angdeg = np.asarray(hdr.pop("angdeg", [90.0, 90.0, 90.0]), dtype=float).ravel()

        fname = str(hdr.get("filename", ""))
        run_id = float(hdr.get("run_id", float("nan")))
        if np.isnan(run_id) and "$" in fname:
            base, ext = os.path.splitext(fname)
            if "$id$" in base:
                prefix, suffix = base.rsplit("$id$", 1)
                try:
                    run_id = float(int(suffix))
                    fname = prefix + ext
                except ValueError:
                    pass
            elif "$" in base:
                prefix, suffix = base.rsplit("$", 1)
                try:
                    run_id = float(int(suffix))
                    fname = prefix + ext
                except ValueError:
                    pass

        exp = IX_experiment()
        exp._do_check_combo_arg = False
        try:
            exp.angular_units = hdr.get("angular_units", "rad")
            exp.filename = fname
            exp.filepath = str(hdr.get("filepath", ""))
            exp.run_id = run_id
            exp.efix = hdr.get("efix", 0.0)
            exp.emode = hdr.get("emode", 0)
            if nout != 3:
                exp.alatt = alatt
                exp.angdeg = angdeg
            if "cu" in hdr:
                exp.u = hdr["cu"]
            elif "u" in hdr:
                exp.u = hdr["u"]
            if "cv" in hdr:
                exp.v = hdr["cv"]
            elif "v" in hdr:
                exp.v = hdr["v"]
            for ang_field in ("psi", "omega", "dpsi", "gl", "gs"):
                if ang_field in hdr:
                    setattr(exp, ang_field, hdr[ang_field])
            if "en" in hdr:
                exp.en = hdr["en"]
            if "uoffset" in hdr:
                exp.uoffset = hdr["uoffset"]
            if "ulen" in hdr:
                exp.ulen = hdr["ulen"]
            if "ulabel" in hdr:
                exp.ulabel = hdr["ulabel"]
            if "u_to_rlu" in hdr and hdr["u_to_rlu"] is not None and np.size(hdr["u_to_rlu"]) > 0:
                u_mat = np.asarray(hdr["u_to_rlu"], dtype=float)
                b_mat = bmatrix(alatt, angdeg, nargout=1)
                ub_inv = np.linalg.inv(ubmatrix(exp.u, exp.v, b_mat, nargout=1))
                if np.any(np.abs(u_mat[:3, :3] - ub_inv) > 4.0e-7):
                    exp._u_to_rlu = u_mat[:3, :3].copy()
            if nout != 3:
                exp.angular_units = "deg"
        finally:
            exp._do_check_combo_arg = True

        exp.check_combo_arg()
        if nout == 3:
            return exp, alatt, angdeg
        if nout == 2:
            return exp, alatt
        return exp

    # ------------------------------------------------------------------
    # Combine experiments (combine_.m)
    # ------------------------------------------------------------------
    def combine(
        self,
        exper_cellarray: Sequence[Any],
        allow_equal_headers: bool = False,
        keep_runid: bool = True,
        this_runid_map: Optional[Any] = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Combine experiment metadata across multiple sqw/experiment objects."""
        return self.combine_experiments(
            exper_cellarray,
            allow_equal_headers=allow_equal_headers,
            keep_runid=keep_runid,
            this_runid_map=this_runid_map,
            nargout=nargout,
        )

    def combine_experiments(
        self,
        exper_cellarray: Sequence[Any],
        allow_equal_headers: bool = False,
        keep_runid: bool = True,
        this_runid_map: Optional[Any] = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Combine this experiment (or list of experiments) with other experiment lists.

        Port of ``@IX_experiment/private/combine_.m``.
        """
        all_groups: List[List[IX_experiment]] = [[self]]
        for item in exper_cellarray:
            if isinstance(item, IX_experiment):
                all_groups.append([item])
            elif isinstance(item, HashableArray):
                all_groups.append([x for x in item])
            elif isinstance(item, (list, tuple)):
                all_groups.append(list(item))
            else:
                raise ValueError(
                    f"HERBERT:IX_experiment:invalid_argument: "
                    f"Expected IX_experiment or sequence of IX_experiment, got {type(item).__name__}."
                )

        combined: List[IX_experiment] = []
        seen_hashes: Dict[str, int] = {}
        runid_change_maps: List[Dict[int, int]] = []
        next_run_id = 1

        for grp_idx, group in enumerate(all_groups):
            change_map: Dict[int, int] = {}
            for exp in group:
                exp_copy = copy.deepcopy(exp)
                old_id = int(exp_copy.run_id) if not np.isnan(exp_copy.run_id) else grp_idx + 1
                _, h = build_hash(exp_copy)
                if not allow_equal_headers and h in seen_hashes:
                    raise ValueError(
                        "HORACE:IX_experiment:invalid_argument: "
                        "Equal experiment headers are prohibited unless allow_equal_headers=True."
                    )
                seen_hashes[h] = len(combined)
                if not keep_runid or np.isnan(exp_copy.run_id):
                    new_id = next_run_id
                    next_run_id += 1
                    exp_copy.run_id = float(new_id)
                else:
                    new_id = old_id
                    next_run_id = max(next_run_id, new_id + 1)
                change_map[old_id] = new_id
                combined.append(exp_copy)
            runid_change_maps.append(change_map)

        nout = _infer_nargout(default=2, explicit_nargout=nargout)
        if nout == 1:
            return combined
        return combined, runid_change_maps
