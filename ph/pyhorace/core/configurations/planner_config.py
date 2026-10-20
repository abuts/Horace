"""Horace scan planner configuration class for PyHorace.

Translated from `horace_core/configuration/@planner_config/planner_config.m`.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .config_base import config_base
from .config_store import HoraceConfigError, config_store


class planner_config(config_base):
    """Class defines logging parameters of various Horace algorithms.
       When an algorithm is running, this config class recors time spent
       on each step of the algorithm and modifies algorithm logging 
       frequency to provide reasonable user experience to display
       algorithm progress with reasonable frequency.


    To see the list of current configuration option values:
        >>> planner_config()

    To set values:
        >>> pc = planner_config()
        >>> pc.name1 = val1
    or:
        >>> set(planner_config, 'name1', val1, 'name2', val2, ...)

    To fetch values:
        >>> pc = planner_config()
        >>> val1 = pc.name1
    or:
        >>> val1, val2, ... = get(planner_config, 'name1', 'name2', ...)

    `planner_config` properties are:
    --------------------------------
    - `par_file`  - The full path to the file containing detector positions (par file)
    - `u`         - 3-vector along beam direction
    - `v`         - 3-vector in scattering plane perpendicular to `u`
    - `ei`        - Incident energy (positive scalar)
    - `en_transf` - Energy transfer (scalar)
    - `psimin`    - Minimum psi angle (scalar)
    - `psimax`    - Maximum psi angle (scalar)
    - `alatt`     - Lattice parameters `[a, b, c]` (3-vector)
    - `angdeg`    - Lattice angles `[alpha, beta, gamma]` in degrees (3-vector)
    - `latpt`     - Lattice point `[h, k, l]` (3-vector)
    """

    _SAVED_PROPERTIES: List[str] = [
        "par_file",
        "u",
        "v",
        "ei",
        "en_transf",
        "psimin",
        "psimax",
        "alatt",
        "angdeg",
        "latpt",
    ]

    def __init__(self) -> None:
        super().__init__("planner_config")
        self.par_file_: str = ""
        self.u_: List[float] = [1, 0, 0]
        self.v_: List[float] = [0, 1, 0]
        self.ei_: float = 100.0
        self.en_transf_: float = 50.0
        self.psimin_: float = 0.0
        self.psimax_: float = 90.0
        self.alatt_: List[float] = [2.83, 2.83, 2.93]
        self.angdeg_: List[float] = [90, 90, 90]
        self.latpt_: List[float] = [1, 1, 1]

    @staticmethod
    def _validate_vec3(name: str, val: Any) -> List[float]:
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val)
            if arr.size == 3 and np.issubdtype(arr.dtype, np.number):
                return [float(x) for x in arr.reshape(-1)]
        raise HoraceConfigError(
            "HORACE:planner_config:invalid_argument",
            f"The input {name} should be numeric 3-vector. Actually it is: {val!r}",
        )

    @staticmethod
    def _validate_scalar(name: str, val: Any, positive: bool = False) -> float:
        if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
            fval = float(val)
            if positive and fval <= 0:
                raise HoraceConfigError(
                    "HORACE:planner_config:invalid_argument",
                    f"The input {name} should be single positive number. Actually it is: {val!r}",
                )
            return fval
        raise HoraceConfigError(
            "HORACE:planner_config:invalid_argument",
            f"The input {name} should be single number. Actually it is: {val!r}",
        )

    @property
    def par_file(self) -> str:
        """The file containing detector positions (par file)."""
        return str(self.get_or_restore_field("par_file"))

    @par_file.setter
    def par_file(self, val: str) -> None:
        if not isinstance(val, str):
            raise HoraceConfigError(
                "HORACE:planner_config:invalid_argument",
                f"The input file should be a string containing full file name. Actually it is: {val!r}",
            )
        config_store.instance().store_config(self, "par_file", val)

    @property
    def u(self) -> List[float]:
        """3-vector along beam direction."""
        return list(self.get_or_restore_field("u"))

    @u.setter
    def u(self, val: Any) -> None:
        vec = self._validate_vec3("u", val)
        config_store.instance().store_config(self, "u", vec)

    @property
    def v(self) -> List[float]:
        """3-vector in scattering plane."""
        return list(self.get_or_restore_field("v"))

    @v.setter
    def v(self, val: Any) -> None:
        vec = self._validate_vec3("v", val)
        config_store.instance().store_config(self, "v", vec)

    @property
    def ei(self) -> float:
        """Incident energy (single positive number)."""
        return float(self.get_or_restore_field("ei"))

    @ei.setter
    def ei(self, val: Any) -> None:
        sval = self._validate_scalar("Ei", val, positive=True)
        config_store.instance().store_config(self, "ei", sval)

    @property
    def en_transf(self) -> float:
        """Energy transfer (single number)."""
        return float(self.get_or_restore_field("en_transf"))

    @en_transf.setter
    def en_transf(self, val: Any) -> None:
        sval = self._validate_scalar("en_transf", val)
        config_store.instance().store_config(self, "en_transf", sval)

    @property
    def psimin(self) -> float:
        """Minimum psi angle (single number)."""
        return float(self.get_or_restore_field("psimin"))

    @psimin.setter
    def psimin(self, val: Any) -> None:
        sval = self._validate_scalar("psimin", val)
        config_store.instance().store_config(self, "psimin", sval)

    @property
    def psimax(self) -> float:
        """Maximum psi angle (single number)."""
        return float(self.get_or_restore_field("psimax"))

    @psimax.setter
    def psimax(self, val: Any) -> None:
        sval = self._validate_scalar("psimax", val)
        config_store.instance().store_config(self, "psimax", sval)

    @property
    def alatt(self) -> List[float]:
        """Lattice parameters `[a, b, c]` (numeric 3-vector)."""
        return list(self.get_or_restore_field("alatt"))

    @alatt.setter
    def alatt(self, val: Any) -> None:
        vec = self._validate_vec3("alatt", val)
        config_store.instance().store_config(self, "alatt", vec)

    @property
    def angdeg(self) -> List[float]:
        """Lattice angles `[alpha, beta, gamma]` in degrees (numeric 3-vector)."""
        return list(self.get_or_restore_field("angdeg"))

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        vec = self._validate_vec3("angdeg", val)
        config_store.instance().store_config(self, "angdeg", vec)

    @property
    def latpt(self) -> List[float]:
        """Lattice point `[h, k, l]` (numeric 3-vector)."""
        return list(self.get_or_restore_field("latpt"))

    @latpt.setter
    def latpt(self, val: Any) -> None:
        vec = self._validate_vec3("latpt", val)
        config_store.instance().store_config(self, "latpt", vec)

    def get_storage_field_names(self) -> List[str]:
        """Helper function returns the list of the public names of the fields
        which should be saved.
        """
        return list(self._SAVED_PROPERTIES)

    def get_default_value(self, field_name: str) -> Any:
        """Method gets internal field value bypassing standard get/set
        methods interface. Relies on assumption that each public field has a
        private field with name different by underscore.
        """
        val = getattr(self, f"{field_name}_")
        if isinstance(val, list):
            return list(val)
        return val


__all__ = ["planner_config"]
