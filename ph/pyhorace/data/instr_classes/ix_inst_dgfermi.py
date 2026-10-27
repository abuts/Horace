"""Direct geometry Fermi chopper spectrometer instrument class (IX_inst_DGfermi).

Converted from:
  herbert_core/classes/instrument_classes/instrument/instruments/@IX_inst_DGfermi/
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.ix_aperture import IX_aperture
from pyhorace.data.instr_classes.ix_divergence_profile import IX_divergence_profile
from pyhorace.data.instr_classes.ix_fermi_chopper import IX_fermi_chopper
from pyhorace.data.instr_classes.ix_inst import IX_inst
from pyhorace.data.instr_classes.ix_moderator import IX_moderator


def _parse_energy_and_type(
    energy: float, etype: Union[str, int, float] = "meV"
) -> float:
    """Convert incident energy or wavevector into energy in meV.

    Matches ``@IX_inst/private/get_efix.m``:
    - ``etype == 'mev'`` or ``1``: energy in meV (must be > 0)
    - ``etype == 'ang'`` or ``-1``: wavevector ``k`` in Ang^-1 (``E = (k / 0.694692)^2``)
    """
    k_to_e = 1.0 / (0.694692 ** 2)  # 2.0721248...
    val = float(energy)
    if isinstance(etype, str):
        et = etype.strip().lower()
        if "mev".startswith(et):
            mode = 1
        elif "ang".startswith(et):
            mode = -1
        else:
            raise ValueError(f"HEBERT:IX_inst_DGfermi:invalid_argument: Unrecognised energy type '{etype}'.")
    elif isinstance(etype, (int, float)) and not isinstance(etype, bool):
        if etype == 1:
            mode = 1
        elif etype == -1:
            mode = -1
        else:
            raise ValueError(f"HEBERT:IX_inst_DGfermi:invalid_argument: Invalid numeric energy type '{etype}'.")
    else:
        raise ValueError("HEBERT:IX_inst_DGfermi:invalid_argument: Invalid energy type.")

    if val <= 0:
        raise ValueError("HEBERT:IX_inst_DGfermi:invalid_argument: Energy or wavevector must be greater than zero.")
    if mode == 1:
        return val
    return k_to_e * (val ** 2)


@register_serializable
class IX_inst_DGfermi(IX_inst):
    """Direct geometry Fermi chopper spectrometer class.

    Converted from MATLAB ``IX_inst_DGfermi``.
    """

    def class_version(self) -> int:
        return 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._moderator: IX_moderator = IX_moderator()
        self._aperture: IX_aperture = IX_aperture()
        self._fermi_chopper: IX_fermi_chopper = IX_fermi_chopper()
        self._energy: float = 0.0
        self._horiz_div: IX_divergence_profile = IX_divergence_profile()
        self._vert_div: IX_divergence_profile = IX_divergence_profile()
        self._has_divergence: bool = False
        self._name: str = ""

        if not args and not kwargs:
            return

        if len(args) == 1 and not kwargs and isinstance(args[0], IX_inst_DGfermi):
            other = args[0]
            self._moderator = IX_moderator(other._moderator)
            self._aperture = IX_aperture(other._aperture)
            self._fermi_chopper = IX_fermi_chopper(other._fermi_chopper)
            self._energy = other._energy
            self._horiz_div = IX_divergence_profile(other._horiz_div)
            self._vert_div = IX_divergence_profile(other._vert_div)
            self._has_divergence = other._has_divergence
            self._name = other._name
            return

        if len(args) == 1 and not kwargs and isinstance(args[0], dict):
            d = args[0]
            self._do_check_combo_arg = False
            for k, v in d.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._do_check_combo_arg = True
            self.check_combo_arg()
            return

        # Support old v1 positional form:
        # IX_inst_DGfermi(moderator, aperture, fermi_chopper, energy, etype, name)
        # and v2 positional form:
        # IX_inst_DGfermi(moderator, aperture, fermi_chopper, energy, horiz_div, vert_div, name)
        pos_list = list(args)
        kw = dict(kwargs)

        # Check if 5th positional argument is an energy type string ('meV', 'ang') or +/-1
        if len(pos_list) >= 5 and isinstance(pos_list[4], (str, int, float)) and not isinstance(pos_list[4], IX_divergence_profile):
            etype_arg = pos_list.pop(4)
            pos_list[3] = _parse_energy_and_type(pos_list[3], etype_arg)
            if len(pos_list) == 5 and isinstance(pos_list[4], str) and "name" not in kw:
                kw["name"] = pos_list.pop(4)

        pos_params = [
            "moderator",
            "aperture",
            "fermi_chopper",
            "energy",
            "horiz_div",
            "vert_div",
            "name",
        ]
        _, remains = self.set_positional_and_key_val_arguments(
            pos_params, True, *pos_list, **kw
        )
        if remains:
            raise ValueError(f"Unrecognised extra arguments: {remains}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def moderator(self) -> IX_moderator:
        return self._moderator

    @moderator.setter
    def moderator(self, val: IX_moderator) -> None:
        if not isinstance(val, IX_moderator):
            raise ValueError("HEBERT:IX_inst_DGfermi:invalid_argument: Moderator must be an IX_moderator object.")
        self._moderator = IX_moderator(val)
        if self._energy > 0:
            self._moderator.energy = self._energy
        elif self._moderator.energy > 0:
            self._energy = self._moderator.energy
            if all(self._fermi_chopper._mandatory_field_set):
                self._fermi_chopper.energy = self._energy
            else:
                self._fermi_chopper._energy = self._energy
        self.clear_hash()

    @property
    def source(self) -> IX_moderator:
        return self._moderator

    @source.setter
    def source(self, val: IX_moderator) -> None:
        self.moderator = val

    @property
    def aperture(self) -> IX_aperture:
        return self._aperture

    @aperture.setter
    def aperture(self, val: IX_aperture) -> None:
        if not isinstance(val, IX_aperture):
            raise ValueError("HEBERT:IX_inst_DGfermi:invalid_argument: Aperture must be an IX_aperture object.")
        self._aperture = IX_aperture(val)

    @property
    def fermi_chopper(self) -> IX_fermi_chopper:
        return self._fermi_chopper

    @fermi_chopper.setter
    def fermi_chopper(self, val: IX_fermi_chopper) -> None:
        if not isinstance(val, IX_fermi_chopper):
            raise ValueError("HEBERT:IX_inst_DGfermi:invalid_argument: Fermi chopper must be an IX_fermi_chopper object.")
        self._fermi_chopper = IX_fermi_chopper(val)
        if self._energy > 0:
            self._fermi_chopper.energy = self._energy
        elif self._fermi_chopper.energy > 0:
            self._energy = self._fermi_chopper.energy
            self._moderator.energy = self._energy

    @property
    def energy(self) -> float:
        return self._energy

    @energy.setter
    def energy(self, val: float) -> None:
        v = float(val)
        if v < 0:
            raise ValueError("HEBERT:IX_inst_DGfermi:invalid_argument: Incident energy must be >= 0.")
        self._energy = v
        self._moderator.energy = v
        if all(self._fermi_chopper._mandatory_field_set):
            self._fermi_chopper.energy = v
        else:
            self._fermi_chopper._energy = v
        self.clear_hash()

    @property
    def horiz_div(self) -> IX_divergence_profile:
        return self._horiz_div

    @horiz_div.setter
    def horiz_div(self, val: Optional[IX_divergence_profile]) -> None:
        if val is None:
            self._horiz_div = IX_divergence_profile()
            return
        if not isinstance(val, IX_divergence_profile):
            raise ValueError("HEBERT:IX_inst_DGfermi:invalid_argument: horiz_div must be an IX_divergence_profile.")
        self._horiz_div = IX_divergence_profile(val)
        self._has_divergence = True
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def vert_div(self) -> IX_divergence_profile:
        return self._vert_div

    @vert_div.setter
    def vert_div(self, val: Optional[IX_divergence_profile]) -> None:
        if val is None:
            self._vert_div = IX_divergence_profile()
            return
        if not isinstance(val, IX_divergence_profile):
            raise ValueError("HEBERT:IX_inst_DGfermi:invalid_argument: vert_div must be an IX_divergence_profile.")
        self._vert_div = IX_divergence_profile(val)
        self._has_divergence = True
        if self._do_check_combo_arg:
            self.check_combo_arg()

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def saveable_fields(self) -> List[str]:
        return [
            "moderator",
            "aperture",
            "fermi_chopper",
            "energy",
            "horiz_div",
            "vert_div",
            "name",
        ]

    def check_combo_arg(self) -> None:
        if self._energy > 0:
            self._moderator.energy = self._energy
            self._fermi_chopper.energy = self._energy
        if self._has_divergence:
            h_empty = self._horiz_div.angles.size == 0
            v_empty = self._vert_div.angles.size == 0
            if h_empty != v_empty:
                raise ValueError(
                    "HEBERT:IX_inst_DGfermi:invalid_argument: "
                    "Both horiz_div and vert_div must be provided together."
                )

    def convert_old_struct(
        self, inputs: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], IX_inst_DGfermi]:
        out = dict(inputs)
        if out.get("class_version_") in (0, 1) or "moderator_" in out or "name_" in out:
            out.pop("class_version_", None)
            out = {(k[:-1] if k.endswith("_") else k): v for k, v in out.items()}
        if ver is None or ver == 1 or (isinstance(ver, float) and math.isnan(ver)):
            out.setdefault("horiz_div", IX_divergence_profile())
            out.setdefault("vert_div", IX_divergence_profile())
            if "source" in out and "moderator" not in out and isinstance(out["source"], IX_moderator):
                out["moderator"] = out.pop("source")
            elif "source" in out and "moderator" in out:
                out.pop("source", None)
        return out, self
