"""Direct geometry disk chopper spectrometer instrument class (IX_inst_DGdisk).

Converted from:
  herbert_core/classes/instrument_classes/instrument/instruments/@IX_inst_DGdisk/
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.ix_divergence_profile import IX_divergence_profile
from pyhorace.data.instr_classes.ix_doubledisk_chopper import IX_doubledisk_chopper
from pyhorace.data.instr_classes.ix_inst import IX_inst
from pyhorace.data.instr_classes.ix_mod_shape_mono import IX_mod_shape_mono
from pyhorace.data.instr_classes.ix_moderator import IX_moderator


@register_serializable
class IX_inst_DGdisk(IX_inst):
    """Direct geometry disk chopper spectrometer class.

    Converted from MATLAB ``IX_inst_DGdisk``.
    """

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._mod_shape_mono: IX_mod_shape_mono = IX_mod_shape_mono()
        self._horiz_div: IX_divergence_profile = IX_divergence_profile()
        self._vert_div: IX_divergence_profile = IX_divergence_profile()
        self._name: str = ""

        if not args and not kwargs:
            return

        if len(args) == 1 and not kwargs and isinstance(args[0], IX_inst_DGdisk):
            other = args[0]
            self._mod_shape_mono = IX_mod_shape_mono(other._mod_shape_mono)
            self._horiz_div = IX_divergence_profile(other._horiz_div)
            self._vert_div = IX_divergence_profile(other._vert_div)
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

        # Support both forms of constructor:
        # Form 1 (6 or 7 args): IX_inst_DGdisk(moderator, shaping_chopper, mono_chopper, energy, horiz_div, vert_div, [name])
        # Form 2 (3 or 4 args): IX_inst_DGdisk(mod_shape_mono, horiz_div, vert_div, [name])
        if len(args) > 0 and isinstance(args[0], IX_mod_shape_mono):
            pos_params = [
                "mod_shape_mono",
                "horiz_div",
                "vert_div",
                "name",
            ]
        else:
            pos_params = [
                "moderator",
                "shaping_chopper",
                "mono_chopper",
                "energy",
                "horiz_div",
                "vert_div",
                "name",
            ]
        _, remains = self.set_positional_and_key_val_arguments(
            pos_params, True, *args, **kwargs
        )
        if remains:
            raise ValueError(f"Unrecognised extra arguments: {remains}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def mod_shape_mono(self) -> IX_mod_shape_mono:
        return self._mod_shape_mono

    @mod_shape_mono.setter
    def mod_shape_mono(self, val: IX_mod_shape_mono) -> None:
        if not isinstance(val, IX_mod_shape_mono):
            raise ValueError("HEBERT:IX_inst_DGdisk:invalid_argument: mod_shape_mono must be an IX_mod_shape_mono.")
        self._mod_shape_mono = IX_mod_shape_mono(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def moderator(self) -> IX_moderator:
        return self._mod_shape_mono.moderator

    @moderator.setter
    def moderator(self, val: IX_moderator) -> None:
        self._mod_shape_mono.moderator = val
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def source(self) -> IX_moderator:
        return self._mod_shape_mono.moderator

    @source.setter
    def source(self, val: IX_moderator) -> None:
        self.moderator = val

    @property
    def shaping_chopper(self) -> IX_doubledisk_chopper:
        return self._mod_shape_mono.shaping_chopper

    @shaping_chopper.setter
    def shaping_chopper(self, val: IX_doubledisk_chopper) -> None:
        self._mod_shape_mono.shaping_chopper = val
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def mono_chopper(self) -> IX_doubledisk_chopper:
        return self._mod_shape_mono.mono_chopper

    @mono_chopper.setter
    def mono_chopper(self, val: IX_doubledisk_chopper) -> None:
        self._mod_shape_mono.mono_chopper = val
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def energy(self) -> float:
        return self._mod_shape_mono.energy

    @energy.setter
    def energy(self, val: float) -> None:
        self._mod_shape_mono.energy = float(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def horiz_div(self) -> IX_divergence_profile:
        return self._horiz_div

    @horiz_div.setter
    def horiz_div(self, val: IX_divergence_profile) -> None:
        if not isinstance(val, IX_divergence_profile):
            raise ValueError("HEBERT:IX_inst_DGdisk:invalid_argument: horiz_div must be an IX_divergence_profile.")
        self._horiz_div = IX_divergence_profile(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def vert_div(self) -> IX_divergence_profile:
        return self._vert_div

    @vert_div.setter
    def vert_div(self, val: IX_divergence_profile) -> None:
        if not isinstance(val, IX_divergence_profile):
            raise ValueError("HEBERT:IX_inst_DGdisk:invalid_argument: vert_div must be an IX_divergence_profile.")
        self._vert_div = IX_divergence_profile(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def shaped_mod(self) -> bool:
        return self._mod_shape_mono.shaped_mod

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def saveable_fields(self) -> List[str]:
        return [
            "mod_shape_mono",
            "horiz_div",
            "vert_div",
            "name",
        ]

    def check_combo_arg(self) -> None:
        self._mod_shape_mono.check_combo_arg(True)
