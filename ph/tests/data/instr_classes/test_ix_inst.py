"""Unit tests for IX_inst, IX_null_inst, IX_inst_DGfermi, and IX_inst_DGdisk.

Converted from:
  _test/test_instrument_classes/test_IX_inst.m
"""
import numpy as np
import pytest

from pyhorace.data.instr_classes.ix_aperture import IX_aperture
from pyhorace.data.instr_classes.ix_divergence_profile import IX_divergence_profile
from pyhorace.data.instr_classes.ix_doubledisk_chopper import IX_doubledisk_chopper
from pyhorace.data.instr_classes.ix_fermi_chopper import IX_fermi_chopper
from pyhorace.data.instr_classes.ix_inst import IX_inst
from pyhorace.data.instr_classes.ix_inst_dgdisk import IX_inst_DGdisk
from pyhorace.data.instr_classes.ix_inst_dgfermi import IX_inst_DGfermi
from pyhorace.data.instr_classes.ix_mod_shape_mono import IX_mod_shape_mono
from pyhorace.data.instr_classes.ix_moderator import IX_moderator
from pyhorace.data.instr_classes.ix_null_inst import IX_null_inst


def _maps_components():
    """Return MAPS instrument test components matching maps_instrument_struct_for_tests.m."""
    aperture = IX_aperture(-10.0, 0.0642, 0.0642)
    moderator = IX_moderator(
        12.0,
        32.0,
        "ikcarp",
        [4.2932, 25.0, 0.05],
        temperature=300.0,
        thickness=0.05,
        width=0.12,
        height=0.12,
        energy=300.0,
        name="Water",
    )
    fermi_chopper = IX_fermi_chopper(
        1.851,
        400.0,
        0.049,
        1.3,
        0.00228,
        width=0.052,
        height=0.065,
        energy=300.0,
    )
    ang = np.array([-3.0, -2.0, 0.0, 2.0, 3.0]) / 100.0
    prof = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    horiz_div = IX_divergence_profile(ang, prof)
    vert_div = IX_divergence_profile(ang, prof)
    return moderator, aperture, fermi_chopper, horiz_div, vert_div


def _let_components():
    """Return LET instrument test components matching let_instrument_struct_for_tests.m."""
    moderator = IX_moderator(
        23.5,
        0.0,
        "ikcarp",
        [33.876, 80.0, 0.0],
        temperature=25.0,
        thickness=0.03,
        width=0.04,
        height=0.04,
        energy=3.7,
        name="Hydrogen",
    )
    shaping_chopper = IX_doubledisk_chopper(
        15.67,
        70.0,
        0.29,
        0.031,
        aperture_width=0.04,
        aperture_height=0.056,
        jitter=0.0,
        name="Pulse shaping",
    )
    mono_chopper = IX_doubledisk_chopper(
        1.5,
        240.0,
        0.29,
        0.015,
        aperture_width=0.04,
        aperture_height=0.045,
        jitter=0.0,
        name="Monochromating",
    )
    ang = np.array([-3.0, -2.0, 0.0, 2.0, 3.0]) / 100.0
    prof = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    horiz_div = IX_divergence_profile(ang, prof)
    vert_div = IX_divergence_profile(ang, prof)
    return moderator, shaping_chopper, mono_chopper, horiz_div, vert_div


class TestIXInst:
    """Test suite for IX_inst, IX_null_inst, IX_inst_DGfermi, and IX_inst_DGdisk."""

    def test_null_inst(self) -> None:
        null1 = IX_null_inst()
        null2 = IX_null_inst()
        assert null1 == null2
        assert null1.name == ""
        d = null1.to_dict()
        assert IX_null_inst.from_dict(d) == null1

    def test_ix_inst_base(self) -> None:
        mod, _, _, _, _ = _maps_components()
        inst = IX_inst(mod, "BaseInst")
        assert inst.name == "BaseInst"
        assert inst.source == mod

    def test_ix_inst_dgfermi_construction_and_energy_sync(self) -> None:
        mod, ap, fc, hdiv, vdiv = _maps_components()
        inst = IX_inst_DGfermi(mod, ap, fc, 300.0, hdiv, vdiv, name="MAPS")
        assert inst.name == "MAPS"
        assert inst.energy == 300.0
        assert inst.moderator.energy == 300.0
        assert inst.fermi_chopper.energy == 300.0

        # Changing instrument energy updates moderator and fermi_chopper energies
        inst.energy = 450.0
        assert inst.moderator.energy == 450.0
        assert inst.fermi_chopper.energy == 450.0

        # Legacy v1 constructor form with etype ('meV' or 'ang')
        inst_v1 = IX_inst_DGfermi(mod, ap, fc, 250.0, "meV", "MAPS_v1")
        assert inst_v1.name == "MAPS_v1"
        assert np.isclose(inst_v1.energy, 250.0)

        inst_k = IX_inst_DGfermi(mod, ap, fc, 10.0, "ang", "MAPS_k")
        assert np.isclose(inst_k.energy, (10.0 / 0.694692) ** 2, rtol=1e-12)

    def test_ix_inst_dgfermi_divergence_pair_validation(self) -> None:
        mod, ap, fc, hdiv, _ = _maps_components()
        with pytest.raises(ValueError):
            IX_inst_DGfermi(mod, ap, fc, 300.0, horiz_div=hdiv)

    def test_ix_inst_dgdisk_construction_both_forms(self) -> None:
        mod, shape_ch, mono_ch, hdiv, vdiv = _let_components()

        # Form 1: individual components
        inst1 = IX_inst_DGdisk(mod, shape_ch, mono_chopper=mono_ch, energy=3.7, horiz_div=hdiv, vert_div=vdiv, name="LET")
        assert inst1.name == "LET"
        assert inst1.energy == 3.7
        assert inst1.moderator.energy == 3.7
        assert inst1.shaped_mod is False

        # Form 2: IX_mod_shape_mono object
        msm = IX_mod_shape_mono(mod, shape_ch, mono_ch, 3.7)
        inst2 = IX_inst_DGdisk(msm, hdiv, vdiv, "LET")
        assert inst1 == inst2

        # Mutating moderator parameter in place on IX_inst_DGdisk
        inst2.moderator.pp[0] = 200.0
        assert inst2.shaped_mod is True

    def test_serialization_roundtrip_dgfermi_and_dgdisk(self) -> None:
        mod, ap, fc, hdiv, vdiv = _maps_components()
        fermi_inst = IX_inst_DGfermi(mod, ap, fc, 300.0, hdiv, vdiv, name="MAPS")
        fermi_rec = IX_inst_DGfermi.from_dict(fermi_inst.to_dict())
        assert fermi_rec == fermi_inst

        mod_l, shape_ch, mono_ch, hdiv_l, vdiv_l = _let_components()
        disk_inst = IX_inst_DGdisk(mod_l, shape_ch, mono_ch, 3.7, hdiv_l, vdiv_l, "LET")
        disk_rec = IX_inst_DGdisk.from_dict(disk_inst.to_dict())
        assert disk_rec == disk_inst
