"""Unit tests for sigvar and data_op_interface classes.

Translated from:
- _test/test_IX_classes/test_sigvar_ops1_2_Part1.m
- _test/test_sqw_pageOpMethods/test_data_op_order.m
"""

import numpy as np
import pytest

from pyhorace.core.data_op_interface import (
    DataOpArray,
    DataOpInterface,
    acos,
    acosh,
    acot,
    acoth,
    acsc,
    acsch,
    asec,
    asech,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    cot,
    coth,
    csc,
    csch,
    data_op_interface,
    exp,
    log,
    log10,
    minus,
    mldivide,
    mpower,
    mrdivide,
    mtimes,
    plus,
    sec,
    sech,
    sin,
    sinh,
    size_array_split,
    sqrt,
    tan,
    tanh,
    uminus,
    uplus,
)
from pyhorace.data.sigvar import SigVar, sigvar
from pyhorace.dataset.base import IXDataset


class TestSigvarOps12Part1:
    """Tests sigvar objects and binary_op_manager / binary_op_manager_single.

    Translated from _test/test_IX_classes/test_sigvar_ops1_2_Part1.m.
    """

    def test_sig_var_throws(self):
        sv = sigvar()

        # sigvar is not initialized by characters
        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            sv.sig_var = "a"

        # sigvar needs 2xNelements data
        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            sv.sig_var = np.random.rand(3, 3)

        # sigvar needs 2xNelements data
        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            sv.sig_var = np.random.rand(1, 5)

    def test_sig_var_set_get_col(self):
        sv = sigvar()
        data = np.random.rand(2, 10)

        sv.sig_var = data.T
        np.testing.assert_array_equal(sv.sig_var, data)
        np.testing.assert_array_equal(sv.s, data[0, :])
        np.testing.assert_array_equal(sv.e, data[1, :])

    def test_sig_var_set_get_row(self):
        sv = sigvar()
        data = np.random.rand(2, 10)

        sv.sig_var = data
        np.testing.assert_array_equal(sv.sig_var, data)
        np.testing.assert_array_equal(sv.s, data[0, :])
        np.testing.assert_array_equal(sv.e, data[1, :])

    # --------------------------------------------------------------------------
    # Test adding scalar objects
    # - tests the workings of binary_op_manager_single
    # --------------------------------------------------------------------------
    def test_wscal_scal__wscal_scal(self):
        # Add scalar objects with scalar signal
        w1 = sigvar(3, 25)
        w2 = sigvar(4, 144)
        wsum = sigvar(7, 169)

        wsum_test = w1 + w2
        assert wsum == wsum_test

        wsum_test = w2 + w1
        assert wsum == wsum_test

    def test_wscal_1by2__wscal_1by2(self):
        # Add scalar objects that have vector signal arrays
        w1 = sigvar([31, 5], [2, 3])
        w2 = sigvar([14, 16], 4)
        wsum = sigvar([45, 21], [6, 7])

        wsum_test = w1 + w2
        assert wsum == wsum_test

        wsum_test = w2 + w1
        assert wsum == wsum_test

    def test_wscal_1by2__wscal_2by1_FAIL(self):
        # Add scalar objects that have different size signal arrays
        w_1by2 = sigvar([31, 5], [2, 3])
        w_2by1 = sigvar([[14], [16]], 4)

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(w_1by2, w_2by1)

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(w_2by1, w_1by2)

    def test_wscal_scal__flt_scal(self):
        # Add scalar object with scalar signal to scalar float
        w = sigvar(3, 25)
        f = 4
        wsum = sigvar(7, 25)

        wsum_test = w + f
        assert wsum == wsum_test

        wsum_test = f + w
        assert wsum == wsum_test

    def test_wscal_1by2__flt_scal(self):
        # Add scalar object with vector signal to scalar float
        w = sigvar([31, 5], [2, 3])
        f = 7
        wsum = sigvar([38, 12], [2, 3])

        wsum_test = w + f
        assert wsum == wsum_test

        wsum_test = f + w
        assert wsum == wsum_test

    def test_wscal_1by2__flt_1by2(self):
        # Add scalar object to float array with same size as signal array
        w = sigvar([31, 5], [2, 3])
        f = np.array([14, 16])
        wsum = sigvar([45, 21], [2, 3])

        wsum_test = w + f
        assert wsum == wsum_test

        wsum_test = f + w
        assert wsum == wsum_test

    def test_wscal_1by2__flt_2by1_FAIL(self):
        # Add scalar object to float array with different shape to signal array
        w = sigvar([31, 5], [2, 3])
        f = np.array([[14], [16]])

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(w, f)

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(f, w)

    def test_wscal_1by2__flt_2by2_FAIL(self):
        # Add scalar object to float array with different size to signal array
        # Want to test that the float is *not* resolved into stack of arrays
        w = sigvar([31, 5], [2, 3])
        f = np.array([[14, 19], [16, 2]])

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(w, f)

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(f, w)

    # --------------------------------------------------------------------------
    # Test adding arrays of objects - neither a float array
    # --------------------------------------------------------------------------
    def test_w1by3_1by2__wscal_1by2(self):
        # Add object array to scalar object; elements with consistent signal arrays
        k1 = sigvar([31, 5], [2, 3])
        k2 = sigvar([14, 16], 4)
        k3 = sigvar([22, 18], [11, 12])
        k4 = sigvar([15, 14], [2, 3])

        w1 = DataOpArray([k1, k2, k3])
        w2 = k4
        wsum = DataOpArray([k1 + k4, k2 + k4, k3 + k4])

        wsum_test = w1 + w2
        assert np.array_equal(wsum, wsum_test)

        wsum_test = w2 + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__w1by3_mixed(self):
        # Add two objects that have elements with consistent signal arrays
        k11 = sigvar([31, 5], [2, 3])
        k12 = sigvar([[14], [16]], 4)
        k13 = sigvar([[22, 18, 14], [9, 11, -0.5]], [[11, 12, 6], [0.4, 0.8, 1.4]])

        k21 = sigvar([131, 15], [2, 3])
        k22 = sigvar([[24], [26]], 3)
        k23 = sigvar(
            [[122, 118, 114], [29, 211, -20.5]], [[311, 312, 36], [0.34, 0.38, 1.34]]
        )

        w1 = DataOpArray([k11, k12, k13])
        w2 = DataOpArray([k21, k22, k23])
        wsum = DataOpArray([k11 + k21, k12 + k22, k13 + k23])

        wsum_test = w1 + w2
        assert np.array_equal(wsum, wsum_test)

        wsum_test = w2 + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__w1by2_mixed_FAIL(self):
        # Add two objects with inconsistent sizes
        k11 = sigvar([31, 5], [2, 3])
        k12 = sigvar([[14], [16]], 4)
        k13 = sigvar([[22, 18, 14], [9, 11, -0.5]], [[11, 12, 6], [0.4, 0.8, 1.4]])

        k21 = sigvar([131, 15], [2, 3])
        k22 = sigvar([[24], [26]], 3)

        w1 = DataOpArray([k11, k12, k13])
        w2 = DataOpArray([k21, k22])

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w1, w2)

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w2, w1)

    # --------------------------------------------------------------------------
    # Test adding arrays of objects - one a float array
    # --------------------------------------------------------------------------
    def test_w1by3_mixed__flt_scal(self):
        # Vector object, scalar float
        k11 = sigvar([31, 5], [2, 3])
        k12 = sigvar([[14], [16]], 4)
        k13 = sigvar([[22, 18, 14], [9, 11, -0.5]], [[11, 12, 6], [0.4, 0.8, 1.4]])

        w1 = DataOpArray([k11, k12, k13])
        flt = 4

        wsum = DataOpArray([k11 + flt, k12 + flt, k13 + flt])

        wsum_test = w1 + flt
        assert np.array_equal(wsum, wsum_test)

        wsum_test = flt + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__flt_1by3(self):
        # Vector object, vector float same size as object array
        k11 = sigvar([31, 5], [2, 3])
        k12 = sigvar([[14], [16]], 4)
        k13 = sigvar([[22, 18, 14], [9, 11, -0.5]], [[11, 12, 6], [0.4, 0.8, 1.4]])

        w1 = DataOpArray([k11, k12, k13])
        flt = np.array([4, 5, 6])

        wsum = DataOpArray([k11 + flt[0], k12 + flt[1], k13 + flt[2]])

        wsum_test = w1 + flt
        assert np.array_equal(wsum, wsum_test)

        wsum_test = flt + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w3by1_mixed__flt_3by1(self):
        # Vector object, vector float same size as object array (column vectors)
        k11 = sigvar([31, 5], [2, 3])
        k12 = sigvar([[14], [16]], 4)
        k13 = sigvar([[22, 18, 14], [9, 11, -0.5]], [[11, 12, 6], [0.4, 0.8, 1.4]])

        w1 = DataOpArray([[k11], [k12], [k13]])
        flt = np.array([[4], [5], [6]])

        wsum = DataOpArray([[k11 + flt[0, 0]], [k12 + flt[1, 0]], [k13 + flt[2, 0]]])

        wsum_test = w1 + flt
        assert np.array_equal(wsum, wsum_test)

        wsum_test = flt + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__flt_3by1_FAIL(self):
        # Vector object, vector float same number but different size as object array
        k11 = sigvar([31, 5], [2, 3])
        k12 = sigvar([[14], [16]], 4)
        k13 = sigvar([[22, 18, 14], [9, 11, -0.5]], [[11, 12, 6], [0.4, 0.8, 1.4]])

        w1 = DataOpArray([k11, k12, k13])
        flt = np.array([[4], [5], [6]])

        with pytest.raises(ValueError, match="HERBERT:size_array_split:invalid_argument"):
            plus(w1, flt)

        with pytest.raises(ValueError, match="HERBERT:size_array_split:invalid_argument"):
            plus(flt, w1)

    def test_w1by3_2by1__flt_2by3(self):
        # Vector object, array float that can be resolved into stack
        k11 = sigvar([[31], [5]], [[2], [3]])
        k12 = sigvar([[14], [16]], 4)
        k13 = sigvar([[22], [18]], [[11], [12]])

        w1 = DataOpArray([k11, k12, k13])
        flt = np.array([[4, 5, 6], [11, 13, 15]])

        wsum = DataOpArray([k11 + flt[:, 0:1], k12 + flt[:, 1:2], k13 + flt[:, 2:3]])

        wsum_test = w1 + flt
        assert np.array_equal(wsum, wsum_test)

        wsum_test = flt + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_1by2__flt_2by3_FAIL(self):
        # Vector object, array float that can be resolved into stack
        # Fails as objects have size [1, 2] which does not match root size [2, 1]
        k11 = sigvar([31, 5], [2, 3])
        k12 = sigvar([14, 16], 4)
        k13 = sigvar([22, 18], [11, 12])

        w1 = DataOpArray([k11, k12, k13])
        flt = np.array([[4, 5, 6], [11, 13, 15]])

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(w1, flt)

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(flt, w1)

    def test_w1by3_mixed__flt_2by3_FAIL(self):
        # Vector object, array float that can be resolved into stack
        # Fails as objects do not all have the same signal sizes
        k11 = sigvar([[31], [5]], [[2], [3]])
        k12 = sigvar([[14], [16]], 4)
        k13 = sigvar([22, 18], [11, 12])

        w1 = DataOpArray([k11, k12, k13])
        flt = np.array([[4, 5, 6], [11, 13, 15]])

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(w1, flt)

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(flt, w1)


class TestSigvarPropertiesAndMethods:
    """Tests sigvar construction, properties, serialization, unary and binary ops."""

    def test_empty_construction(self):
        w = sigvar()
        assert w.n_elements == 0
        assert w.s.size == 0
        assert w.e.size == 0
        assert w.msk.size == 0
        assert not w.is_mask_defined
        nd, sz = w.dimensions()
        assert nd == ()
        assert sz == ()

    def test_construction_variants(self):
        # Signal only -> variance defaults to zeros on access
        w1 = sigvar([1.0, 2.0, 3.0])
        assert w1.n_elements == 3
        np.testing.assert_array_equal(w1.s, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(w1.e, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(w1.msk, [True, True, True])
        assert not w1.is_mask_defined

        # Signal, variance, and mask
        w2 = sigvar([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [True, False, True])
        assert w2.is_mask_defined
        np.testing.assert_array_equal(w2.msk, [True, False, True])

        # Copy from another sigvar
        w3 = sigvar(w2)
        assert w3 == w2

        # Keyword arguments
        w4 = sigvar(s=[1.0, 2.0], e=[0.5, 0.5])
        np.testing.assert_array_equal(w4.s, [1.0, 2.0])
        np.testing.assert_array_equal(w4.e, [0.5, 0.5])

        # Invalid negative variance
        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            sigvar([1.0, 2.0], [-0.1, 0.2])

        # Incompatible variance length
        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            sigvar([1.0, 2.0, 3.0], [0.1, 0.2])

    def test_dimensions_and_img_size(self):
        w_scal = sigvar(5.0, 1.0)
        assert w_scal.NUM_DIMS == 1
        assert w_scal.img_size == (1, 1)
        assert w_scal.dimensions() == (0, ())

        w_col = sigvar([[1.0], [2.0], [3.0]], [[0.1], [0.2], [0.3]])
        assert w_col.NUM_DIMS == 1
        assert w_col.img_size == (3, 1)
        assert w_col.dimensions() == (1, (3,))

        w_row = sigvar([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        assert w_row.NUM_DIMS == 1
        assert w_row.img_size == (1, 3)
        assert w_row.dimensions() == (2, (1, 3))

        w_2d = sigvar(np.ones((2, 4)), np.ones((2, 4)))
        assert w_2d.NUM_DIMS == 2
        assert w_2d.img_size == (2, 4)
        assert w_2d.dimensions() == (2, (2, 4))

    def test_serialization_and_old_struct(self):
        w = sigvar([10.0, 20.0], [1.0, 4.0], [True, False])
        assert w.class_version() == 2
        assert list(w.saveable_fields()) == ["s", "e", "msk"]

        s_dict = w.to_struct()
        w_restored = sigvar.from_struct(s_dict)
        assert w_restored == w

        # Old version struct with signal_, variance_, mask_
        old_struct = {
            "serial_name": "sigvar",
            "version": 1,
            "signal_": np.array([10.0, 20.0]),
            "variance_": np.array([1.0, 4.0]),
            "mask_": np.array([True, False]),
        }
        w_from_old = sigvar.loadobj(old_struct)
        assert w_from_old == w

    def test_binary_operations_formulas(self):
        s1 = np.array([4.0, 9.0])
        e1 = np.array([0.5, 1.5])
        s2 = np.array([2.0, 3.0])
        e2 = np.array([0.25, 0.75])
        w1 = sigvar(s1, e1)
        w2 = sigvar(s2, e2)

        # minus
        w_sub = w1 - w2
        np.testing.assert_allclose(w_sub.s, s1 - s2)
        np.testing.assert_allclose(w_sub.e, e1 + e2)

        # mtimes
        w_mul = w1 * w2
        np.testing.assert_allclose(w_mul.s, s1 * s2)
        np.testing.assert_allclose(w_mul.e, (s2**2) * e1 + (s1**2) * e2)

        # mrdivide (w1 / w2)
        w_rdiv = w1 / w2
        np.testing.assert_allclose(w_rdiv.s, s1 / s2)
        np.testing.assert_allclose(
            w_rdiv.e, (e1 / (s2**2) + e2 * ((s1 / (s2**2)) ** 2))
        )

        # mldivide (w1 \ w2 == w2 / w1)
        w_ldiv = mldivide(w1, w2)
        np.testing.assert_allclose(w_ldiv.s, s2 / s1)
        np.testing.assert_allclose(
            w_ldiv.e, (e2 / (s1**2) + e1 * ((s2 / (s1**2)) ** 2))
        )

        # mpower (w1 ** w2)
        w_pow = w1**w2
        expected_s = s1**s2
        expected_e = (expected_s**2) * (
            e1 * ((s2 / s1) ** 2) + e2 * (np.log(s1) ** 2)
        )
        np.testing.assert_allclose(w_pow.s, expected_s)
        np.testing.assert_allclose(w_pow.e, expected_e)

    def test_unary_operations_formulas(self):
        # Value in (0.2, 0.8) valid for sin, cos, tan, asin, acos, atan, atanh, sqrt, log, etc.
        x = np.array([0.4, 0.6])
        v = np.array([0.01, 0.02])
        w = sigvar(x, v)

        # uplus / uminus
        assert (+w) == w
        w_neg = -w
        np.testing.assert_allclose(w_neg.s, -x)
        np.testing.assert_allclose(w_neg.e, v)

        # sin / cos / tan
        w_sin = sin(w)
        np.testing.assert_allclose(w_sin.s, np.sin(x))
        np.testing.assert_allclose(w_sin.e, np.abs(1.0 - np.sin(x) ** 2) * v)

        w_cos = cos(w)
        np.testing.assert_allclose(w_cos.s, np.cos(x))
        np.testing.assert_allclose(w_cos.e, np.abs(1.0 - np.cos(x) ** 2) * v)

        w_tan = tan(w)
        np.testing.assert_allclose(w_tan.s, np.tan(x))
        np.testing.assert_allclose(w_tan.e, ((1.0 + np.tan(x) ** 2) ** 2) * v)

        # asin / acos / atan
        w_asin = asin(w)
        s_asin = np.arcsin(x)
        np.testing.assert_allclose(w_asin.s, s_asin)
        np.testing.assert_allclose(w_asin.e, v / np.abs(1.0 - s_asin**2))

        w_acos = acos(w)
        s_acos = np.arccos(x)
        np.testing.assert_allclose(w_acos.s, s_acos)
        np.testing.assert_allclose(w_acos.e, v / np.abs(1.0 - s_acos**2))

        w_atan = atan(w)
        s_atan = np.arctan(x)
        np.testing.assert_allclose(w_atan.s, s_atan)
        np.testing.assert_allclose(w_atan.e, v / ((1.0 + s_atan**2) ** 2))

        # sinh / cosh / tanh / asinh / atanh
        w_sinh = sinh(w)
        np.testing.assert_allclose(w_sinh.s, np.sinh(x))
        np.testing.assert_allclose(w_sinh.e, (1.0 + np.sinh(x) ** 2) * v)

        w_cosh = cosh(w)
        np.testing.assert_allclose(w_cosh.s, np.cosh(x))
        np.testing.assert_allclose(w_cosh.e, np.abs(np.cosh(x) ** 2 - 1.0) * v)

        w_tanh = tanh(w)
        np.testing.assert_allclose(w_tanh.s, np.tanh(x))
        np.testing.assert_allclose(w_tanh.e, ((1.0 - np.tanh(x) ** 2) ** 2) * v)

        w_asinh = asinh(w)
        s_asinh = np.arcsinh(x)
        np.testing.assert_allclose(w_asinh.s, s_asinh)
        np.testing.assert_allclose(w_asinh.e, v / (1.0 + s_asinh**2))

        w_atanh = atanh(w)
        np.testing.assert_allclose(w_atanh.s, np.arctanh(x))
        np.testing.assert_allclose(w_atanh.e, v / ((1.0 - np.arctanh(x) ** 2) ** 2))

        # exp / log / log10 / sqrt
        w_exp = exp(w)
        np.testing.assert_allclose(w_exp.s, np.exp(x))
        np.testing.assert_allclose(w_exp.e, (np.exp(x) ** 2) * v)

        w_log = log(w)
        np.testing.assert_allclose(w_log.s, np.log(x))
        np.testing.assert_allclose(w_log.e, v / (x**2))

        w_log10 = log10(w)
        np.testing.assert_allclose(w_log10.s, np.log10(x))
        np.testing.assert_allclose(w_log10.e, v / ((x * np.log(10.0)) ** 2))

        w_sqrt = sqrt(w)
        np.testing.assert_allclose(w_sqrt.s, np.sqrt(x))
        np.testing.assert_allclose(w_sqrt.e, v / (4.0 * x))

        # Reciprocal trig/hyperbolic functions: csc, sec, cot, csch, sech, coth, asech, acsch, acot
        w_csc = csc(w)
        s_csc = 1.0 / np.sin(x)
        np.testing.assert_allclose(w_csc.s, s_csc)
        np.testing.assert_allclose(w_csc.e, (s_csc**2) * np.abs(s_csc**2 - 1.0) * v)

        w_sec = sec(w)
        s_sec = 1.0 / np.cos(x)
        np.testing.assert_allclose(w_sec.s, s_sec)
        np.testing.assert_allclose(w_sec.e, (s_sec**2) * np.abs(s_sec**2 - 1.0) * v)

        w_cot = cot(w)
        s_cot = 1.0 / np.tan(x)
        np.testing.assert_allclose(w_cot.s, s_cot)
        np.testing.assert_allclose(w_cot.e, ((1.0 + s_cot**2) ** 2) * v)

        w_csch = csch(w)
        s_csch = 1.0 / np.sinh(x)
        np.testing.assert_allclose(w_csch.s, s_csch)
        np.testing.assert_allclose(w_csch.e, (s_csch**2) * (1.0 + s_csch**2) * v)

        w_sech = sech(w)
        s_sech = 1.0 / np.cosh(x)
        np.testing.assert_allclose(w_sech.s, s_sech)
        np.testing.assert_allclose(w_sech.e, (s_sech**2) * np.abs(1.0 - s_sech**2) * v)

        w_coth = coth(w)
        s_coth = 1.0 / np.tanh(x)
        np.testing.assert_allclose(w_coth.s, s_coth)
        np.testing.assert_allclose(w_coth.e, ((1.0 - s_coth**2) ** 2) * v)

        w_asech = asech(w)
        s_asech = np.arccosh(1.0 / x)
        np.testing.assert_allclose(w_asech.s, s_asech)
        np.testing.assert_allclose(w_asech.e, v / (np.abs(1.0 - s_asech**2) * (s_asech**2)))

        w_acsch = acsch(w)
        s_acsch = np.arcsinh(1.0 / x)
        np.testing.assert_allclose(w_acsch.s, s_acsch)
        np.testing.assert_allclose(w_acsch.e, v / ((s_acsch**2 + 1.0) * (s_acsch**2)))

        w_acot = acot(w)
        s_acot = np.arctan(1.0 / x)
        np.testing.assert_allclose(w_acot.s, s_acot)
        np.testing.assert_allclose(w_acot.e, v / ((1.0 + s_acot**2) ** 2))

        # Values > 1 for acosh, asec, acsc, acoth
        x_gt1 = np.array([1.5, 2.5])
        w_gt1 = sigvar(x_gt1, v)

        w_acosh = acosh(w_gt1)
        s_acosh = np.arccosh(x_gt1)
        np.testing.assert_allclose(w_acosh.s, s_acosh)
        np.testing.assert_allclose(w_acosh.e, v / np.abs(s_acosh**2 - 1.0))

        w_asec = asec(w_gt1)
        s_asec = np.arccos(1.0 / x_gt1)
        np.testing.assert_allclose(w_asec.s, s_asec)
        np.testing.assert_allclose(w_asec.e, v / (np.abs(s_asec**2 - 1.0) * (s_asec**2)))

        w_acsc = acsc(w_gt1)
        s_acsc = np.arcsin(1.0 / x_gt1)
        np.testing.assert_allclose(w_acsc.s, s_acsc)
        np.testing.assert_allclose(w_acsc.e, v / (np.abs(s_acsc**2 - 1.0) * (s_acsc**2)))

        w_acoth = acoth(w_gt1)
        s_acoth = np.arctanh(1.0 / x_gt1)
        np.testing.assert_allclose(w_acoth.s, s_acoth)
        np.testing.assert_allclose(w_acoth.e, v / ((1.0 - s_acoth**2) ** 2))


class TestSigvarIXDatasetOpsPart2:
    """Port of sigvar + IX_dataset binary operation tests from test_IX_dataset_1d_ops1_2_Part2.m."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.p1 = IXDataset(signal=rng.random((20, 1)), error=rng.random((20, 1)), title="p1")
        self.p2 = IXDataset(signal=rng.random((20, 1)), error=rng.random((20, 1)), title="p2")
        self.p3 = IXDataset(signal=rng.random((20, 1)), error=rng.random((20, 1)), title="p3")
        self.pbig = IXDataset(signal=rng.random((50, 1)), error=rng.random((50, 1)), title="pbig")

    def test_wscal_20by1__wscal_20by1(self):
        rng = np.random.default_rng(101)
        w1 = self.p1
        w2 = sigvar(rng.random(w1.signal.shape), rng.random(w1.error.shape))

        s = w1.signal + w2.s
        var = (w1.error**2) + w2.e

        wsum = w1.copy()
        wsum.signal = s
        wsum.error = np.sqrt(var)

        wsum_test = w1 + w2
        assert wsum == wsum_test

        wsum_sv = sigvar(s, var)
        wsum_test_sv = w2 + w1
        assert wsum_sv == wsum_test_sv

    def test_wscal_20by1__wscal_21by1_FAIL(self):
        rng = np.random.default_rng(102)
        w1 = self.p1
        w2 = sigvar(rng.random((21, 1)), rng.random((21, 1)))

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w1, w2)

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(w2, w1)

    def test_w1by3_1by2__wscal_1by2(self):
        rng = np.random.default_rng(103)
        k1, k2, k3 = self.p1, self.p2, self.p3
        k4 = sigvar(rng.random(k1.signal.shape), rng.random(k1.error.shape))

        w1 = DataOpArray([k1, k2, k3])
        w2 = k4

        wsum = DataOpArray([k1 + k4, k2 + k4, k3 + k4])
        wsum_test = w1 + w2
        assert np.array_equal(wsum, wsum_test)

        wsum2 = DataOpArray([k4 + k1, k4 + k2, k4 + k3])
        wsum_test2 = w2 + w1
        assert np.array_equal(wsum2, wsum_test2)

    def test_w1by3_mixed__w1by3_mixed(self):
        rng = np.random.default_rng(104)
        k11, k12, k13 = self.p1, self.p2, self.pbig
        k21 = sigvar(rng.random(k11.signal.shape), rng.random(k11.error.shape))
        k22 = sigvar(rng.random(k12.signal.shape), rng.random(k12.error.shape))
        k23 = sigvar(rng.random(k13.signal.shape), rng.random(k13.error.shape))

        w1 = DataOpArray([k11, k12, k13])
        w2 = DataOpArray([k21, k22, k23])

        wsum = DataOpArray([k11 + k21, k12 + k22, k13 + k23])
        wsum_test = w1 + w2
        assert np.array_equal(wsum, wsum_test)

        wsum2 = DataOpArray([k21 + k11, k22 + k12, k23 + k13])
        wsum_test2 = w2 + w1
        assert np.array_equal(wsum2, wsum_test2)

    def test_w1by3_mixed__w1by2_mixed_FAIL(self):
        rng = np.random.default_rng(105)
        k11, k12, k13 = self.p1, self.p2, self.pbig
        k21 = sigvar(rng.random(k11.signal.shape), rng.random(k11.error.shape))
        k22 = sigvar(rng.random(k12.signal.shape), rng.random(k12.error.shape))

        w1 = DataOpArray([k11, k12, k13])
        w2 = DataOpArray([k21, k22])

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w1, w2)

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w2, w1)


class TestDataOpInterfaceOrderAndAbstract:
    """Tests data_op_interface abstract contract and operation order / priority."""

    def test_abstract_interface(self):
        # Cannot instantiate data_op_interface directly
        with pytest.raises(TypeError):
            data_op_interface()

        # Incomplete subclass without get_NUM_DIMS / get_img_size cannot be instantiated
        class IncompleteDataOp(data_op_interface):
            pass

        with pytest.raises(TypeError):
            IncompleteDataOp()

    def test_priority_and_operation_order(self):
        sv_scal = sigvar(3.0, 1.0)
        sv_img = sigvar([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        ix_img = IXDataset(signal=np.array([10.0, 20.0, 30.0]), error=np.array([1.0, 1.0, 1.0]))

        prio, sv_size, has_pix, has_img = data_op_interface.get_priority(5.0)
        assert prio == 0
        assert sv_size == (1, 1)
        assert not has_pix
        assert not has_img

        prio, sv_size, has_pix, has_img = data_op_interface.get_priority(sv_scal)
        assert prio == 1
        assert sv_size == (1, 1)
        assert not has_pix
        assert not has_img

        prio, sv_size, has_pix, has_img = data_op_interface.get_priority(sv_img)
        assert prio == 11
        assert sv_size == (1, 3)
        assert not has_pix
        assert has_img

        prio, sv_size, has_pix, has_img = data_op_interface.get_priority(ix_img)
        assert prio == 12
        assert sv_size == (1, 3)
        assert not has_pix
        assert has_img

        # Operation order between sv_img (prio 11) and ix_img (prio 12)
        flip, op_kind = data_op_interface.get_operation_order(sv_img, ix_img)
        assert flip is True
        assert op_kind == 0

        flip, op_kind = data_op_interface.get_operation_order(ix_img, sv_img)
        assert flip is False
        assert op_kind == 0

        # Invalid operation kind when op1 has no pixels and op2 has pixels
        with pytest.raises(RuntimeError, match="HERBERT:data_op_interface:runtime_error"):
            data_op_interface.get_operation_kind(False, True, True, True)
