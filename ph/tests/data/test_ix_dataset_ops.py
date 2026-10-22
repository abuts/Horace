"""Unit tests for IX_dataset_1d binary operations with IX_dataset_1d, floats, and sigvar.

Translated from:
- _test/test_IX_classes/test_IX_dataset_1d_ops1_2_Part1.m
- _test/test_IX_classes/test_IX_dataset_1d_ops1_2_Part2.m
"""

import numpy as np
import pytest

from pyhorace.core.data_op_interface import DataOpArray, plus
from pyhorace.data import IX_axis, IX_dataset_1d, sigvar


@pytest.fixture
def ref_datasets():
    """Deterministic equivalent of testdata_IX_datasets_ref.mat."""
    rng = np.random.default_rng(42)
    x = 20.0 * np.sort(rng.random(20))
    xb = 20.0 * np.sort(rng.random(21))

    y1 = 5.0 * (0.5 + rng.random(20))
    e1 = 0.5 + rng.random(20)
    y2 = 5.0 * (0.5 + rng.random(20))
    e2 = 0.5 + rng.random(20)
    y3 = 5.0 * (0.5 + rng.random(20))
    e3 = 0.5 + rng.random(20)

    ax = IX_axis("Energy transfer", "meV", "$w")
    p1 = IX_dataset_1d(x, y1, e1, "Point data, not distribution", ax, "Counts", False)
    p2 = IX_dataset_1d(x, y2, e2, "Point data, not distribution", ax, "Counts", False)
    p3 = IX_dataset_1d(x, y3, e3, "Point data, not distribution", ax, "Counts", False)

    h1 = IX_dataset_1d(xb, y1, e1, "Histogram data, distribution", ax, "Counts", True)
    h2 = IX_dataset_1d(xb, y2, e2, "Histogram data, distribution", ax, "Counts", True)
    h3 = IX_dataset_1d(xb, y3, e3, "Histogram data, distribution", ax, "Counts", True)

    nx_big = 516
    x_big = nx_big * np.sort(rng.random(nx_big))
    xb_big = nx_big * np.sort(rng.random(nx_big + 1))
    y_big = 10.0 * np.exp(-0.5 * ((x_big - 250.0) / 125.0) ** 2)
    e_big = 0.5 + rng.random(nx_big)
    pbig = IX_dataset_1d(x_big, y_big, e_big, "Point data, distribution", ax, "Counts", True)
    hbig = IX_dataset_1d(xb_big, y_big, e_big, "Point data, distribution", ax, "Counts", True)

    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "pbig": pbig,
        "hbig": hbig,
    }


class TestIXDataset1dOpsPart1:
    """Tests adding IX_dataset_1d objects to themselves or floats (Part 1)."""

    def test_wscal_20by1__wscal_20by1(self, ref_datasets):
        w1 = ref_datasets["p1"].copy()
        w2 = ref_datasets["p2"].copy()

        s = w1.signal + w2.signal
        e = np.sqrt(w1.error**2 + w2.error**2)

        wsum = w1.copy()
        wsum.signal = s
        wsum.error = e
        wsum_test = w1 + w2
        assert wsum == wsum_test

        wsum = w2.copy()
        wsum.signal = s
        wsum.error = e
        wsum_test = w2 + w1
        assert wsum == wsum_test

    def test_wscal_20by1__wscal_516by1_FAIL(self, ref_datasets):
        w1 = ref_datasets["p1"]
        w2 = ref_datasets["pbig"]

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w1, w2)

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w2, w1)

    def test_wscal_20by1__flt_scal(self, ref_datasets):
        w = ref_datasets["p1"].copy()
        f = 4.0

        wsum = w.copy()
        wsum.signal = w.signal + 4.0

        wsum_test = w + f
        assert wsum == wsum_test

        wsum_test = f + w
        assert wsum == wsum_test

    def test_wscal_20by1__flt_20by1(self, ref_datasets):
        w = ref_datasets["p1"].copy()
        rng = np.random.default_rng(101)
        f = rng.random(w.signal.shape)

        wsum = w.copy()
        wsum.signal = w.signal + f

        wsum_test = w + f
        assert wsum == wsum_test

        wsum_test = f + w
        assert wsum == wsum_test

    def test_wscal_20by1__flt_1by20_FAIL(self, ref_datasets):
        w = ref_datasets["p1"]
        rng = np.random.default_rng(102)
        f = rng.random((1, w.signal.size))

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w, f)

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(f, w)

    def test_wscal_20by1__flt_20by20_FAIL(self, ref_datasets):
        w = ref_datasets["p1"]
        rng = np.random.default_rng(103)
        n = w.signal.size
        f = rng.random((n, n))

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w, f)

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(f, w)

    def test_w1by3_20by1__wscal_20by1(self, ref_datasets):
        k1 = ref_datasets["p1"]
        k2 = ref_datasets["p2"]
        k3 = ref_datasets["p3"]
        k4 = ref_datasets["h1"]

        w1 = DataOpArray.from_list([k1, k2, k3], shape=(1, 3))
        w2 = k4

        wsum = DataOpArray.from_list([k1 + k4, k2 + k4, k3 + k4], shape=(1, 3))
        wsum_test = w1 + w2
        assert np.array_equal(wsum, wsum_test)

        wsum = DataOpArray.from_list([k4 + k1, k4 + k2, k4 + k3], shape=(1, 3))
        wsum_test = w2 + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__w1by3_mixed(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        k21 = ref_datasets["h1"]
        k22 = ref_datasets["h2"]
        k23 = ref_datasets["hbig"]

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        w2 = DataOpArray.from_list([k21, k22, k23], shape=(1, 3))

        wsum = DataOpArray.from_list([k11 + k21, k12 + k22, k13 + k23], shape=(1, 3))
        wsum_test = w1 + w2
        assert np.array_equal(wsum, wsum_test)

        wsum = DataOpArray.from_list([k21 + k11, k22 + k12, k23 + k13], shape=(1, 3))
        wsum_test = w2 + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__w1by2_mixed_FAIL(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        k21 = ref_datasets["h1"]
        k22 = ref_datasets["h2"]

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        w2 = DataOpArray.from_list([k21, k22], shape=(1, 2))

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w1, w2)

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w2, w1)

    def test_w1by3_mixed__flt_scal(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        flt = 4.0

        wsum = DataOpArray.from_list([k11 + flt, k12 + flt, k13 + flt], shape=(1, 3))

        wsum_test = w1 + flt
        assert np.array_equal(wsum, wsum_test)

        wsum_test = flt + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__flt_1by3(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        flt = np.array([[4.0, 5.0, 6.0]])

        wsum = DataOpArray.from_list([k11 + 4.0, k12 + 5.0, k13 + 6.0], shape=(1, 3))

        wsum_test = w1 + flt
        assert np.array_equal(wsum, wsum_test)

        wsum_test = flt + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w3by1_mixed__flt_3by1(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(3, 1))
        flt = np.array([[4.0], [5.0], [6.0]])

        wsum = DataOpArray.from_list([k11 + 4.0, k12 + 5.0, k13 + 6.0], shape=(3, 1))

        wsum_test = w1 + flt
        assert np.array_equal(wsum, wsum_test)

        wsum_test = flt + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__flt_3by1_FAIL(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        flt = np.array([[4.0], [5.0], [6.0]])

        with pytest.raises(ValueError, match="HERBERT:size_array_split:invalid_argument"):
            plus(w1, flt)

        with pytest.raises(ValueError, match="HERBERT:size_array_split:invalid_argument"):
            plus(flt, w1)

    def test_w1by3_20by1__flt_20by3(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["p3"]

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        rng = np.random.default_rng(104)
        flt = rng.random((k11.signal.size, 3))

        wsum = DataOpArray.from_list(
            [k11 + flt[:, 0:1], k12 + flt[:, 1:2], k13 + flt[:, 2:3]], shape=(1, 3)
        )

        wsum_test = w1 + flt
        assert np.array_equal(wsum, wsum_test)

        wsum_test = flt + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w20by3_mixed__flt_20by3_FAIL(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        rng = np.random.default_rng(105)
        flt = rng.random((k11.signal.size, 3))

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w1, flt)

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(flt, w1)


class TestIXDataset1dOpsPart2:
    """Tests adding IX_dataset_1d objects and sigvar objects (Part 2)."""

    def test_wscal_20by1__wscal_20by1(self, ref_datasets):
        w1 = ref_datasets["p1"].copy()
        rng = np.random.default_rng(201)
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

    def test_wscal_20by1__wscal_21by1_FAIL(self, ref_datasets):
        w1 = ref_datasets["p1"]
        rng = np.random.default_rng(202)
        sh = (w1.signal.shape[0] + 1, 1)
        w2 = sigvar(rng.random(sh), rng.random(sh))

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w1, w2)

        with pytest.raises(ValueError, match="HERBERT:sigvar:invalid_argument"):
            plus(w2, w1)

    def test_w1by3_1by2__wscal_1by2(self, ref_datasets):
        k1 = ref_datasets["p1"]
        k2 = ref_datasets["p2"]
        k3 = ref_datasets["p3"]
        rng = np.random.default_rng(203)
        k4 = sigvar(rng.random(k1.signal.shape), rng.random(k1.error.shape))

        w1 = DataOpArray.from_list([k1, k2, k3], shape=(1, 3))
        w2 = k4

        wsum = DataOpArray.from_list([k1 + k4, k2 + k4, k3 + k4], shape=(1, 3))
        wsum_test = w1 + w2
        assert np.array_equal(wsum, wsum_test)

        wsum = DataOpArray.from_list([k4 + k1, k4 + k2, k4 + k3], shape=(1, 3))
        wsum_test = w2 + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__w1by3_mixed(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        rng = np.random.default_rng(204)
        k21 = sigvar(rng.random(k11.signal.shape), rng.random(k11.error.shape))
        k22 = sigvar(rng.random(k12.signal.shape), rng.random(k12.error.shape))
        k23 = sigvar(rng.random(k13.signal.shape), rng.random(k13.error.shape))

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        w2 = DataOpArray.from_list([k21, k22, k23], shape=(1, 3))

        wsum = DataOpArray.from_list([k11 + k21, k12 + k22, k13 + k23], shape=(1, 3))
        wsum_test = w1 + w2
        assert np.array_equal(wsum, wsum_test)

        wsum = DataOpArray.from_list([k21 + k11, k22 + k12, k23 + k13], shape=(1, 3))
        wsum_test = w2 + w1
        assert np.array_equal(wsum, wsum_test)

    def test_w1by3_mixed__w1by2_mixed_FAIL(self, ref_datasets):
        k11 = ref_datasets["p1"]
        k12 = ref_datasets["p2"]
        k13 = ref_datasets["pbig"]

        rng = np.random.default_rng(205)
        k21 = sigvar(rng.random(k11.signal.shape), rng.random(k11.error.shape))
        k22 = sigvar(rng.random(k12.signal.shape), rng.random(k12.error.shape))

        w1 = DataOpArray.from_list([k11, k12, k13], shape=(1, 3))
        w2 = DataOpArray.from_list([k21, k22], shape=(1, 2))

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w1, w2)

        with pytest.raises(ValueError, match="HERBERT:data_op_interface:invalid_argument"):
            plus(w2, w1)
