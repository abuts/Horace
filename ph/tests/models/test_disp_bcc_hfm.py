"""Unit tests for ``disp_bcc_hfm`` dispersion relation.

Translated from MATLAB ``_test/test_sqw_models/test_disp_bcc_hfm.m``.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import pytest

from pyhorace.models.disp_bcc_hfm import disp_bcc_hfm


def QsymRelated(
    qh: float, qk: float, ql: float, rlp: Optional[np.ndarray] = None
) -> np.ndarray:
    """Get all symmetry related Q points of an input point for a cubic lattice.

    The excitation energies from a dispersion relation at each of the output points
    should all be identical.

    Parameters
    ----------
    qh, qk, ql : float
        Point in reciprocal lattice (in rlu) (each is scalar).
    rlp : np.ndarray, optional
        Set of reciprocal lattice points; shape ``(nrlp, 3)``.
        Each element should be an integer and each row corresponds to a
        zone centre in the reciprocal lattice.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 3)`` of symmetry related Q points offset by all the
        zone centres in input argument ``rlp``, if given.
    """
    rows = []
    for sx, sy, sz in itertools.product([1.0, -1.0], repeat=3):
        pt = [sx * qh, sy * qk, sz * ql]
        for p in itertools.permutations(pt):
            rows.append(p)
    Q = np.unique(np.array(rows, dtype=np.float64), axis=0)
    nQ = Q.shape[0]

    if rlp is not None:
        rlp_arr = np.atleast_2d(np.asarray(rlp, dtype=np.float64))
        nrlu = rlp_arr.shape[0]
        if nrlu == 1:
            Q = Q + rlp_arr
        else:
            Q = np.tile(Q, (nrlu, 1)) + np.column_stack(
                [
                    np.repeat(rlp_arr[:, 0], nQ),
                    np.repeat(rlp_arr[:, 1], nQ),
                    np.repeat(rlp_arr[:, 2], nQ),
                ]
            )
    return Q


class TestDispBccHfm:
    """Test bcc Heisenberg ferromagnet dispersion relation ``disp_bcc_hfm``."""

    @pytest.fixture(autouse=True)
    def setup_data(self) -> None:
        qh_ref = 0.216
        qk_ref = 0.314
        ql_ref = 0.271

        rlp = np.array([[0, 0, 0], [1, 1, 0], [-2, 2, 4]], dtype=np.float64)

        Q_h00 = QsymRelated(qh_ref, 0.0, 0.0, rlp)
        Q_hh0 = QsymRelated(qh_ref, qh_ref, 0.0, rlp)
        Q_hhh = QsymRelated(qh_ref, qh_ref, qh_ref, rlp)
        Q_hk0 = QsymRelated(qh_ref, qk_ref, 0.0, rlp)
        Q_hhk = QsymRelated(qh_ref, qh_ref, qk_ref, rlp)
        Q_hkl = QsymRelated(qh_ref, qk_ref, ql_ref, rlp)

        self.qh_ref = qh_ref
        self.qk_ref = qk_ref
        self.ql_ref = ql_ref
        self.Qh = [
            Q_h00[:, 0],
            Q_hh0[:, 0],
            Q_hhh[:, 0],
            Q_hk0[:, 0],
            Q_hhk[:, 0],
            Q_hkl[:, 0],
        ]
        self.Qk = [
            Q_h00[:, 1],
            Q_hh0[:, 1],
            Q_hhh[:, 1],
            Q_hk0[:, 1],
            Q_hhk[:, 1],
            Q_hkl[:, 1],
        ]
        self.Ql = [
            Q_h00[:, 2],
            Q_hh0[:, 2],
            Q_hhh[:, 2],
            Q_hk0[:, 2],
            Q_hhk[:, 2],
            Q_hkl[:, 2],
        ]

        self.Seff = 10.0
        self.gap = 3.94
        self.JS = np.array(
            [
                0.762368339977811,
                0.080068138934902,
                0.938246224630109,
                0.974473843479017,
                0.607110428021082,
                0.957494958695934,
                0.330650541159530,
                0.901215142217029,
                0.000365623763306,
                0.483280019453646,
                0.147474747267827,
                0.314992086852311,
                0.171375458208079,
                0.592906936782863,
                0.657608011954500,
            ],
            dtype=np.float64,
        )

        self.w_JS = np.array(
            [
                4.373082526631028,
                0.530200153173490,
                11.481791003296113,
                31.377318004399211,
                4.803863639321013,
                10.664848701393527,
                4.936824265569184,
                19.490447148602474,
                0.008621153138463,
                13.257772891438805,
                1.612818613826166,
                1.033756435200059,
                6.426932593773794,
                2.701274574774424,
                17.281668078687623,
            ],
            dtype=np.float64,
        )
        self.w_tot = 1.339212197832254e02

        self.par0 = np.concatenate([[self.Seff], np.zeros(16, dtype=np.float64)])
        self.atol = 1e-12
        self.rtol = 1e-12

    def _check_single_js(self, js_idx_1based: int) -> None:
        par = self.par0.copy()
        par[js_idx_1based + 1] = self.JS[js_idx_1based - 1]
        for i in range(len(self.Qh)):
            w, wt = disp_bcc_hfm(self.Qh[i], self.Qk[i], self.Ql[i], par)
            assert w[0][0] > 0, "energy must be greater than zero"
            np.testing.assert_allclose(
                w[0], np.full_like(w[0], w[0][0]), atol=self.atol, rtol=self.rtol
            )
            np.testing.assert_allclose(
                wt[0], np.full_like(wt[0], self.Seff / 2.0), atol=self.atol, rtol=self.rtol
            )

    def test_gap(self) -> None:
        par = [self.Seff, self.gap]
        for i in range(len(self.Qh)):
            w, wt = disp_bcc_hfm(self.Qh[i], self.Qk[i], self.Ql[i], par)
            assert w[0][0] > 0, "energy must be greater than zero"
            np.testing.assert_allclose(
                w[0], np.full_like(w[0], w[0][0]), atol=self.atol, rtol=self.rtol
            )
            np.testing.assert_allclose(w[0][0], self.gap, atol=self.atol, rtol=self.rtol)
            np.testing.assert_allclose(
                wt[0], np.full_like(wt[0], self.Seff / 2.0), atol=self.atol, rtol=self.rtol
            )

    def test_JS1(self) -> None:
        self._check_single_js(1)

    def test_JS2(self) -> None:
        self._check_single_js(2)

    def test_JS3(self) -> None:
        self._check_single_js(3)

    def test_JS4(self) -> None:
        self._check_single_js(4)

    def test_JS5(self) -> None:
        self._check_single_js(5)

    def test_JS6(self) -> None:
        self._check_single_js(6)

    def test_JS7(self) -> None:
        self._check_single_js(7)

    def test_JS8(self) -> None:
        self._check_single_js(8)

    def test_JS9(self) -> None:
        self._check_single_js(9)

    def test_JS10(self) -> None:
        self._check_single_js(10)

    def test_JS11(self) -> None:
        self._check_single_js(11)

    def test_JS12(self) -> None:
        self._check_single_js(12)

    def test_JS13(self) -> None:
        self._check_single_js(13)

    def test_JS14(self) -> None:
        self._check_single_js(14)

    def test_JS15(self) -> None:
        self._check_single_js(15)

    def test_JS1_to_JS15(self) -> None:
        par = self.par0.copy()
        par[2:17] = self.JS
        for i in range(len(self.Qh)):
            w, wt = disp_bcc_hfm(self.Qh[i], self.Qk[i], self.Ql[i], par)
            assert w[0][0] > 0, "energy must be greater than zero"
            np.testing.assert_allclose(
                w[0], np.full_like(w[0], w[0][0]), atol=self.atol, rtol=self.rtol
            )
            np.testing.assert_allclose(
                wt[0], np.full_like(wt[0], self.Seff / 2.0), atol=self.atol, rtol=self.rtol
            )

    def test_JS1_to_JS6(self) -> None:
        par = self.par0.copy()
        par[2:8] = self.JS[:6]

        wfull, wtfull = disp_bcc_hfm(self.qh_ref, self.qk_ref, self.ql_ref, par)
        w, wt = disp_bcc_hfm(self.qh_ref, self.qk_ref, self.ql_ref, par[:8])
        np.testing.assert_allclose(wfull[0], w[0], atol=self.atol, rtol=self.rtol)
        np.testing.assert_allclose(wtfull[0], wt[0], atol=self.atol, rtol=self.rtol)

    def test_absolute_energy_JS_one_at_a_time(self) -> None:
        for i in range(15):
            par = self.par0.copy()
            par[i + 2] = self.JS[i]
            w, wt = disp_bcc_hfm(self.qh_ref, self.qk_ref, self.ql_ref, par)
            np.testing.assert_allclose(w[0], self.w_JS[i], atol=self.atol, rtol=self.rtol)
            np.testing.assert_allclose(wt[0], self.Seff / 2.0, atol=self.atol, rtol=self.rtol)

    def test_absolute_energy_value_gap_JS1_to_JS15(self) -> None:
        par = np.concatenate([[self.Seff, self.gap], self.JS])
        w, wt = disp_bcc_hfm(self.qh_ref, self.qk_ref, self.ql_ref, par)
        np.testing.assert_allclose(w[0], self.w_tot, atol=self.atol, rtol=self.rtol)
        np.testing.assert_allclose(wt[0], self.Seff / 2.0, atol=self.atol, rtol=self.rtol)

