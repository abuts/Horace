"""Cylindrical 3He detector tube class (IX_det_He3tube)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_abstract_type import (
    IX_det_abstractType,
    format_scalar_result,
    is_integer_id,
    matlab_size,
    rand_truncexp2,
    size_array_stack,
)

_C_EFF_F = np.array(
    [
        0.7648360390553052,
        -0.3700950778935237,
        0.1582704090813516,
        -6.0170218669705407e-02,
        2.0465515957968953e-02,
        -6.2690181465706840e-03,
        1.7408667184745830e-03,
        -4.4101378999425122e-04,
        1.0252117967127217e-04,
        -2.1988904738111659e-05,
        4.3729347905629990e-06,
        -8.0998753944849788e-07,
        1.4031240949230472e-07,
        -2.2815971698619819e-08,
        3.4943984983382137e-09,
        -5.0562696807254781e-10,
        6.9315483353094009e-11,
        -9.0261598195695569e-12,
        1.1192324844699897e-12,
        -1.3204992654891612e-13,
        1.4100387524251801e-14,
        -8.6430862467068437e-16,
        -1.1129985821867194e-16,
        -4.5505266221823604e-16,
        3.8885561437496108e-16,
    ],
    dtype=float,
)

_C_EFF_G = np.array(
    [
        2.033429926215546,
        -2.3123407369310212e-02,
        7.0671915734894875e-03,
        -7.5970017538257162e-04,
        7.4848652541832373e-05,
        4.5642679186460588e-05,
        -2.3097291253000307e-05,
        1.9697221715275770e-06,
        2.4115259271262346e-06,
        -7.1302220919333692e-07,
        -2.5124427621592282e-07,
        1.3246884875139919e-07,
        3.4364196805913849e-08,
        -2.2891359549026546e-08,
        -6.7281240212491156e-09,
        3.8292458615085678e-09,
        1.6451021034313840e-09,
        -5.5868962123284405e-10,
        -4.2052310689211225e-10,
        4.3217612266666094e-11,
        9.9547699528024225e-11,
        1.2882834243832519e-11,
        -1.9103066351000564e-11,
        -7.6805495297094239e-12,
        1.8568853399347773e-12,
    ],
    dtype=float,
)

_C_DEL_F = np.array(
    [
        1.457564928500728,
        -0.2741263150129247,
        1.4102406058428482e-02,
        1.1868136977190956e-02,
        -4.7000120888695418e-03,
        6.7071002620380348e-04,
        1.2315212155928235e-04,
        -8.7985748380390304e-05,
        1.8952644758594150e-05,
        4.4101711646149510e-07,
        -1.5292393205490473e-06,
        4.5050196748941396e-07,
        -2.9971703975339992e-08,
        -2.3573145628841274e-08,
        9.6228336343706644e-09,
        -1.3038786850216866e-09,
        -2.9423462000188749e-10,
        1.8813720970012326e-10,
        -3.7682054143672871e-11,
        -1.9125961925325896e-12,
        3.3516145414580478e-12,
        -9.0842416922143343e-13,
        4.3951786654616853e-14,
        4.5793924208226145e-14,
        -1.4916540225229369e-14,
    ],
    dtype=float,
)

_C_DEL_G = np.array(
    [
        1.980495234559052,
        1.3148750635418816e-02,
        -3.5137830163154959e-03,
        1.4111112411286597e-04,
        -2.4707009281715875e-05,
        -4.9602024972950076e-08,
        1.5268651833078018e-06,
        -4.8070752083129165e-07,
        -3.5826648758785495e-08,
        6.0264253483044428e-08,
        -4.2948016776289677e-09,
        -7.5840171520624722e-09,
        1.0468151234732659e-09,
        1.1267346944343615e-09,
        -1.4810551229871294e-10,
        -1.9605287726598419e-10,
        9.8596597553068932e-12,
        3.6752354493074790e-11,
        3.2634850377633029e-12,
        -6.6207839211074316e-12,
        -1.9158341579839089e-12,
        9.6091495871419851e-13,
        6.3198529742791721e-13,
        -6.4681177081027385e-14,
        -1.8198241524824965e-13,
    ],
    dtype=float,
)

_C_VAR_F = np.array(
    [
        1.226904583058190,
        -0.3621914072547197,
        6.0117947617747081e-02,
        1.8037337764424607e-02,
        -1.4439005957980123e-02,
        3.8147446724517908e-03,
        1.3679160269450818e-05,
        -3.7851338401354573e-04,
        1.3568342238781006e-04,
        -1.3336183765173537e-05,
        -7.5468390663036011e-06,
        3.7919580869305580e-06,
        -6.4560788919254541e-07,
        -1.0509789897250599e-07,
        9.0282233408123247e-08,
        -2.1598200223849062e-08,
        -2.6200750125049410e-10,
        1.8693270043002030e-09,
        -6.0097600840247623e-10,
        4.7263196689684150e-11,
        3.3052446335446462e-11,
        -1.4738090470256537e-11,
        2.1945176231774610e-12,
        4.7409048908875206e-13,
        -3.3502478569147342e-13,
    ],
    dtype=float,
)

_C_VAR_G = np.array(
    [
        1.862646413811875,
        7.5988886169808666e-02,
        -8.3110620384910993e-03,
        1.1236935254690805e-03,
        -1.0549380723194779e-04,
        -3.8256672783453238e-05,
        2.2883355513325654e-05,
        -2.4595515448511130e-06,
        -2.2063956882489855e-06,
        7.2331970290773207e-07,
        2.2080170614557915e-07,
        -1.2957057474505262e-07,
        -2.9737380539129887e-08,
        2.2171316129693253e-08,
        5.9127004825576534e-09,
        -3.7179338302495424e-09,
        -1.4794271269158443e-09,
        5.5412448241032308e-10,
        3.8726354734119894e-10,
        -4.6562413924533530e-11,
        -9.2734525614091013e-11,
        -1.1246343578630302e-11,
        1.6909724176450425e-11,
        5.6146245985821963e-12,
        -2.7408274955176282e-12,
    ],
    dtype=float,
)

_C_VARW_F = np.array(
    [
        2.408884004758557,
        0.1441097208627301,
        -5.0093583831079742e-02,
        1.0574012517851051e-02,
        -4.7245491418700381e-04,
        -5.6874753986616233e-04,
        2.2050994176359695e-04,
        -3.0071128379836054e-05,
        -6.5175276460682774e-06,
        4.2908624511150961e-06,
        -8.8327783029362728e-07,
        -3.5778896608773536e-08,
        7.6164115048182878e-08,
        -2.1399959173606931e-08,
        1.1599700144859781e-09,
        1.2029935880786269e-09,
        -4.6385151497574384e-10,
        5.7945164222417134e-11,
        1.5725836188806852e-11,
        -9.1953450409576476e-12,
        1.7449824918358559e-12,
        1.2301937246661510e-13,
        -1.6739387653785798e-13,
        4.5505543777579760e-14,
        -4.3223757906218907e-15,
    ],
    dtype=float,
)

_C_VARW_G = np.array(
    [
        1.970558139796674,
        1.9874189524780751e-02,
        -5.3520719319403742e-03,
        2.3885486654173116e-04,
        -4.1428357951582839e-05,
        -6.3229035418110869e-07,
        2.8594609307941443e-06,
        -8.5378305322625359e-07,
        -8.2383358224191738e-08,
        1.1218202137786015e-07,
        -6.0736651874560010e-09,
        -1.4453200922748266e-08,
        1.7154640064021009e-09,
        2.1673530992138979e-09,
        -2.4074988114186624e-10,
        -3.7678839381882767e-10,
        1.1723938486696284e-11,
        7.0125182882740944e-11,
        7.5127332133106960e-12,
        -1.2478237332302910e-11,
        -3.8880659802842388e-12,
        1.7635456983633446e-12,
        1.2439449470491581e-12,
        -9.4195068411906391e-14,
        -3.4105815394092076e-13,
    ],
    dtype=float,
)


def _effchb(a: float, b: float, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Clenshaw Chebyshev evaluation on [a, b] (equivalent to CHEBEV / effchb)."""
    d = np.zeros_like(x, dtype=float)
    dd = np.zeros_like(x, dtype=float)
    y = (2.0 * x - a - b) / (b - a)
    y2 = 2.0 * y
    for j in range(c.size - 1, 0, -1):
        sv = d
        d = (y2 * d - dd) + c[j]
        dd = sv
    return (y * d - dd) + 0.5 * c[0]


def _effic_alf(alf: np.ndarray) -> np.ndarray:
    """Efficiency of a 3He tube given dimensionless path length alf."""
    val = np.zeros_like(alf, dtype=float)
    ilo = alf <= 9.0
    ihi = alf >= 10.0
    imix = (alf > 9.0) & (alf < 10.0)

    if np.any(ilo):
        a_lo = alf[ilo]
        val[ilo] = (np.pi / 4.0) * (a_lo * _effchb(0.0, 10.0, _C_EFF_F, a_lo))
    if np.any(ihi):
        a_hi = alf[ihi]
        y = 1.0 - 18.0 / a_hi
        val[ihi] = 1.0 - _effchb(-1.0, 1.0, _C_EFF_G, y) / (a_hi**2)
    if np.any(imix):
        a_mix = alf[imix]
        eff_f = (np.pi / 4.0) * (a_mix * _effchb(0.0, 10.0, _C_EFF_F, a_mix))
        y = 1.0 - 18.0 / a_mix
        eff_g = 1.0 - _effchb(-1.0, 1.0, _C_EFF_G, y) / (a_mix**2)
        val[imix] = (10.0 - a_mix) * eff_f + (a_mix - 9.0) * eff_g
    return val


def _mean_d_alf(alf: np.ndarray) -> np.ndarray:
    """Mean depth of absorption in a 3He tube in reduced units."""
    val = np.zeros_like(alf, dtype=float)
    ilo = alf <= 9.0
    ihi = alf >= 10.0
    imix = (alf > 9.0) & (alf < 10.0)

    if np.any(ilo):
        a_lo = alf[ilo]
        val[ilo] = -0.125 * (a_lo * _effchb(0.0, 10.0, _C_DEL_F, a_lo))
    if np.any(ihi):
        a_hi = alf[ihi]
        y = 1.0 - 18.0 / a_hi
        eff = 1.0 - _effchb(-1.0, 1.0, _C_EFF_G, y) / (a_hi**2)
        val[ihi] = (2.0 * _effchb(-1.0, 1.0, _C_DEL_G, y) / a_hi - np.pi / 4.0) / eff
    if np.any(imix):
        a_mix = alf[imix]
        del_f = -0.125 * (a_mix * _effchb(0.0, 10.0, _C_DEL_F, a_mix))
        y = 1.0 - 18.0 / a_mix
        eff = 1.0 - _effchb(-1.0, 1.0, _C_EFF_G, y) / (a_mix**2)
        del_g = (2.0 * _effchb(-1.0, 1.0, _C_DEL_G, y) / a_mix - np.pi / 4.0) / eff
        val[imix] = (10.0 - a_mix) * del_f + (a_mix - 9.0) * del_g
    return val


def _var_d_alf(alf: np.ndarray) -> np.ndarray:
    """Variance of depth of absorption in a 3He tube in reduced units."""
    g0 = (32.0 - 3.0 * (np.pi**2)) / 48.0
    g1 = 14.0 / 3.0 - (np.pi**2) / 8.0

    val = np.zeros_like(alf, dtype=float)
    ilo = alf <= 9.0
    ihi = alf >= 10.0
    imix = (alf > 9.0) & (alf < 10.0)

    if np.any(ilo):
        a_lo = alf[ilo]
        val[ilo] = 0.25 * _effchb(0.0, 10.0, _C_VAR_F, a_lo)
    if np.any(ihi):
        a_hi = alf[ihi]
        y = 1.0 - 18.0 / a_hi
        val[ihi] = g0 + g1 * _effchb(-1.0, 1.0, _C_VAR_G, y) / (a_hi**2)
    if np.any(imix):
        a_mix = alf[imix]
        var_f = 0.25 * _effchb(0.0, 10.0, _C_VAR_F, a_mix)
        y = 1.0 - 18.0 / a_mix
        var_g = g0 + g1 * _effchb(-1.0, 1.0, _C_VAR_G, y) / (a_mix**2)
        val[imix] = (10.0 - a_mix) * var_f + (a_mix - 9.0) * var_g
    return val


def _var_y_alf(alf: np.ndarray) -> np.ndarray:
    """Variance of width of absorption in a 3He tube in reduced units."""
    val = np.zeros_like(alf, dtype=float)
    ilo = alf <= 9.0
    ihi = alf >= 10.0
    imix = (alf > 9.0) & (alf < 10.0)

    if np.any(ilo):
        a_lo = alf[ilo]
        val[ilo] = 0.25 * _effchb(0.0, 10.0, _C_VARW_F, a_lo)
    if np.any(ihi):
        a_hi = alf[ihi]
        y = 1.0 - 18.0 / a_hi
        eff = 1.0 - _effchb(-1.0, 1.0, _C_EFF_G, y) / (a_hi**2)
        val[ihi] = (-_effchb(-1.0, 1.0, _C_VARW_G, y) / (a_hi**2) + 1.0 / 3.0) / eff
    if np.any(imix):
        a_mix = alf[imix]
        varw_f = 0.25 * _effchb(0.0, 10.0, _C_VARW_F, a_mix)
        y = 1.0 - 18.0 / a_mix
        eff = 1.0 - _effchb(-1.0, 1.0, _C_EFF_G, y) / (a_mix**2)
        varw_g = (-_effchb(-1.0, 1.0, _C_VARW_G, y) / (a_mix**2) + 1.0 / 3.0) / eff
        val[imix] = (10.0 - a_mix) * varw_f + (a_mix - 9.0) * varw_g
    return val


def _rand_xy2(alf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random points in the absorbing ellipsoid."""
    x = rand_truncexp2(alf) / alf
    y = 2.0 * np.random.rand(*alf.shape) - 1.0
    reject = (x**2) >= (1.0 - y**2)
    while np.any(reject):
        a_rej = alf[reject]
        x_sub = rand_truncexp2(a_rej) / a_rej
        y_sub = 2.0 * np.random.rand(*a_rej.shape) - 1.0
        x[reject] = x_sub
        y[reject] = y_sub
        reject_sub = (x_sub**2) >= (1.0 - y_sub**2)
        new_reject = np.zeros_like(reject)
        new_reject[reject] = reject_sub
        reject = new_reject
    return x, y


class IX_det_He3tube(IX_det_abstractType):
    """Defines array of cylindrical 3He detectors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._dia: np.ndarray = np.zeros((1, 1), dtype=float)
        self._height: np.ndarray = np.zeros((1, 1), dtype=float)
        self._wall: np.ndarray = np.zeros((1, 1), dtype=float)
        self._atms: np.ndarray = np.zeros((1, 1), dtype=float)

        if len(args) > 0 or len(kwargs) > 0:
            property_names = ["dia", "height", "wall", "atms"]
            options = {"key_dash": True, "mandatory_props": [True, True, True, True]}
            _, remains = self.set_positional_and_key_val_arguments(
                property_names, options, *args, **kwargs
            )
            if remains:
                raise ValueError(
                    f"HERBERT:IX_det_He3tube:invalid_argument: "
                    f"Unrecognised extra parameters provided as input to "
                    f"IX_det_He3tube constructor: {remains}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dia(self) -> np.ndarray:
        return self._dia.copy()

    @dia.setter
    def dia(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_He3tube:invalid_argument: "
                "Tube diameter(s) must be greater or equal to zero"
            )
        self._dia = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def width(self) -> np.ndarray:
        return self._dia.copy()

    @property
    def height(self) -> np.ndarray:
        return self._height.copy()

    @height.setter
    def height(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_He3tube:invalid_argument: "
                "Detector element height(s) must be greater or equal to zero"
            )
        self._height = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def wall(self) -> np.ndarray:
        return self._wall.copy()

    @wall.setter
    def wall(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_He3tube:invalid_argument: "
                "Wall thickness(es) must be greater or equal to zero"
            )
        self._wall = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def atms(self) -> np.ndarray:
        return self._atms.copy()

    @atms.setter
    def atms(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_He3tube:invalid_argument: "
                "Partial pressure(s) of 3He must be greater or equal to zero"
            )
        self._atms = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def inner_rad(self) -> np.ndarray:
        return 0.5 * (self._dia - 2.0 * self._wall)

    def get_ndet(self) -> int:
        return int(self._dia.size)

    # ------------------------------------------------------------------
    # Reorder and replicate
    # ------------------------------------------------------------------
    def reorder(self, ix: Any) -> IX_det_He3tube:
        ndet = self.ndet
        ix_arr = np.asarray(ix, dtype=int).ravel(order="F")
        ok, _, _, perm, _ = is_integer_id(ix_arr)
        if not (ok and perm and ix_arr.size == ndet):
            raise ValueError(
                "HERBERT:IX_det_He3tube:invalid_argument: "
                "Index array is not a permutation of the integers 1 to the "
                "number of detector elements"
            )
        ix_0 = ix_arr - 1
        obj_out = copy.deepcopy(self)
        obj_out._dia = self._dia[ix_0].reshape(-1, 1)
        obj_out._height = self._height[ix_0].reshape(-1, 1)
        obj_out._wall = self._wall[ix_0].reshape(-1, 1)
        obj_out._atms = self._atms[ix_0].reshape(-1, 1)
        obj_out.clear_hash()
        return obj_out

    def replicate(self, n: int) -> IX_det_He3tube:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
            raise ValueError(
                "HERBERT:IX_det_He3tube:invalid_argument: "
                "Replication factor must be a positive integer"
            )
        obj_out = copy.deepcopy(self)
        if n > 1:
            obj_out._dia = np.tile(self._dia, (int(n), 1))
            obj_out._height = np.tile(self._height, (int(n), 1))
            obj_out._wall = np.tile(self._wall, (int(n), 1))
            obj_out._atms = np.tile(self._atms, (int(n), 1))
            obj_out.clear_hash()
        return obj_out

    # ------------------------------------------------------------------
    # Internal helper: macro_xs_dia
    # ------------------------------------------------------------------
    def _macro_xs_dia(
        self, ind: np.ndarray, npath: np.ndarray, wvec: np.ndarray
    ) -> np.ndarray:
        const = 1.000937073600000e02
        ind_0 = ind.ravel(order="F") - 1
        irad = self.inner_rad[ind_0, 0]
        dist_press = (irad / npath[0, :]) * self._atms[ind_0, 0]

        if ind.size == 1:
            sz_w = matlab_size(wvec)
            alf = (const * dist_press[0]) / wvec.reshape(sz_w, order="F")
        elif wvec.size == 1:
            sz_i = matlab_size(ind)
            alf = (const / float(wvec.item())) * dist_press.reshape(sz_i, order="F")
        else:
            sz_w = matlab_size(wvec)
            alf = (const * dist_press.reshape(sz_w, order="F")) / wvec.reshape(
                sz_w, order="F"
            )
        return alf

    # ------------------------------------------------------------------
    # Detector evaluation methods
    # ------------------------------------------------------------------
    def effic(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf = self._macro_xs_dia(ind, npath, wvec)
        val = _effic_alf(alf)
        return format_scalar_result(val, sz)

    def mean_d(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf = self._macro_xs_dia(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        scale = self.inner_rad[ind_0, 0] / npath[0, :]
        if ind.size != 1:
            scale = scale.reshape(alf.shape, order="F")
        else:
            scale = float(scale[0])
        val = scale * _mean_d_alf(alf)
        return format_scalar_result(val, sz)

    def var_d(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf = self._macro_xs_dia(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        scale = (self.inner_rad[ind_0, 0] / npath[0, :]) ** 2
        if ind.size != 1:
            scale = scale.reshape(alf.shape, order="F")
        else:
            scale = float(scale[0])
        val = scale * _var_d_alf(alf)
        return format_scalar_result(val, sz)

    def mean_x(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf = self._macro_xs_dia(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        scale = self.inner_rad[ind_0, 0]
        if ind.size != 1:
            scale = scale.reshape(alf.shape, order="F")
        else:
            scale = float(scale[0])
        val = scale * _mean_d_alf(alf)
        return format_scalar_result(val, sz)

    def mean_y(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.zeros(sz, dtype=float), sz)

    def mean_z(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.zeros(sz, dtype=float), sz)

    def mean(self, *args: Any) -> np.ndarray:
        mx = np.asarray(self.mean_x(*args), dtype=float)
        my = np.asarray(self.mean_y(*args), dtype=float)
        mz = np.asarray(self.mean_z(*args), dtype=float)
        sz_mx = matlab_size(mx)
        xyz = np.vstack(
            [
                mx.ravel(order="F").reshape(1, -1),
                my.ravel(order="F").reshape(1, -1),
                mz.ravel(order="F").reshape(1, -1),
            ]
        )
        sz_full = size_array_stack((3, 1), sz_mx)
        return xyz.reshape(sz_full, order="F")

    def var_x(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf = self._macro_xs_dia(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        scale = self.inner_rad[ind_0, 0] ** 2
        if ind.size != 1:
            scale = scale.reshape(alf.shape, order="F")
        else:
            scale = float(scale[0])
        val = scale * _var_d_alf(alf)
        return format_scalar_result(val, sz)

    def var_y(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf = self._macro_xs_dia(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        scale = self.inner_rad[ind_0, 0] ** 2
        if ind.size != 1:
            scale = scale.reshape(alf.shape, order="F")
        else:
            scale = float(scale[0])
        val = scale * _var_y_alf(alf)
        return format_scalar_result(val, sz)

    def var_z(self, *args: Any) -> float | np.ndarray:
        sz, ind, _, _ = self.parse_ind_npath_wvec_(*args)
        ind_0 = ind.ravel(order="F") - 1
        height = self._height[ind_0, 0]
        if ind.size != 1:
            val = (height.reshape(sz, order="F") ** 2) / 12.0
        else:
            val = ((height.item() ** 2) / 12.0) * np.ones(sz, dtype=float)
        return format_scalar_result(val, sz)

    def covariance(self, *args: Any) -> np.ndarray:
        cxx = np.asarray(self.var_x(*args), dtype=float)
        cyy = np.asarray(self.var_y(*args), dtype=float)
        czz = np.asarray(self.var_z(*args), dtype=float)
        sz_cxx = matlab_size(cxx)

        n = cxx.size
        val = np.zeros((3, 3, n), dtype=float, order="F")
        val[0, 0, :] = cxx.ravel(order="F")
        val[1, 1, :] = cyy.ravel(order="F")
        val[2, 2, :] = czz.ravel(order="F")

        sz_full = size_array_stack((3, 3), sz_cxx)
        return val.reshape(sz_full, order="F")

    def rand(self, *args: Any) -> np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf = self._macro_xs_dia(ind, npath, wvec)
        alf_col = alf.ravel(order="F").reshape(-1, 1)
        x, y = _rand_xy2(alf_col)

        ind_0 = ind.ravel(order="F") - 1
        irad = self.inner_rad[ind_0, 0].reshape(-1, 1)
        h = self._height[ind_0, 0].reshape(-1, 1)
        n = alf_col.size

        xyz = np.vstack(
            [
                (irad * (2.0 * x - np.sqrt(1.0 - y**2))).T,
                (irad * y).T,
                (h * (np.random.rand(n, 1) - 0.5)).T,
            ]
        )
        sz_full = size_array_stack((3, 1), sz)
        return xyz.reshape(sz_full, order="F")

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 2

    def saveable_fields(self) -> list[str]:
        return ["dia", "height", "wall", "atms"]

    def check_combo_arg(self) -> IX_det_He3tube:
        flds = self.saveable_fields()
        self.expand_internal_properties_to_max_length(flds)
        if not np.all(self._dia >= 2.0 * self._wall):
            raise ValueError(
                "HERBERT:IX_det_He3tube:invalid_argument: "
                "Tube diameter(s) must be greater or equal to twice the wall thickness(es)"
            )
        self.clear_hash()
        return self

    def convert_old_struct(
        self, s: dict[str, Any], ver: float
    ) -> tuple[dict[str, Any], IX_det_He3tube]:
        inputs = dict(s)
        if "class_version_" in inputs and inputs["class_version_"] == 1:
            inputs.pop("class_version_", None)
            new_inputs = {}
            for key, val in inputs.items():
                new_key = key[:-1] if key.endswith("_") else key
                new_inputs[new_key] = val
            return new_inputs, self
        new_inputs = {}
        for key, val in inputs.items():
            if key in ("dia_", "height_", "wall_", "atms_"):
                new_inputs[key[:-1]] = val
            else:
                new_inputs[key] = val
        return new_inputs, self

