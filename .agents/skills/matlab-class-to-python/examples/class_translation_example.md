# End-to-End Class Translation Example

This example demonstrates the complete translation cycle of a scientific MATLAB `@serializable` class and its unit test into Python with `Serializable` and `pytest`.

---

## 1. Input: MATLAB Class (`Lattice.m`)

```matlab
classdef Lattice < serializable
    % LATTICE Direct and reciprocal crystal lattice representation
    properties (Access = private)
        a_ = 1.0;
        b_ = 1.0;
        c_ = 1.0;
        alpha_ = 90.0;
        beta_ = 90.0;
        gamma_ = 90.0;
    end

    properties (Dependent)
        a
        b
        c
        alpha
        beta
        gamma
        volume
        reciprocal_lengths
    end

    methods
        function obj = Lattice(varargin)
            if nargin == 0
                return;
            end
            pos_args = obj.saveableFields();
            [obj, remains] = obj.set_positional_and_key_val_arguments( ...
                pos_args, false, varargin{:});
            if ~isempty(remains)
                error('LATTICE:InvalidArgs', 'Unrecognised extra arguments provided');
            end
        end

        function v = get.a(obj), v = obj.a_; end
        function obj = set.a(obj, val)
            if val <= 0
                error('LATTICE:InvalidLengths', 'Lattice parameters must be positive');
            end
            obj.a_ = double(val);
            if obj.do_check_combo_arg_
                obj = obj.check_combo_arg();
            end
        end
        % (Similarly for b, c, alpha, beta, gamma ...)

        function v = get.volume(obj)
            ca = cosd(obj.alpha_);
            cb = cosd(obj.beta_);
            cg = cosd(obj.gamma_);
            v = obj.a_ * obj.b_ * obj.c_ * sqrt(1 - ca^2 - cb^2 - cg^2 + 2*ca*cb*cg);
        end

        function r = get.reciprocal_lengths(obj)
            % Reciprocal lattice vector lengths: a* = 2*pi*(b x c)/V
            sa = sind(obj.alpha_);
            sb = sind(obj.beta_);
            sg = sind(obj.gamma_);
            v = obj.volume;
            astar = (2 * pi * obj.b_ * obj.c_ * sa) / v;
            bstar = (2 * pi * obj.a_ * obj.c_ * sb) / v;
            cstar = (2 * pi * obj.a_ * obj.b_ * sg) / v;
            r = [astar, bstar, cstar];
        end

        function obj = check_combo_arg(obj)
            ca = cosd(obj.alpha_);
            cb = cosd(obj.beta_);
            cg = cosd(obj.gamma_);
            det_metric = 1 - ca^2 - cb^2 - cg^2 + 2*ca*cb*cg;
            if det_metric <= 0
                error('LATTICE:InvalidAngles', ...
                    'Unit cell angles do not form a physically valid 3D basis');
            end
        end

        function q_cart = rlu_to_cart(obj, hkl)
            % Converts (N x 3) HKL vectors to Cartesian coordinates (cubic/orthorhombic)
            if size(hkl, 2) ~= 3
                error('LATTICE:InvalidDims', 'HKL array must have 3 columns');
            end
            rl = obj.reciprocal_lengths;
            q_cart = hkl .* rl;
        end

        function ver = classVersion(~)
            ver = 1;
        end

        function flds = saveableFields(~)
            flds = {'a', 'b', 'c', 'alpha', 'beta', 'gamma'};
        end
    end
end
```

---

## 2. Input: MATLAB Unit Test (`TestLattice.m`)

```matlab
classdef TestLattice < matlab.unittest.TestCase
    methods (Test)
        function test_default_constructor(tc)
            lat = Lattice();
            tc.assertEqual(lat.a, 1.0);
            tc.assertEqual(lat.volume, 1.0, 'AbsTol', 1e-12);
            tc.assertEqual(lat.reciprocal_lengths, [2*pi, 2*pi, 2*pi], 'AbsTol', 1e-12);
        end

        function test_custom_cubic(tc)
            lat = Lattice(2.0, 2.0, 2.0, 90, 90, 90);
            tc.assertEqual(lat.volume, 8.0, 'AbsTol', 1e-12);
            tc.assertEqual(lat.reciprocal_lengths, [pi, pi, pi], 'AbsTol', 1e-12);
        end

        function test_invalid_lengths_and_angles(tc)
            tc.verifyError(@() Lattice(-1, 2, 2, 90, 90, 90), 'LATTICE:InvalidLengths');
            tc.verifyError(@() Lattice(2, 2, 2, 120, 120, 120), 'LATTICE:InvalidAngles');
        end

        function test_rlu_to_cart(tc)
            lat = Lattice(2.0, 4.0, 5.0, 90, 90, 90);
            hkl = [1, 0, 0; 0, 1, 0];
            q = lat.rlu_to_cart(hkl);
            expected = [pi, 0, 0; 0, pi/2, 0];
            tc.assertEqual(q, expected, 'AbsTol', 1e-12);
        end

        function test_serialization_roundtrip(tc)
            lat = Lattice(2.0, 3.0, 4.0, 90, 90, 120);
            bytes = lat.serialize();
            [lat_rec, nbytes] = serializable.deserialize(bytes);
            tc.assertEqual(nbytes, numel(bytes));
            tc.assertEqual(lat, lat_rec);
        end
    end
end
```

---

## 3. Step 2 Translation: Pytest Unit Test (`test_lattice.py`)

```python
"""Unit tests for Lattice class, converted from TestLattice.m."""

import math
import numpy as np
import pytest
from pyhorace.core.serializable import Serializable
from lattice import Lattice


def test_default_constructor():
    lat = Lattice()
    assert lat.a == 1.0
    assert lat.volume == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        lat.reciprocal_lengths,
        [2 * math.pi, 2 * math.pi, 2 * math.pi],
        atol=1e-12,
    )


def test_custom_cubic():
    lat = Lattice(2.0, 2.0, 2.0, 90.0, 90.0, 90.0)
    assert lat.volume == pytest.approx(8.0, abs=1e-12)
    np.testing.assert_allclose(
        lat.reciprocal_lengths, [math.pi, math.pi, math.pi], atol=1e-12
    )


def test_invalid_lengths_and_angles():
    with pytest.raises(
        ValueError, match="Lattice parameters must be positive"
    ):
        Lattice(-1.0, 2.0, 2.0, 90.0, 90.0, 90.0)

    with pytest.raises(
        ValueError, match="Unit cell angles do not form a physically valid 3D basis"
    ):
        Lattice(2.0, 2.0, 2.0, 120.0, 120.0, 120.0)


def test_rlu_to_cart():
    lat = Lattice(2.0, 4.0, 5.0, 90.0, 90.0, 90.0)
    hkl = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = lat.rlu_to_cart(hkl)
    expected = np.array([[math.pi, 0.0, 0.0], [0.0, math.pi / 2, 0.0]])
    np.testing.assert_allclose(q, expected, atol=1e-12)


def test_serialization_roundtrip():
    lat = Lattice(2.0, 3.0, 4.0, 90.0, 90.0, 120.0)
    raw_bytes = lat.serialize()
    lat_rec, nbytes = Serializable.deserialize(raw_bytes)
    assert nbytes == len(raw_bytes)
    assert lat == lat_rec
```

---

## 4. Step 3 Translation: Python Class (`lattice.py`)

Notice how:
1. Stored properties (`a`, `b`, `c`, `alpha`, `beta`, `gamma`) are declared directly as Pydantic fields rather than repeating MATLAB's boilerplate `Dependent` properties + private `a_` attributes.
2. Single-field validation (`a, b, c > 0`) uses `@field_validator`, while multi-property validation uses `check_combo_arg(self)` (which `Serializable` automatically invokes after initialization and on attribute assignment when `self.do_check_combo_arg` is `True`).
3. `scipy.special.cosdg` and `sindg` are used so `cosd(90.0) == 0.0` exactly.

```python
"""Crystal lattice representation translated from Lattice.m."""

from __future__ import annotations

import math
from typing import Any, ClassVar, Sequence
import numpy as np
from pydantic import field_validator
from scipy.special import cosdg, sindg

from pyhorace.core.serializable import Serializable


class Lattice(Serializable):
    """Direct and reciprocal crystal lattice representation."""

    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    _fields_to_save: ClassVar[Sequence[str]] = (
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
    )

    @field_validator("a", "b", "c")
    @classmethod
    def _validate_positive_lengths(cls, val: float) -> float:
        if val <= 0:
            raise ValueError("Lattice parameters must be positive")
        return float(val)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if args or kwargs:
            _, remains = self.set_positional_and_key_val_arguments(
                self.saveable_fields(), False, *args, **kwargs
            )
            if remains:
                raise ValueError("Unrecognised extra arguments provided")

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Sequence[str]:
        return self._fields_to_save

    def check_combo_arg(self) -> "Lattice":
        ca = float(cosdg(self.alpha))
        cb = float(cosdg(self.beta))
        cg = float(cosdg(self.gamma))
        det_metric = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
        if det_metric <= 0.0:
            raise ValueError(
                "Unit cell angles do not form a physically valid 3D basis"
            )
        return self

    @property
    def volume(self) -> float:
        """Calculates direct unit cell volume in Angstrom^3."""
        ca = float(cosdg(self.alpha))
        cb = float(cosdg(self.beta))
        cg = float(cosdg(self.gamma))
        arg = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
        return self.a * self.b * self.c * math.sqrt(max(arg, 0.0))

    @property
    def reciprocal_lengths(self) -> np.ndarray:
        """Reciprocal lattice vector lengths [a*, b*, c*] in Angstrom^-1."""
        sa = float(sindg(self.alpha))
        sb = float(sindg(self.beta))
        sg = float(sindg(self.gamma))
        v = self.volume
        astar = (2.0 * math.pi * self.b * self.c * sa) / v
        bstar = (2.0 * math.pi * self.a * self.c * sb) / v
        cstar = (2.0 * math.pi * self.a * self.b * sg) / v
        return np.array([astar, bstar, cstar], dtype=np.float64)

    def rlu_to_cart(self, hkl: np.ndarray) -> np.ndarray:
        """Convert (N x 3) reciprocal lattice units to Cartesian coordinates."""
        hkl_arr = np.asarray(hkl, dtype=np.float64)
        if hkl_arr.ndim != 2 or hkl_arr.shape[1] != 3:
            raise ValueError("HKL array must have 3 columns")
        return hkl_arr * self.reciprocal_lengths
```

---

## 5. Step 4 Verification: Running `pytest`

```bash
$ pytest test_lattice.py -v
============================= test session starts =============================
collected 5 items

test_lattice.py::test_default_constructor PASSED                        [ 20%]
test_lattice.py::test_custom_cubic PASSED                               [ 40%]
test_lattice.py::test_invalid_lengths_and_angles PASSED                 [ 60%]
test_lattice.py::test_rlu_to_cart PASSED                                [ 80%]
test_lattice.py::test_serialization_roundtrip PASSED                    [100%]

============================== 5 passed in 0.08s ==============================
```
