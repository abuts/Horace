# End-to-End Class Translation Example

This example demonstrates the complete translation cycle of a scientific MATLAB class and its unit test into Python with `pytest`.

---

## 1. Input: MATLAB Class (`Lattice.m`)

```matlab
classdef Lattice
    % LATTICE Direct and reciprocal crystal lattice representation
    properties
        a = 1.0;
        b = 1.0;
        c = 1.0;
        alpha = 90.0;
        beta = 90.0;
        gamma = 90.0;
    end

    properties (Dependent)
        volume
        reciprocal_lengths
    end

    methods
        function obj = Lattice(a, b, c, alpha, beta, gamma)
            if nargin == 0
                return;
            end
            if nargin < 6
                error('LATTICE:InvalidArgs', 'Must provide either 0 or 6 parameters');
            end
            if any([a, b, c] <= 0)
                error('LATTICE:InvalidLengths', 'Lattice parameters must be positive');
            end
            obj.a = double(a);
            obj.b = double(b);
            obj.c = double(c);
            obj.alpha = double(alpha);
            obj.beta = double(beta);
            obj.gamma = double(gamma);
        end

        function v = get.volume(obj)
            ca = cosd(obj.alpha);
            cb = cosd(obj.beta);
            cg = cosd(obj.gamma);
            v = obj.a * obj.b * obj.c * sqrt(1 - ca^2 - cb^2 - cg^2 + 2*ca*cb*cg);
        end

        function r = get.reciprocal_lengths(obj)
            % Reciprocal lattice vector lengths: a* = 2*pi*(b x c)/V
            sa = sind(obj.alpha);
            sb = sind(obj.beta);
            sg = sind(obj.gamma);
            v = obj.volume;
            astar = (2 * pi * obj.b * obj.c * sa) / v;
            bstar = (2 * pi * obj.a * obj.c * sb) / v;
            cstar = (2 * pi * obj.a * obj.b * sg) / v;
            r = [astar, bstar, cstar];
        end

        function q_cart = rlu_to_cart(obj, hkl)
            % Converts (N x 3) HKL vectors to Cartesian coordinates (cubic approximation)
            if size(hkl, 2) ~= 3
                error('LATTICE:InvalidDims', 'HKL array must have 3 columns');
            end
            rl = obj.reciprocal_lengths;
            q_cart = hkl .* rl;
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

        function test_invalid_lengths(tc)
            tc.verifyError(@() Lattice(-1, 2, 2, 90, 90, 90), 'LATTICE:InvalidLengths');
            tc.verifyError(@() Lattice(2, 2), 'LATTICE:InvalidArgs');
        end

        function test_rlu_to_cart(tc)
            lat = Lattice(2.0, 4.0, 5.0, 90, 90, 90);
            hkl = [1, 0, 0; 0, 1, 0];
            q = lat.rlu_to_cart(hkl);
            expected = [pi, 0, 0; 0, pi/2, 0];
            tc.assertEqual(q, expected, 'AbsTol', 1e-12);
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


def test_invalid_lengths():
    with pytest.raises(
        ValueError, match="Lattice parameters must be positive"
    ):
        Lattice(-1.0, 2.0, 2.0, 90.0, 90.0, 90.0)

    with pytest.raises(
        ValueError, match="Must provide either 0 or 6 parameters"
    ):
        Lattice(2.0, 2.0)


def test_rlu_to_cart():
    lat = Lattice(2.0, 4.0, 5.0, 90.0, 90.0, 90.0)
    hkl = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = lat.rlu_to_cart(hkl)
    expected = np.array([[math.pi, 0.0, 0.0], [0.0, math.pi / 2, 0.0]])
    np.testing.assert_allclose(q, expected, atol=1e-12)
```

---

## 4. Step 3 Translation: Python Class (`lattice.py`)

```python
"""Crystal lattice representation translated from Lattice.m."""

from __future__ import annotations
import math
from typing import Optional
import numpy as np


class Lattice:
    """Direct and reciprocal crystal lattice representation."""

    def __init__(
        self,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        args = [a, b, c, alpha, beta, gamma]
        provided = [x is not None for x in args]

        if not any(provided):
            # Default cubic cell
            self.a: float = 1.0
            self.b: float = 1.0
            self.c: float = 1.0
            self.alpha: float = 90.0
            self.beta: float = 90.0
            self.gamma: float = 90.0
            return

        if not all(provided):
            raise ValueError("Must provide either 0 or 6 parameters")

        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("Lattice parameters must be positive")

        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    @property
    def volume(self) -> float:
        """Calculates direct unit cell volume in Angstrom^3."""
        ca = math.cos(math.radians(self.alpha))
        cb = math.cos(math.radians(self.beta))
        cg = math.cos(math.radians(self.gamma))
        arg = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
        return self.a * self.b * self.c * math.sqrt(max(arg, 0.0))

    @property
    def reciprocal_lengths(self) -> np.ndarray:
        """Reciprocal lattice vector lengths [a*, b*, c*] in Angstrom^-1."""
        sa = math.sin(math.radians(self.alpha))
        sb = math.sin(math.radians(self.beta))
        sg = math.sin(math.radians(self.gamma))
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
collected 4 items

test_lattice.py::test_default_constructor PASSED                        [ 25%]
test_lattice.py::test_custom_cubic PASSED                               [ 50%]
test_lattice.py::test_invalid_lengths PASSED                            [ 75%]
test_lattice.py::test_rlu_to_cart PASSED                                [100%]

============================== 4 passed in 0.08s ==============================
```
