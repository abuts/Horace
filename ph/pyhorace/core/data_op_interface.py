import numpy as np
from typing import Any, Union, TypeVar
import copy

from .sigvar import SigVar

T = TypeVar('T', bound='DataOpInterface')

class DataOpInterface:
    """
    Mixin class providing arithmetic operators with automatic error propagation
    for classes that contain `signal` and `error` attributes (representing variance).
    """

    @property
    def _sigvar(self) -> SigVar:
        return SigVar(getattr(self, 'signal'), getattr(self, 'error'))

    def _from_sigvar(self: T, sv: SigVar) -> T:
        new_obj = copy.deepcopy(self)
        setattr(new_obj, 'signal', sv.s)
        setattr(new_obj, 'error', sv.e)
        return new_obj

    def __add__(self: T, other: Any) -> T:
        if isinstance(other, DataOpInterface):
            return self._from_sigvar(self._sigvar + other._sigvar)
        return self._from_sigvar(self._sigvar + other)

    def __radd__(self: T, other: Any) -> T:
        return self.__add__(other)

    def __sub__(self: T, other: Any) -> T:
        if isinstance(other, DataOpInterface):
            return self._from_sigvar(self._sigvar - other._sigvar)
        return self._from_sigvar(self._sigvar - other)

    def __rsub__(self: T, other: Any) -> T:
        return self._from_sigvar(other - self._sigvar)

    def __mul__(self: T, other: Any) -> T:
        if isinstance(other, DataOpInterface):
            return self._from_sigvar(self._sigvar * other._sigvar)
        return self._from_sigvar(self._sigvar * other)

    def __rmul__(self: T, other: Any) -> T:
        return self.__mul__(other)

    def __truediv__(self: T, other: Any) -> T:
        if isinstance(other, DataOpInterface):
            return self._from_sigvar(self._sigvar / other._sigvar)
        return self._from_sigvar(self._sigvar / other)

    def __rtruediv__(self: T, other: Any) -> T:
        return self._from_sigvar(other / self._sigvar)

    def __pow__(self: T, power: Union[float, int]) -> T:
        return self._from_sigvar(self._sigvar ** power)
