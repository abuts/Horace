import numpy as np
from typing import Any, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
from ..core.data_op_interface import DataOpInterface, matlab_size
from ..core.serializable import Serializable, compare_values


@dataclass(eq=False)
class IXDataset(DataOpInterface, Serializable):
    """
    Base class for N-dimensional datasets (equivalent to Herbert's IX_dataset_*d).
    """
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    error: np.ndarray = field(default_factory=lambda: np.array([]))
    title: str = ""

    def __init__(
        self,
        signal: Any = None,
        error: Any = None,
        title: str = "",
    ):
        Serializable.__init__(self)
        self.signal = np.asarray([] if signal is None else signal, dtype=float)
        self.error = np.asarray([] if error is None else error, dtype=float)
        self.title = title
        if self.signal.shape != self.error.shape:
            if self.error.size == 0:
                self.error = np.zeros_like(self.signal)
            else:
                raise ValueError("Signal and error shapes must match.")

    @property
    def ndim(self) -> int:
        return self.signal.ndim

    def get_NUM_DIMS(self) -> int:
        sz = [d for d in matlab_size(self.signal) if d != 1]
        return 1 if len(sz) == 0 else len(sz)

    def get_img_size(self) -> Tuple[int, ...]:
        return matlab_size(self.signal)

    def sigvar(self) -> Any:
        from ..data.sigvar import sigvar
        return sigvar(self.signal, self.error**2)

    def sigvar_size(self) -> Tuple[int, ...]:
        return matlab_size(self.signal)

    def sigvar_set(self, sigvarobj: Any) -> "IXDataset":
        from ..data.sigvar import sigvar
        if not isinstance(sigvarobj, sigvar):
            raise ValueError("Source of new signal and variance data must be a sigvar object")
        if self.sigvar_size() != sigvarobj.sigvar_size():
            raise ValueError("IXDataset object and sigvar object have inconsistent sizes")
        self.signal = sigvarobj.s.reshape(self.signal.shape, order="F")
        self.error = np.sqrt(sigvarobj.e).reshape(self.error.shape, order="F")
        return self

    def copy(self) -> "IXDataset":
        return IXDataset(signal=self.signal.copy(), error=self.error.copy(), title=self.title)

    def saveable_fields(self) -> Sequence[str]:
        return ["signal", "error", "title"]

    def to_dict(self) -> dict:
        return {
            "signal": self.signal.tolist(),
            "error": self.error.tolist(),
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IXDataset":
        return cls(
            signal=np.array(data.get("signal", [])),
            error=np.array(data.get("error", [])),
            title=data.get("title", ""),
        )
