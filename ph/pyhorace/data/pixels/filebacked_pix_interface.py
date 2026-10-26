"""File-backed pixel accessor interface and stub implementation.

In MATLAB Horace, ``PixelDataFileBacked``, ``pix_write_handle``, and
``pixfile_combine_info`` interact with binary files via ``f_accessor_``
(either ``memmapfile`` or ``horace_binfile_interface`` / ``faccess_sqw_v*``).

In PyHorace, all file-backed pixel access is routed through
:class:`FileBackedPixInterface`. Until the full SQW binary file readers/writers
(``faccess_sqw_v*``) are implemented, :class:`FileBackedPixStub` provides the
file-backed pixel accessor stub used by ``PixelDataFileBacked``.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError


class FileBackedPixInterface(abc.ABC):
    """Abstract interface for file-backed pixel storage (replaces ``f_accessor_``)."""

    DEFAULT_NUM_PIX_FIELDS: int = 9
    DATA_FORMAT: np.dtype = np.dtype(np.float32)

    @property
    @abc.abstractmethod
    def full_filename(self) -> str:
        """Full path to the underlying pixel file."""

    @full_filename.setter
    @abc.abstractmethod
    def full_filename(self, value: str) -> None:
        """Set full path to the underlying pixel file."""

    @property
    @abc.abstractmethod
    def num_pixels(self) -> int:
        """Total number of pixels stored in the file."""

    @property
    @abc.abstractmethod
    def offset(self) -> int:
        """Byte offset to the start of the pixel block in the file."""

    @property
    @abc.abstractmethod
    def tail(self) -> int:
        """Number of trailing bytes after the pixel block in the file."""

    @property
    @abc.abstractmethod
    def writable(self) -> bool:
        """Whether the file accessor is opened with write permissions."""

    @property
    @abc.abstractmethod
    def is_activated(self) -> bool:
        """Whether the file handle / accessor is currently active (open)."""

    @abc.abstractmethod
    def get_pix_slice(
        self,
        field_indices: Union[int, Sequence[int], slice],
        start_pix: int,
        end_pix: int,
    ) -> np.ndarray:
        """Read pixel fields for 1-based inclusive pixel range ``[start_pix, end_pix]``.

        Parameters
        ----------
        field_indices : int, sequence of int, or slice
            0-based row indices (in ``0..8``) of pixel fields to read.
        start_pix : int
            1-based starting pixel index (inclusive).
        end_pix : int
            1-based ending pixel index (inclusive).

        Returns
        -------
        np.ndarray
            2D float64 array of shape ``(num_fields, end_pix - start_pix + 1)``.
        """

    @abc.abstractmethod
    def set_pix_slice(
        self,
        field_indices: Union[int, Sequence[int], slice],
        start_pix: int,
        end_pix: int,
        values: np.ndarray,
    ) -> None:
        """Write pixel fields for 1-based inclusive pixel range ``[start_pix, end_pix]``."""

    @abc.abstractmethod
    def get_pix_at_indices(
        self,
        field_indices: Union[int, Sequence[int], slice],
        pix_indices: np.ndarray,
    ) -> np.ndarray:
        """Read pixel fields at arbitrary 1-based pixel indices ``pix_indices``."""

    @abc.abstractmethod
    def set_pix_at_indices(
        self,
        field_indices: Union[int, Sequence[int], slice],
        pix_indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Write pixel fields at arbitrary 1-based pixel indices ``pix_indices``."""

    @abc.abstractmethod
    def deactivate(self) -> None:
        """Close the underlying file handle so the file can be moved or deleted."""

    @abc.abstractmethod
    def activate(self, new_filename: Optional[str] = None, writable: Optional[bool] = None) -> None:
        """Re-open the underlying file handle (optionally at ``new_filename``)."""

    def open_raw_memmap(self) -> Any:
        """Direct low-level binary memmap accessor (unimplemented stub)."""
        raise NotImplementedError(
            "HORACE:FileBackedPixInterface:NotImplemented - "
            "Direct low-level binary memmapfile access is not yet implemented; "
            "use FileBackedPixInterface methods instead."
        )


def _normalize_row_indices(
    field_indices: Union[int, Sequence[int], slice, np.ndarray]
) -> Union[slice, np.ndarray]:
    if isinstance(field_indices, slice):
        return field_indices
    if isinstance(field_indices, (int, np.integer)):
        return np.array([int(field_indices)], dtype=np.intp)
    arr = np.asarray(field_indices, dtype=np.intp).ravel()
    return arr


class FileBackedPixStub(FileBackedPixInterface):
    """Stub implementation of :class:`FileBackedPixInterface`.

    Provides file-backed pixel access semantics (including page-range slicing,
    1-based index gathering, single-precision float32 storage coercion, and
    file activation/deactivation lifecycle) without requiring the full MATLAB
    ``memmapfile`` / ``faccess_sqw_v*`` binary file readers to be implemented.
    """

    def __init__(
        self,
        filename: str = "",
        num_pixels: int = 0,
        offset: int = 1,
        tail: int = 0,
        writable: bool = False,
        data: Optional[np.ndarray] = None,
        create_file: bool = False,
    ) -> None:
        self._full_filename: str = str(filename) if filename else ""
        self._offset: int = int(offset)
        self._tail: int = int(tail)
        self._writable: bool = bool(writable)
        self._is_activated: bool = False

        if data is not None:
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                if arr.size == 0:
                    arr = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float32)
                else:
                    arr = arr.reshape(self.DEFAULT_NUM_PIX_FIELDS, -1, order="F")
            self._buffer: np.ndarray = np.asfortranarray(arr, dtype=np.float32).copy(order="F")
            self._num_pixels: int = int(self._buffer.shape[1])
        else:
            self._num_pixels = int(num_pixels)
            self._buffer = np.zeros(
                (self.DEFAULT_NUM_PIX_FIELDS, self._num_pixels), dtype=np.float32, order="F"
            )

        if self._full_filename:
            if create_file:
                self._write_backing_file()
                self._is_activated = True
            elif data is not None:
                self._is_activated = True
            else:
                self._load_from_file(self._full_filename)
                self._is_activated = True

    def _write_backing_file(self) -> None:
        if not self._full_filename:
            return
        parent = os.path.dirname(os.path.abspath(self._full_filename))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        # Write a simple header + float32 pixel data so file exists on disk
        # and can be moved/re-activated or inspected.
        pad_bytes = max(0, self._offset - 1)
        with open(self._full_filename, "wb") as f:
            if pad_bytes > 0:
                f.write(b"\x00" * pad_bytes)
            f.write(self._buffer.tobytes(order="F"))
            if self._tail > 0:
                f.write(b"\x00" * self._tail)

    def _load_from_file(self, filepath: str) -> None:
        if not os.path.isfile(filepath):
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"Cannot find file containing pixel data: {filepath}"
            )
        # Check if it is an SQW binary file that _read_binary_sqw_dnd can parse
        try:
            from pyhorace.sqw.read_dnd import _read_binary_sqw_dnd

            _, pix_obj, sqw_type = _read_binary_sqw_dnd(filepath, read_pixels=True)
            if sqw_type == 1 and pix_obj is not None:
                arr = np.asarray(pix_obj.data, dtype=np.float32, order="F")
                self._buffer = arr.copy(order="F")
                self._num_pixels = int(arr.shape[1])
                return
        except Exception:
            pass

        # Otherwise try reading raw float32 values at offset
        file_size = os.path.getsize(filepath)
        pad_bytes = max(0, self._offset - 1)
        avail_bytes = max(0, file_size - pad_bytes - self._tail)
        row_bytes = self.DEFAULT_NUM_PIX_FIELDS * 4
        n_pix = avail_bytes // row_bytes
        if n_pix > 0:
            with open(filepath, "rb") as f:
                if pad_bytes > 0:
                    f.seek(pad_bytes)
                raw = f.read(n_pix * row_bytes)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(
                (self.DEFAULT_NUM_PIX_FIELDS, n_pix), order="F"
            )
            self._buffer = arr.copy(order="F")
            self._num_pixels = int(n_pix)
        else:
            self._buffer = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float32, order="F")
            self._num_pixels = 0

    @classmethod
    def from_faccess(cls, faccess: Any, writable: bool = False) -> "FileBackedPixStub":
        """Create a :class:`FileBackedPixStub` from a file accessor or FakeFAccess object."""
        # Support FakeFAccess or horace_binfile_interface-like objects
        full_filename = ""
        if hasattr(faccess, "full_filename") and faccess.full_filename:
            full_filename = str(faccess.full_filename)
        elif hasattr(faccess, "filepath") and hasattr(faccess, "filename"):
            fp = str(getattr(faccess, "filepath", "") or "")
            fn = str(getattr(faccess, "filename", "") or "")
            full_filename = os.path.join(fp, fn) if fp else fn

        npix = int(getattr(faccess, "npix", getattr(faccess, "num_pixels", 0)) or 0)
        pix_pos = int(getattr(faccess, "pix_position", getattr(faccess, "offset", 1)) or 1)
        eof_pos = int(getattr(faccess, "eof_pix_position", 0) or 0)
        tail = 0

        data = None
        if hasattr(faccess, "get_pix"):
            try:
                data = faccess.get_pix(1, npix) if npix > 0 else np.empty((9, 0), dtype=np.float32)
            except TypeError:
                data = faccess.get_pix()
        elif hasattr(faccess, "fake_data"):
            data = getattr(faccess, "fake_data")
        elif hasattr(faccess, "_buffer"):
            data = getattr(faccess, "_buffer")

        if data is not None:
            return cls(
                filename=full_filename,
                num_pixels=npix,
                offset=pix_pos,
                tail=tail,
                writable=writable,
                data=data,
                create_file=False,
            )

        if full_filename and os.path.isfile(full_filename):
            if eof_pos > 0:
                fsize = os.path.getsize(full_filename)
                tail = max(0, fsize - eof_pos)
            return cls(
                filename=full_filename,
                num_pixels=npix,
                offset=pix_pos,
                tail=tail,
                writable=writable,
            )

        raise HoraceError(
            f"HORACE:PixelDataFileBacked:invalid_argument - "
            f"Cannot initialize FileBackedPixStub from {faccess!r}"
        )

    @property
    def full_filename(self) -> str:
        return self._full_filename

    @full_filename.setter
    def full_filename(self, value: str) -> None:
        self._full_filename = str(value)

    @property
    def num_pixels(self) -> int:
        return self._num_pixels

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def tail(self) -> int:
        return self._tail

    @property
    def writable(self) -> bool:
        return self._writable

    @writable.setter
    def writable(self, value: bool) -> None:
        self._writable = bool(value)

    @property
    def is_activated(self) -> bool:
        return self._is_activated

    def _check_activated(self) -> None:
        if not self._is_activated and self._full_filename:
            if not os.path.isfile(self._full_filename):
                raise HoraceError(
                    f"HORACE:PixelDataFileBacked:runtime_error - "
                    f"Filebacked object is deactivated or file does not exist: {self._full_filename}"
                )
            self._is_activated = True

    def get_pix_slice(
        self,
        field_indices: Union[int, Sequence[int], slice],
        start_pix: int,
        end_pix: int,
    ) -> np.ndarray:
        self._check_activated()
        rows = _normalize_row_indices(field_indices)
        if end_pix < start_pix or self._num_pixels == 0:
            n_rows = (
                self.DEFAULT_NUM_PIX_FIELDS
                if isinstance(rows, slice) and rows == slice(None)
                else len(range(*rows.indices(self.DEFAULT_NUM_PIX_FIELDS)))
                if isinstance(rows, slice)
                else len(rows)
            )
            return np.empty((n_rows, 0), dtype=np.float64)
        s0 = max(0, int(start_pix) - 1)
        e0 = min(self._num_pixels, int(end_pix))
        sub = self._buffer[rows, s0:e0]
        if sub.ndim == 1:
            sub = sub.reshape(1, -1)
        return np.asarray(sub, dtype=np.float64)

    def set_pix_slice(
        self,
        field_indices: Union[int, Sequence[int], slice],
        start_pix: int,
        end_pix: int,
        values: np.ndarray,
    ) -> None:
        self._check_activated()
        rows = _normalize_row_indices(field_indices)
        s0 = max(0, int(start_pix) - 1)
        e0 = int(end_pix)
        vals = np.asarray(values, dtype=np.float32)
        if e0 > self._num_pixels:
            # Expand buffer if writing past current end (e.g. via pix_write_handle)
            new_buf = np.zeros(
                (self.DEFAULT_NUM_PIX_FIELDS, e0), dtype=np.float32, order="F"
            )
            if self._num_pixels > 0:
                new_buf[:, : self._num_pixels] = self._buffer
            self._buffer = new_buf
            self._num_pixels = e0
        self._buffer[rows, s0:e0] = vals
        if self._writable and self._full_filename and os.path.isfile(self._full_filename):
            self._write_backing_file()

    def get_pix_at_indices(
        self,
        field_indices: Union[int, Sequence[int], slice],
        pix_indices: np.ndarray,
    ) -> np.ndarray:
        self._check_activated()
        rows = _normalize_row_indices(field_indices)
        idx0 = np.asarray(pix_indices, dtype=np.intp).ravel() - 1
        if idx0.size == 0 or self._num_pixels == 0:
            n_rows = (
                self.DEFAULT_NUM_PIX_FIELDS
                if isinstance(rows, slice) and rows == slice(None)
                else len(range(*rows.indices(self.DEFAULT_NUM_PIX_FIELDS)))
                if isinstance(rows, slice)
                else len(rows)
            )
            return np.empty((n_rows, 0), dtype=np.float64)
        sub = self._buffer[rows][:, idx0]
        if sub.ndim == 1:
            sub = sub.reshape(1, -1)
        return np.asarray(sub, dtype=np.float64)

    def set_pix_at_indices(
        self,
        field_indices: Union[int, Sequence[int], slice],
        pix_indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self._check_activated()
        rows = _normalize_row_indices(field_indices)
        idx0 = np.asarray(pix_indices, dtype=np.intp).ravel() - 1
        vals = np.asarray(values, dtype=np.float32)
        if isinstance(rows, slice):
            self._buffer[rows, idx0] = vals
        else:
            self._buffer[np.ix_(rows, idx0)] = (
                vals if vals.ndim == 2 else vals.reshape(len(rows), -1)
            )
        if self._writable and self._full_filename and os.path.isfile(self._full_filename):
            self._write_backing_file()

    def deactivate(self) -> None:
        """Close/release the file handle so the file can be moved or deleted."""
        self._is_activated = False

    def activate(
        self, new_filename: Optional[str] = None, writable: Optional[bool] = None
    ) -> None:
        """Re-activate the file handle, optionally updating ``full_filename``."""
        if new_filename is not None:
            self._full_filename = str(new_filename)
        if writable is not None:
            self._writable = bool(writable)
        if self._full_filename and not os.path.isfile(self._full_filename):
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"Cannot find file containing pixel data: {self._full_filename}"
            )
        self._is_activated = True

    def clone(self) -> "FileBackedPixStub":
        """Create an independent copy of this stub."""
        return FileBackedPixStub(
            filename=self._full_filename,
            num_pixels=self._num_pixels,
            offset=self._offset,
            tail=self._tail,
            writable=self._writable,
            data=self._buffer.copy(order="F"),
            create_file=False,
        )
