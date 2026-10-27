"""Block Allocation Table (BAT) and pack_blocks helper for Horace binary v4 format."""

from __future__ import annotations

import copy
import struct
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.exceptions import HerbertError, HoraceError
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.io.data_block import data_block


def pack_blocks(
    free_spaces: np.ndarray,
    block_sizes: Union[Sequence[int], np.ndarray],
    last_gap_pos: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Place blocks into available free spaces or append at the end of the file.

    Parameters
    ----------
    free_spaces : np.ndarray
        2xN array where row 0 is start positions and row 1 is space sizes.
    block_sizes : Sequence[int] or np.ndarray
        1D array of sizes of blocks to pack.
    last_gap_pos : int
        End-of-file position (must be > 0).

    Returns
    -------
    positions : np.ndarray
        1D array of assigned positions for each block.
    free_spaces : np.ndarray
        Updated 2xM array of remaining free spaces.
    last_gap_pos : int
        Updated end-of-file position.
    """
    if free_spaces is None:
        raise HerbertError(
            "HERBERT:utilities:invalid_argument",
            "free_spaces must be a 2xN numeric array",
        )
    fs = np.asarray(free_spaces)
    if fs.ndim != 2 or (fs.size > 0 and fs.shape[0] != 2) or not np.issubdtype(fs.dtype, np.number):
        if fs.size == 0 and fs.ndim == 2 and (fs.shape[0] == 2 or fs.shape[1] == 0):
            fs = np.zeros((2, 0), dtype=np.int64)
        else:
            raise HerbertError(
                "HERBERT:utilities:invalid_argument",
                "free_spaces must be a 2xN numeric array",
            )
    else:
        fs = fs.astype(np.int64, copy=True)

    bs = np.asarray(block_sizes)
    if bs.ndim > 2 or (bs.ndim == 2 and min(bs.shape) > 1) or not np.issubdtype(bs.dtype, np.number):
        if bs.size == 0 and np.issubdtype(bs.dtype, np.number):
            bs = np.zeros(0, dtype=np.int64)
        else:
            raise HerbertError(
                "HERBERT:utilities:invalid_argument",
                "block_sizes must be a 1D numeric vector",
            )
    else:
        bs = bs.reshape(-1).astype(np.int64, copy=True)

    if not isinstance(last_gap_pos, (int, float, np.number)) or np.ndim(last_gap_pos) != 0 or last_gap_pos <= 0:
        raise HerbertError(
            "HERBERT:utilities:invalid_argument",
            "last_gap_pos must be a positive numeric scalar",
        )
    eof_pos = int(last_gap_pos)

    n_blocks = bs.size
    if n_blocks == 0:
        return np.zeros(0, dtype=np.uint64), fs.astype(np.uint64), eof_pos

    # Sort block_sizes descending (stable sort matching MATLAB sort(..., 'descend'))
    block_ids = np.argsort(-bs, kind="mergesort")
    block_sizes_sorted = bs[block_ids]

    # Sort free_spaces columns by size (row 1) descending (stable)
    if fs.shape[1] > 0:
        fs_order = np.argsort(-fs[1, :], kind="mergesort")
        fs = fs[:, fs_order]

    positions = np.zeros(n_blocks, dtype=np.int64)
    for i in range(n_blocks):
        bsize = int(block_sizes_sorted[i])
        bid = int(block_ids[i])

        # 1. Look for exact fit
        exact_idx = np.flatnonzero(fs[1, :] == bsize) if fs.shape[1] > 0 else np.array([], dtype=int)
        if exact_idx.size > 0:
            fit_idx = int(exact_idx[0])
            positions[bid] = fs[0, fit_idx]
            fs[1, fit_idx] = 0
            continue

        # 2. Look for larger fit
        larger_idx = np.flatnonzero(fs[1, :] > bsize) if fs.shape[1] > 0 else np.array([], dtype=int)
        if larger_idx.size > 0:
            fit_idx = int(larger_idx[0])
            positions[bid] = fs[0, fit_idx]
            fs[0, fit_idx] += bsize
            fs[1, fit_idx] -= bsize
            continue

        # 3. Place at last_gap_pos
        positions[bid] = eof_pos
        eof_pos += bsize

    if fs.shape[1] > 0:
        fs = fs[:, fs[1, :] > 0]
    else:
        fs = np.zeros((2, 0), dtype=np.int64)

    return positions, fs, eof_pos


def _merge_adjacent_space(fs: np.ndarray) -> np.ndarray:
    """Merge contiguous free space blocks together."""
    if fs.size == 0 or fs.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.uint64)
    fs = np.asarray(fs, dtype=np.uint64)
    if fs.shape[1] == 1:
        return fs.copy()

    indx = np.argsort(fs[0, :], kind="mergesort")
    fs = fs[:, indx]

    start_pos = fs[0, :]
    block_sizes = fs[1, :]
    end_pos = start_pos + block_sizes
    adjacent = np.concatenate(([False], start_pos[1:] == end_pos[:-1]))
    if np.any(adjacent):
        n_new = int( start_pos.size - np.sum(adjacent) )
        fs_new = np.zeros((2, n_new), dtype=np.uint64)
        ic = 0
        for i in range(start_pos.size):
            if adjacent[i]:
                fs_new[1, ic - 1] += fs[1, i]
            else:
                fs_new[0, ic] = fs[0, i]
                fs_new[1, ic] = fs[1, i]
                ic += 1
        return fs_new
    return fs


@register_serializable
class blockAllocationTable(Serializable):
    """Block Allocation Table (BAT) managing binary data blocks in Horace v4 files."""

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self.position_: int = 0
        self.bat_blocks_size_: int = 0
        self.initialized_: bool = False
        self.blocks_list_: List[data_block] = []
        self.block_names_: List[str] = []
        self.free_spaces_and_size_: np.ndarray = np.zeros((2, 0), dtype=np.uint64)
        self.end_of_file_pos_: int = 0

        if len(args) > 0:
            self.init(*args)

    def init(self, location: int, block_list: Sequence[data_block]) -> "blockAllocationTable":
        """Initialize empty BAT with location and block list."""
        self.position = location
        self.blocks_list = list(block_list)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def bat_bin_size(self) -> int:
        return int(self.bat_blocks_size_ + 4)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks_list_)

    @property
    def initialized(self) -> bool:
        return bool(self.initialized_)

    @property
    def position(self) -> int:
        return int(self.position_)

    @position.setter
    def position(self, val: int) -> None:
        self._set_bat_position(val)

    @property
    def blocks_list(self) -> List[data_block]:
        return self.blocks_list_

    @blocks_list.setter
    def blocks_list(self, val: Sequence[data_block]) -> None:
        self._set_block_list(val)

    @property
    def block_names(self) -> List[str]:
        return list(self.block_names_)

    @property
    def blocks_start_position(self) -> int:
        return int(self.position + self.bat_bin_size)

    @property
    def free_spaces_and_size(self) -> np.ndarray:
        return self.free_spaces_and_size_

    @property
    def end_of_file_pos(self) -> int:
        return int(self.end_of_file_pos_)

    @property
    def ba_table(self) -> np.ndarray:
        return self._get_ba_table_bindata()

    @ba_table.setter
    def ba_table(self, bindata: Union[bytes, bytearray, np.ndarray]) -> None:
        self._set_ba_table_from_bindata(bindata)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def set_changed_block(
        self, bl_instance: data_block, bl_index: int
    ) -> "blockAllocationTable":
        """Set data block without recalculating BAT."""
        bl_name = bl_instance.block_name
        if bl_name not in self.block_names_:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"Block with name {bl_name} is not registered in BAT",
            )
        bl_ind_present = self.block_names_.index(bl_name)
        bl_present = self.blocks_list_[bl_ind_present]
        # Accept either 0-based or 1-based index if matched
        if bl_ind_present != bl_index and (bl_ind_present + 1) != bl_index:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"The block {bl_name} have changed its size or postion. This is not allowed without recalculating BAT",
            )
        if bl_present.position != bl_instance.position or bl_present.size != bl_instance.size:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"The block {bl_name} have changed its size or postion. This is not allowed without recalculating BAT",
            )
        self.blocks_list_[bl_ind_present] = bl_instance
        return self

    def get_data_block(
        self,
        block_name_or_instance: Union[str, data_block, type],
        nargout: Optional[int] = None,
    ) -> Union[data_block, Tuple[data_block, int]]:
        """Get a data_block in the BAT by name or instance."""
        if nargout is None:
            nargout = _infer_nargout()
        the_block, bl_ind = self._get_data_block(block_name_or_instance)
        if nargout <= 1:
            return the_block
        return the_block, bl_ind

    def set_data_block(self, block_instance: data_block) -> "blockAllocationTable":
        """Set data block with defined position and size in BAT."""
        return self._set_data_block(block_instance)

    def find_block_place(
        self,
        data_block_or_name: Union[str, data_block, type],
        block_size: Optional[int] = None,
    ) -> Tuple["blockAllocationTable", int, bool]:
        """Find place to put a data block within BAT."""
        if isinstance(data_block_or_name, type) and issubclass(data_block_or_name, data_block):
            data_block_or_name = data_block_or_name()
        if isinstance(data_block_or_name, data_block):
            if block_size is None:
                block_size = data_block_or_name.size
        elif isinstance(data_block_or_name, str):
            if block_size is None:
                raise HoraceError(
                    "HORACE:blockAllocationTable:invalid_argument",
                    "If block name provided as input of the method, block size have to be provided. It has been not",
                )
        else:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"Method accepts either data_block class instance or the name of the block in BAT. "
                f"The class of the parameter is {type(data_block_or_name).__name__}",
            )
        return self._find_block_place(data_block_or_name, int(block_size))

    def init_obj_info(
        self, obj_to_analyze: Any, *args: str
    ) -> "blockAllocationTable":
        """Initialize BAT for the sqw/dnd object provided as input."""
        nocache = False
        insertion = False
        test_mode = False
        for arg in args:
            if not isinstance(arg, str):
                raise HoraceError(
                    "HORACE:blockAllocationTable:invalid_argument",
                    f"Unrecognized option: {arg!r}",
                )
            low = arg.lower()
            if "-nocache".startswith(low) and len(low) >= 2:
                nocache = True
            elif "-insertion".startswith(low) and len(low) >= 2:
                insertion = True
            elif "-test_mode".startswith(low) and len(low) >= 2:
                test_mode = True
            else:
                raise HoraceError(
                    "HORACE:blockAllocationTable:invalid_argument",
                    f"Unrecognized option: {arg!r}",
                )
        if insertion:
            raise HoraceError(
                "HORACE:blockAllocationTable:not_implemented",
                "insertion mode is not implemented",
            )
        return self._init_obj_info(obj_to_analyze, nocache, test_mode)

    def get_block_pos(
        self, block_name_or_class: Union[str, data_block, type]
    ) -> int:
        """Return the position of a block defined by current BAT."""
        return self._get_block_pos(block_name_or_class)

    def place_undocked_blocks(
        self, obj_to_write: Any, nocache: bool = True
    ) -> "blockAllocationTable":
        """Replace contents of unlocked blocks and find places for them in BAT."""
        return self._place_undocked_blocks(obj_to_write, bool(nocache))

    def put_bat(
        self, fid: Any, position: Optional[int] = None
    ) -> "blockAllocationTable":
        """Store block allocation table at specified location in binary file."""
        if position is None:
            position = self.position
        return self._store_bat(fid, int(position))

    def get_bat(
        self, fid: Any, position: Optional[int] = None
    ) -> "blockAllocationTable":
        """Read block allocation table from specified location in binary file."""
        if position is None:
            position = self.position
        return self._restore_bat(fid, int(position))

    def clear(self) -> "blockAllocationTable":
        """Nullify the positions of data blocks for all blocks."""
        new_list: List[data_block] = []
        for bl in self.blocks_list_:
            bl_copy = copy.copy(bl)
            bl_copy.position = 0
            new_list.append(bl_copy)
        self.blocks_list_ = new_list
        self.initialized_ = False
        self.free_spaces_and_size_ = np.zeros((2, 0), dtype=np.uint64)
        self.end_of_file_pos_ = int(self.position + self.bat_bin_size)
        return self

    def clear_unlocked_blocks(self) -> "blockAllocationTable":
        """Clear positions of all unlocked blocks and update free spaces."""
        return self._clear_unlocked_blocks()

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def classVersion(self) -> int:
        return 1

    def class_version(self) -> int:
        return 1

    def saveableFields(self) -> List[str]:
        return ["position", "ba_table"]

    def saveable_fields(self) -> List[str]:
        return ["position", "ba_table"]

    def from_bare_struct(self, s: dict) -> "blockAllocationTable":
        if "position" in s:
            self.position = s["position"]
        if "ba_table" in s:
            self.ba_table = s["ba_table"]
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _set_bat_position(self, val: int) -> "blockAllocationTable":
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)) or np.ndim(val) != 0 or val < 0:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"Allocation block position should be defined by non-negative scalar. It is {val!r}",
            )
        val_int = int(val)
        old_pos = int(self.position_)
        self.position_ = val_int
        pos_shift = val_int - old_pos
        if pos_shift == 0 or not self.initialized:
            return self

        for i in range(self.n_blocks):
            bl_copy = copy.copy(self.blocks_list_[i])
            bl_copy.position = int(bl_copy.position) + pos_shift
            self.blocks_list_[i] = bl_copy

        if self.free_spaces_and_size_.shape[1] > 0:
            self.free_spaces_and_size_ = self.free_spaces_and_size_.copy()
            for i in range(self.free_spaces_and_size_.shape[1]):
                self.free_spaces_and_size_[0, i] = int(self.free_spaces_and_size_[0, i]) + pos_shift
        return self

    def _set_block_list(self, val: Sequence[data_block]) -> "blockAllocationTable":
        if not isinstance(val, (list, tuple)):
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"Block list value should be cellarray of data_block classes. It is: {type(val).__name__}",
            )
        for idx, item in enumerate(val):
            if not isinstance(item, data_block):
                raise HoraceError(
                    "HORACE:blockAllocationTable:invalid_argument",
                    f"Block list value should be cellarray of data_block classes.\n"
                    f"First non-data_block number is {idx + 1} and its class is: {type(item).__name__}",
                )
        self.blocks_list_ = [copy.copy(b) for b in val]
        self.free_spaces_and_size_ = np.zeros((2, 0), dtype=np.uint64)

        total_bl_size, initialized, bat_blocks_size, block_names, eof_pos = (
            self._calculate_bat_size_and_free_spaces()
        )
        self.bat_blocks_size_ = bat_blocks_size
        self.block_names_ = block_names

        if not initialized:
            self.end_of_file_pos_ = int(self.position_ + self.bat_bin_size)
            self.initialized_ = False
            return self

        self.initialized_ = True
        self.end_of_file_pos_ = int(eof_pos)
        if self.blocks_start_position + total_bl_size != eof_pos:
            self._find_free_spaces()
        return self

    def _calculate_bat_size_and_free_spaces(self) -> Tuple[int, bool, int, List[str], int]:
        n_blocks = len(self.blocks_list_)
        if n_blocks == 0:
            bat_blocks_size = 0
        else:
            bat_blocks_size = 4
        name_list: List[str] = []
        eof_pos = bat_blocks_size
        bl_contents_size = 0
        initialized = False
        for block in self.blocks_list_:
            initialized = initialized or block.allocated
            bl_contents_size += int(block.size)
            bat_blocks_size += int(block.bat_record_size)
            name_list.append(block.block_name)
            if initialized:
                eof_pos = max(eof_pos, int(block.position) + int(block.size))
            else:
                eof_pos = bat_blocks_size
        return bl_contents_size, initialized, bat_blocks_size, name_list, eof_pos

    def _find_free_spaces(self) -> None:
        n_blocks = len(self.blocks_list_)
        bl_ps = np.zeros((2, n_blocks + 1), dtype=np.int64)
        bl_ps[0, 0] = 0
        bl_ps[1, 0] = self.blocks_start_position

        for i, block in enumerate(self.blocks_list_):
            bl_ps[0, i + 1] = int(block.position)
            bl_ps[1, i + 1] = int(block.size)

        if np.any(bl_ps[0, 1:] < self.blocks_start_position):
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                "Data block location overlaps with the space, occupied by BAT",
            )

        indx = np.argsort(bl_ps[0, :], kind="mergesort")
        bl_ps = bl_ps[:, indx]
        bl_ps_end = bl_ps[0, :] + bl_ps[1, :]
        empty_spaces = bl_ps[0, 1:] - bl_ps_end[:-1]
        if np.any(empty_spaces < 0):
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                "Some blocks overlep with each other. BAT is invalid",
            )
        free_space_id = empty_spaces > 0
        bl_ps_end_prev = bl_ps_end[:-1]
        if np.any(free_space_id):
            self.free_spaces_and_size_ = np.vstack(
                (bl_ps_end_prev[free_space_id], empty_spaces[free_space_id])
            ).astype(np.uint64)
        else:
            self.free_spaces_and_size_ = np.zeros((2, 0), dtype=np.uint64)

    def _get_data_block(
        self, block_name_or_instance: Union[str, data_block, type]
    ) -> Tuple[data_block, int]:
        if isinstance(block_name_or_instance, type) and issubclass(block_name_or_instance, data_block):
            block_name = block_name_or_instance().block_name
        elif isinstance(block_name_or_instance, data_block):
            block_name = block_name_or_instance.block_name
        elif isinstance(block_name_or_instance, str):
            block_name = block_name_or_instance
        else:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"Method accepts either data_block class instance or the name of the block in BAT. "
                f"The class of the input is {type(block_name_or_instance).__name__}",
            )
        if block_name not in self.block_names_:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"The block with name {block_name} is not registered in block allocation table",
            )
        bl_ind = self.block_names_.index(block_name)
        return copy.copy(self.blocks_list_[bl_ind]), bl_ind

    def _get_block_pos(
        self, block: Union[str, data_block, type]
    ) -> int:
        if not self.initialized_:
            raise HoraceError(
                "HORACE:blockAllocationTable:runtime_error",
                "Attempt to obtain block position, but the BAT have not been initialized",
            )
        if isinstance(block, type) and issubclass(block, data_block):
            block_name = block().block_name
        elif isinstance(block, data_block):
            block_name = block.block_name
        elif isinstance(block, str):
            block_name = block
        else:
            block_name = str(block)

        if block_name not in self.block_names_:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"Block with Name {block_name} is not among the blocks, defined in BAT",
            )
        bl_ind = self.block_names_.index(block_name)
        return int(self.blocks_list_[bl_ind].position)

    def _set_data_block(self, block_instance: data_block) -> "blockAllocationTable":
        targ_name = block_instance.block_name
        targ_start = int(block_instance.position)
        targ_size = int(block_instance.size)

        initialized_list = []
        targ_found_idx = -1
        start_positions = []
        sizes = []
        other_names = []

        for idx, blk in enumerate(self.blocks_list_):
            if blk.block_name == targ_name:
                targ_found_idx = idx
            else:
                if blk.allocated:
                    initialized_list.append(True)
                    start_positions.append(int(blk.position))
                    sizes.append(int(blk.size))
                    other_names.append(blk.block_name)

        if targ_found_idx < 0:
            raise HoraceError(
                "HORACE:blockAllocatinTable:invalid_argument",
                f"Block with name {targ_name} is not described in the blockAllocationTable",
            )
        if targ_start < self.blocks_start_position:
            raise HoraceError(
                "HORACE:blockAllocatinTable:invalid_argument",
                f"Block: {targ_name} requested position overlaps with location of BAT",
            )

        first_pos = np.array(start_positions, dtype=np.int64)
        next_pos = first_pos + np.array(sizes, dtype=np.int64)
        targ_end = targ_start + targ_size - 1

        if first_pos.size > 0:
            overlap = ((targ_start >= first_pos) & (targ_start < next_pos)) | (
                (targ_end >= first_pos) & (targ_end < next_pos)
            )
            if np.any(overlap):
                overlapped_names = [other_names[i] for i in np.flatnonzero(overlap)]
                raise HoraceError(
                    "HORACE:blockAllocatinTable:invalid_argument",
                    f"Block {targ_name} location overlaps with location of existing blocks {overlapped_names}",
                )

        self.blocks_list_[targ_found_idx] = copy.copy(block_instance)
        self.end_of_file_pos_ = max(
            int(self.end_of_file_pos_), int(block_instance.position) + int(block_instance.size)
        )

        first_pos_all = np.concatenate(([0], first_pos, [targ_start]))
        next_pos_all = np.concatenate(([self.blocks_start_position], next_pos, [targ_end + 1]))
        s_ind = np.argsort(first_pos_all, kind="mergesort")
        first_pos_all = first_pos_all[s_ind]
        next_pos_all = next_pos_all[s_ind]

        free_spaces = first_pos_all[1:] - next_pos_all[:-1] - 1
        valid = free_spaces > 0
        if np.any(valid):
            next_valid = next_pos_all[:-1][valid]
            free_valid = free_spaces[valid]
            self.free_spaces_and_size_ = np.vstack((next_valid, free_valid)).astype(np.uint64)
        else:
            self.free_spaces_and_size_ = np.zeros((2, 0), dtype=np.uint64)
        return self

    def _find_block_place(
        self,
        data_block_name_or_class: Union[str, data_block],
        block_size: int,
    ) -> Tuple["blockAllocationTable", int, bool]:
        the_block_orig, bl_ind = self._get_data_block(data_block_name_or_class)
        the_block = copy.copy(the_block_orig)
        if block_size == the_block.size and the_block.allocated:
            new_pos = int(the_block.position)
            return self, new_pos, False

        if the_block.allocated:
            old_block_place = np.array([[the_block.position], [the_block.size]], dtype=np.uint64)
            if self.free_spaces_and_size_.size > 0:
                fs = np.hstack((self.free_spaces_and_size_, old_block_place))
            else:
                fs = old_block_place
            fs = _merge_adjacent_space(fs)
        else:
            fs = self.free_spaces_and_size_.copy()

        free_space_size = fs[1, :] if fs.shape[1] > 0 else np.zeros(0, dtype=np.uint64)
        will_fit = free_space_size >= block_size

        if not np.any(will_fit):
            new_pos = int(self.end_of_file_pos)
            all_free = int(np.sum(free_space_size))
            compress_bat = bool(all_free >= block_size)
            self.end_of_file_pos_ = int(new_pos + block_size)
            the_block.position = new_pos
            the_block.size = block_size
        else:
            compress_bat = False
            min_fit_size = np.min(free_space_size[will_fit])
            min_indices = np.flatnonzero(free_space_size == min_fit_size)
            min_idx = int(min_indices[0])

            new_pos = int(fs[0, min_idx])
            the_block.position = new_pos
            the_block.size = block_size

            if int(fs[1, min_idx]) > block_size:
                fs[0, min_idx] = new_pos + block_size
                fs[1, min_idx] = int(fs[1, min_idx]) - block_size
            else:
                fs = np.delete(fs, min_idx, axis=1)

        if fs.shape[1] > 0 and int(fs[0, -1]) + int(fs[1, -1]) == self.end_of_file_pos:
            self.end_of_file_pos_ = int(fs[0, -1])
            fs = fs[:, :-1]

        self.free_spaces_and_size_ = fs.astype(np.uint64)
        self.blocks_list_[bl_ind] = the_block
        return self, new_pos, compress_bat

    def _init_obj_info(
        self, obj_to_analyze: Any, nocache: bool, test_mode: bool
    ) -> "blockAllocationTable":
        n_b = self.n_blocks
        block_size = np.zeros(n_b, dtype=np.uint64)
        for i in range(n_b):
            db = copy.copy(self.blocks_list_[i])
            db = db.calc_obj_size(obj_to_analyze, nocache, test_mode)
            block_size[i] = db.size
            self.blocks_list_[i] = db

        block_pos = np.concatenate(
            ([np.uint64(0)], np.cumsum(block_size, dtype=np.uint64))
        ) + np.uint64(self.blocks_start_position)

        for i in range(n_b):
            self.blocks_list_[i].position = int(block_pos[i])

        self.end_of_file_pos_ = int(block_pos[-1])
        self.initialized_ = True
        return self

    def _place_undocked_blocks(
        self, obj_to_write: Any, nocache: bool
    ) -> "blockAllocationTable":
        n_blocks = self.n_blocks
        new_block_sizes = np.zeros(n_blocks, dtype=np.int64)
        unlocked = np.zeros(n_blocks, dtype=bool)
        for i in range(n_blocks):
            block = copy.copy(self.blocks_list_[i])
            if block.locked:
                continue
            unlocked[i] = True
            block = block.calc_obj_size(obj_to_write, nocache, False)
            new_block_sizes[i] = int(block.size)
            self.blocks_list_[i] = block

        block_sizes = new_block_sizes[unlocked]
        positions, free_spaces, eof_pos = pack_blocks(
            self.free_spaces_and_size_, block_sizes, self.end_of_file_pos_
        )
        self.free_spaces_and_size_ = np.asarray(free_spaces, dtype=np.uint64)
        self.end_of_file_pos_ = int(eof_pos)
        ic_changed = 0
        for i in range(n_blocks):
            if not unlocked[i]:
                continue
            block = copy.copy(self.blocks_list_[i])
            block.position = int(positions[ic_changed])
            ic_changed += 1
            self.blocks_list_[i] = block
        return self

    def _clear_unlocked_blocks(self) -> "blockAllocationTable":
        bl_l: List[data_block] = []
        bl_spaces: List[np.ndarray] = []
        for bl in self.blocks_list_:
            bl_copy = copy.copy(bl)
            if bl_copy.locked:
                bl_l.append(bl_copy)
            else:
                bl_spaces.append(
                    np.array([[bl_copy.position], [bl_copy.size]], dtype=np.uint64)
                )
                bl_copy.position = 0
                bl_l.append(bl_copy)

        self.blocks_list_ = bl_l
        if bl_spaces:
            bl_space = np.hstack(bl_spaces)
        else:
            bl_space = np.zeros((2, 0), dtype=np.uint64)

        if self.free_spaces_and_size_.size > 0:
            if bl_space.size > 0:
                bl_space = np.hstack((bl_space, self.free_spaces_and_size_))
            else:
                bl_space = self.free_spaces_and_size_.copy()

        bl_space = _merge_adjacent_space(bl_space)
        if (
            bl_space.shape[1] > 0
            and int(bl_space[0, -1]) + int(bl_space[1, -1]) == int(self.end_of_file_pos_)
        ):
            self.end_of_file_pos_ = int(bl_space[0, -1])
            bl_space = bl_space[:, :-1]

        if int(self.end_of_file_pos_) == int(self.blocks_start_position):
            self.initialized_ = False

        self.free_spaces_and_size_ = bl_space.astype(np.uint64)
        return self

    def _get_ba_table_bindata(self) -> np.ndarray:
        n_obj = self.n_blocks
        parts = [np.frombuffer(struct.pack("<I", n_obj), dtype=np.uint8)]
        for i in range(n_obj):
            parts.append(np.asarray(self.blocks_list_[i].bat_record, dtype=np.uint8).reshape(-1))
        return np.concatenate(parts)

    def _set_ba_table_from_bindata(
        self, bindata: Union[bytes, bytearray, np.ndarray]
    ) -> "blockAllocationTable":
        if isinstance(bindata, (bytes, bytearray)):
            arr = np.frombuffer(bindata, dtype=np.uint8)
        elif isinstance(bindata, np.ndarray) and bindata.dtype == np.uint8:
            arr = bindata.reshape(-1)
        else:
            raise HoraceError(
                "HORACE:blockAllocationTable:invalid_argument",
                f"bat_table setter accepts only uint8 array of elements, containing serialized BlockAllocationTable. "
                f"The input class is: {type(bindata).__name__}",
            )
        self.bat_blocks_size_ = int(arr.size)
        raw = arr.tobytes()
        n_bl = struct.unpack_from("<I", raw, 0)[0]
        pos = 5  # 1-based position for deserialize_bat_record
        cl_table: List[data_block] = []
        while pos <= len(raw):
            blk, pos = data_block.deserialize_bat_record(arr, pos)
            cl_table.append(blk)
            if len(cl_table) == n_bl:
                break
        self.blocks_list = cl_table
        return self

    def _store_bat(self, fid: Any, pos: int) -> "blockAllocationTable":
        from pyhorace.io.horace_binfile_interface import horace_binfile_interface
        from pyhorace.io.sqw_fopen import do_fwrite

        try:
            horace_binfile_interface.move_to_position(fid, pos)
        except HoraceError as e:
            if e.identifier == "HORACE:data_block:io_error":
                raise HoraceError(
                    "HORACE:data_block:io_error",
                    f"{e.message} moving to the start of the BlockAllocationTable postition",
                ) from e
            raise

        bindata = self.ba_table
        raw = bindata.tobytes()
        do_fwrite(fid, struct.pack("<I", len(raw)), "uint8")
        do_fwrite(fid, raw, "uint8")

        try:
            horace_binfile_interface.check_write_error(fid)
        except HoraceError as e:
            if e.identifier == "HORACE:data_block:io_error":
                raise HoraceError(
                    "HORACE:data_block:io_error",
                    f"{e.message} the BlockAllocationTable data",
                ) from e
            raise
        return self

    def _restore_bat(self, fid: Any, pos: int) -> "blockAllocationTable":
        from pyhorace.io.horace_binfile_interface import horace_binfile_interface
        from pyhorace.io.sqw_fopen import do_fread

        try:
            horace_binfile_interface.move_to_position(fid, pos)
        except HoraceError as e:
            if e.identifier == "HORACE:data_block:io_error":
                raise HoraceError(
                    "HORACE:data_block:io_error",
                    f"{e.message} moving to the start of the BlockAllocationTable postition",
                ) from e
            raise

        bat_size_arr = do_fread(fid, 1, "uint32")
        try:
            horace_binfile_interface.check_read_error(fid)
        except HoraceError as e:
            if e.identifier == "HORACE:data_block:io_error":
                raise HoraceError(
                    "HORACE:data_block:io_error",
                    f"{e.message} the BlockAllocationTable data",
                ) from e
            raise

        bat_size = int(bat_size_arr[0]) if bat_size_arr.size > 0 else 0
        bindata = do_fread(fid, bat_size, "*uint8")
        try:
            horace_binfile_interface.check_read_error(fid)
        except HoraceError as e:
            if e.identifier == "HORACE:data_block:io_error":
                raise HoraceError(
                    "HORACE:data_block:io_error",
                    f"{e.message} the BlockAllocationTable data",
                ) from e
            raise

        self.ba_table = bindata
        return self

