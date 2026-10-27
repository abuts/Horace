"""Map of constant-size blocks in legacy (v2/v3) Horace binary SQW/DND files.

Ported from:
- horace_core/sqw/file_io/@const_blocks_map/const_blocks_map.m
- horace_core/sqw/file_io/@const_blocks_map/init.m
- horace_core/sqw/file_io/@const_blocks_map/check_equal_sizes.m
- horace_core/sqw/file_io/@const_blocks_map/private/init_map_.m
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError


class CBlocksMap(dict):
    """Dictionary subclass emulating MATLAB `containers.Map` interface alongside standard `dict`."""

    def __call__(self, key: str) -> Any:
        return self[key]

    @property
    def Count(self) -> int:
        return len(self)

    def isKey(self, key: str) -> bool:
        return key in self

    def remove(self, keys: Union[str, Sequence[str]]) -> "CBlocksMap":
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            self.pop(k, None)
        return self

    def copy(self) -> "CBlocksMap":
        return CBlocksMap(super().copy())


class const_blocks_map:
    """Class supporting the map of constant-size blocks in v2/v3 SQW/DND files for in-place updates."""

    block_names_: List[str] = [
        "main_header",
        "$0_header",
        "$n_header",
        "detpar",
        "dnd_methadata",
        "dnd_data",
        "pix",
        "instr_head",
        "instrument",
        "sample_head",
        "sample",
    ]

    block_positions_: List[Tuple[Union[str, List[str]], Union[str, List[str]]]] = [
        (["main_head_pos_info_", "nfiles_pos_"], "header_pos_"),
        (["header_pos_info_", "efix_pos_"], "detpar_pos_"),
        (["header_pos_info_", "start_pos_"], ["header_pos_info_", "efix_pos_"]),
        (["detpar_pos_info_", "ndet_pos_"], "data_pos_"),
        (["data_fields_locations_", "alatt_pos_"], "s_pos_"),
        ("s_pos_", "dnd_eof_pos_"),
        ("pix_pos_", "eof_pix_pos_"),
        ("instrument_head_pos_", "instrument_pos_"),
        ("instrument_pos_", "sample_head_pos_"),
        ("sample_head_pos_", "sample_pos_"),
        ("sample_pos_", "instr_sample_end_pos_"),
    ]

    const_block_map_: Dict[str, Tuple[Union[str, List[str]], Union[str, List[str]]]] = dict(
        zip(block_names_, block_positions_)
    )

    not_fit_: List[str] = ["instr_head", "instrument", "sample_head", "sample"]

    def __init__(self, pos_info: Optional[Dict[str, Any]] = None) -> None:
        self.cblocks_map_: CBlocksMap = CBlocksMap()
        if pos_info is not None:
            self.init(pos_info)

    @property
    def cblocks_map(self) -> CBlocksMap:
        return self.cblocks_map_

    def set_cblock_param(
        self, block_name: str, pos: int, sz: int
    ) -> "const_blocks_map":
        self.cblocks_map_[block_name] = np.array([[pos], [sz]], dtype=np.int64)
        return self

    def get_must_fit(self) -> CBlocksMap:
        if not self.cblocks_map_:
            return CBlocksMap()
        mp = self.cblocks_map_.copy()
        mp.remove(self.not_fit_)
        return mp

    def init(self, pos_info: Dict[str, Any]) -> "const_blocks_map":
        bsm = self._init_map(pos_info)
        if "num_contrib_files_" in pos_info and not isinstance(pos_info["num_contrib_files_"], str):
            nfiles = int(pos_info["num_contrib_files_"])
            if nfiles > 1 and "$n_header" in bsm and "$0_header" in bsm:
                v_n = np.asarray(bsm["$n_header"])
                v_0 = np.asarray(bsm["$0_header"])
                if v_n.ndim == 1:
                    v_n = v_n.reshape(2, -1)
                if v_0.ndim == 1:
                    v_0 = v_0.reshape(2, -1)
                bsm["header"] = np.hstack([v_n, v_0])
            elif "$0_header" in bsm:
                bsm["header"] = bsm["$0_header"]
            bsm.remove(["$0_header", "$n_header"])
        self.cblocks_map_ = bsm
        return self

    def _init_map(self, pos_info: Dict[str, Any]) -> CBlocksMap:
        bsm = CBlocksMap()
        for the_key in self.block_names_:
            fld_range = self.const_block_map_[the_key]
            ok1, s1 = self._get_value(pos_info, fld_range[0])
            if not ok1:
                continue
            ok2, s2 = self._get_value(pos_info, fld_range[1])
            if not ok2:
                raise HoraceError(
                    "HORACE:const_block_map:invalid_argument",
                    f"Field {fld_range[1]} for block {the_key} is not among input positions structure",
                )
            s1_arr = np.atleast_1d(s1) if s1 is not None and not isinstance(s1, str) else np.array([])
            s2_arr = np.atleast_1d(s2) if s2 is not None and not isinstance(s2, str) else np.array([])
            if s1_arr.size == 1 and s2_arr.size >= 1:
                v1 = int(s1_arr.flat[0])
                v2 = int(s2_arr.flat[0])
                sz = v2 - v1
                if sz > 0:
                    bsm[the_key] = np.array([[v1], [sz]], dtype=np.int64)
                else:
                    bsm[the_key] = np.array([], dtype=np.int64)
            else:
                if the_key.startswith("$0"):
                    if s1_arr.size >= 1 and s2_arr.size >= 1:
                        v1_end = int(s1_arr.flat[-1])
                        v2_end = int(s2_arr.flat[-1])
                        bsm[the_key] = np.array([[v1_end], [v2_end - v1_end]], dtype=np.int64)
                    else:
                        bsm[the_key] = np.array([], dtype=np.int64)
                elif the_key.startswith("$n"):
                    if s1_arr.size >= 2 and s2_arr.size >= 2:
                        start = s2_arr[:-1].astype(np.int64)
                        size = (s1_arr[1:] - s2_arr[:-1]).astype(np.int64)
                        bsm[the_key] = np.vstack([start, size])
                    else:
                        bsm[the_key] = np.array([], dtype=np.int64)
                elif s1_arr.size == 0 or s2_arr.size == 0:
                    bsm[the_key] = np.array([], dtype=np.int64)
                else:
                    v1 = int(s1_arr.flat[0])
                    v2 = int(s2_arr.flat[0])
                    sz = v2 - v1
                    if sz > 0:
                        bsm[the_key] = np.array([[v1], [sz]], dtype=np.int64)
                    else:
                        bsm[the_key] = np.array([], dtype=np.int64)
        return bsm

    @classmethod
    def _get_value(
        cls, struc: Any, key: Union[str, Sequence[str]]
    ) -> Tuple[bool, Any]:
        if struc is None or isinstance(struc, str):
            return False, []
        if isinstance(key, (list, tuple)):
            if len(key) == 0:
                return False, []
            first = key[0]
            if isinstance(struc, dict):
                if first not in struc:
                    return False, []
                subs = struc[first]
            elif hasattr(struc, first):
                subs = getattr(struc, first)
            else:
                return False, []
            if subs is None or (isinstance(subs, (list, tuple, dict, np.ndarray)) and len(subs) == 0):
                return True, []
            if len(key) == 1:
                return True, subs
            return cls._get_value(subs, key[1:])
        else:
            if isinstance(struc, (list, tuple, np.ndarray)):
                if len(struc) == 0:
                    return False, []
                vals = []
                for item in struc:
                    if isinstance(item, dict) and key in item:
                        vals.append(item[key])
                    elif hasattr(item, key):
                        vals.append(getattr(item, key))
                    else:
                        return False, []
                return True, np.asarray(vals)
            elif isinstance(struc, dict):
                if key not in struc:
                    return False, []
                return True, struc[key]
            elif hasattr(struc, key):
                return True, getattr(struc, key)
            return False, []

    def check_equal_sizes(self, other_obj: "const_blocks_map") -> Tuple[bool, str]:
        bs1 = self.get_must_fit()
        bs2 = other_obj.get_must_fit()
        for the_key in bs2.keys():
            if the_key not in bs1:
                return False, f"Base object does not have block with name: {the_key}"
            bl1 = np.asarray(bs1[the_key])
            bl2 = np.asarray(bs2[the_key])
            if bl1.size == bl2.size:
                if bl1.size == 0:
                    continue
                b1_2d = bl1.reshape(2, -1)
                b2_2d = bl2.reshape(2, -1)
                for j in range(b1_2d.shape[1]):
                    if b1_2d[1, j] != b2_2d[1, j]:
                        return (
                            False,
                            f"The sizes of the block: {the_key}, element N{j + 1}/of{b1_2d.shape[1]} are different",
                        )
            else:
                return (
                    False,
                    f"The number of records ({bl1.size}) for the block: {the_key}, different from the N-records to upgrade ({bl2.size})",
                )
        return True, ""


ConstBlocksMap = const_blocks_map

__all__ = [
    "CBlocksMap",
    "ConstBlocksMap",
    "const_blocks_map",
]

