"""Common base class for Horace binary file format version 4 (DND and SQW).

Ported from:
- horace_core/sqw/file_io/@binfile_v4_common/binfile_v4_common.m
- horace_core/sqw/file_io/@binfile_v4_common/*.m
- horace_core/sqw/file_io/@binfile_v4_common/private/*.m
"""

from __future__ import annotations

import abc
import copy
import datetime
import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import Serializable
from pyhorace.io.block_allocation_table import blockAllocationTable
from pyhorace.io.data_block import data_block
from pyhorace.io.horace_binfile_interface import horace_binfile_interface
from pyhorace.io.sqw_fopen import (
    do_fread,
    do_fwrite,
    is_sqw_struct,
    pack_io_outputs,
    sqw_fopen,
)


def _extract_ignored_blocks_arg(*args: Any) -> Tuple[List[str], List[Any]]:
    """Extract `'ignore_blocks'` keyword and following block list from `args`."""
    idx: Optional[int] = None
    for i, a in enumerate(args):
        if isinstance(a, str) and a == "ignore_blocks":
            idx = i
            break
    if idx is None:
        return [], list(args)
    if idx == len(args) - 1:
        raise HoraceError(
            "HORACE:binfile_v4_common:invalid_argument",
            "ignore_blocks keyword is provided without the list to ignore which should follow the keyword",
        )
    raw_list = args[idx + 1]
    if isinstance(raw_list, str):
        ignored = [raw_list]
    elif isinstance(raw_list, (list, tuple, np.ndarray)):
        ignored = [str(x) for x in raw_list]
    else:
        raise HoraceError(
            "HORACE:binfile_v4_common:invalid_argument",
            f"Ignored block list should be cellarray of block names to ignore. It is: {raw_list!r}",
        )
    rem = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]
    return ignored, rem


class binfile_v4_common(horace_binfile_interface):
    """Common implementation for Horace v4 binary DND and SQW file accessors."""

    _v4_fields_to_save: ClassVar[List[str]] = ["full_filename", "num_dims_to_save", "bat"]

    def _init_default_attributes(self) -> None:
        super()._init_default_attributes()
        self.bat_: blockAllocationTable = blockAllocationTable(
            self.max_header_size_, self.data_blocks_list
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bat(self) -> blockAllocationTable:
        return self.bat_

    @bat.setter
    def bat(self, val: blockAllocationTable) -> None:
        if not isinstance(val, blockAllocationTable):
            raise HoraceError(
                "HORACE:binfile_v4_common:invalid_argument",
                f"Attempt to set BlockAllocationTable value using class: {type(val).__name__}",
            )
        self.bat_ = val

    @property
    def data_blocks_list(self) -> List[data_block]:
        return self.get_data_blocks()

    @property
    def data_type(self) -> str:
        return self.get_data_type()

    @property
    def dnd_dimensions(self) -> Any:
        if self.bat_ is None or not self.bat_.initialized or "bl_data_nd_data" not in self.bat_.block_names:
            return "undefined"
        bl, bl_idx = self.bat_.get_data_block("bl_data_nd_data", nargout=2)
        if not bl.dnd_info_defined and self.file_id_ > 0:
            bl.read_dnd_info(self.file_id_)
            self.bat_.set_changed_block(bl, bl_idx)
        if bl.dnd_info_defined and bl._data_size is not None:
            return list(bl._data_size)
        return "undefined"

    # ------------------------------------------------------------------
    # Abstract & Overloadable Protected Methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_data_blocks(self) -> List[data_block]:
        """Return list of `data_block` instances for this v4 file format."""

    @abc.abstractmethod
    def get_data_type(self) -> str:
        """Return legacy `data_type` string (`'b+'` for DND, `'a'` for SQW)."""

    def get_faccess_version(self) -> float:
        return 4.0

    def get_format_for_object(self) -> str:
        return "dnd"

    def get_npix_position(self) -> Any:
        if self.bat_ is None or not self.bat_.initialized:
            return []
        bl, _ = self.bat_.get_data_block("bl_data_nd_data", nargout=2)
        if not bl.dnd_info_defined:
            bl.read_dnd_info(self.file_id_)
        return bl.npix_position

    def get_creation_date(self) -> str:
        from pyhorace.sqw.dnd_base import DnDBase
        from pyhorace.sqw.sqw import SQW

        if self.bat_ is not None and self.bat_.initialized and self.file_id_ > 0:
            meta = self.get_dnd_metadata(nargout=1)
            return meta.creation_date_str
        if self.sqw_holder is not None:
            if isinstance(self.sqw_holder, DnDBase):
                return str(self.sqw_holder.creation_date)
            if isinstance(self.sqw_holder, SQW):
                return str(self.sqw_holder.main_header.creation_date)
        return super().get_creation_date()

    # ------------------------------------------------------------------
    # Initialization & Copying
    # ------------------------------------------------------------------

    def init_from_sqw_obj(self, in_obj: Any, *args: Any) -> "binfile_v4_common":
        from pyhorace.sqw.sqw import SQW

        if not (self._is_sqw_or_dnd_obj(in_obj) or is_sqw_struct(in_obj)):
            raise HoraceError(
                "HORACE:binfile_v4_common:invalid_argument",
                f"Input object must be sqw or dnd type object. It is: {type(in_obj).__name__}",
            )
        if hasattr(in_obj, "dimensions"):
            dims_res = in_obj.dimensions() if callable(in_obj.dimensions) else in_obj.dimensions
            if isinstance(dims_res, tuple) and len(dims_res) == 2 and isinstance(dims_res[0], (int, np.integer)):
                self.num_dim_ = int(dims_res[0])
            elif isinstance(dims_res, (int, np.integer)):
                self.num_dim_ = int(dims_res)
            else:
                self.num_dim_ = int(getattr(in_obj, "num_dim", len(dims_res)))
        elif isinstance(in_obj, dict) and "data" in in_obj:
            self.num_dim_ = int(len(getattr(in_obj["data"], "p", [])))

        init_args = list(args) + ["-nocache"]
        self.bat_.init_obj_info(in_obj, *init_args)
        cn = type(self).__name__.lower()
        if "dnd" in cn and isinstance(in_obj, SQW):
            self.sqw_holder_ = in_obj.data
        else:
            self.sqw_holder_ = in_obj
        return self

    def init_from_sqw_file(self, *args: Any) -> "binfile_v4_common":
        update = False
        rem_args: List[Any] = []
        for a in args:
            if isinstance(a, str) and a.strip().lower().lstrip("-") == "update":
                update = True
            else:
                rem_args.append(a)
        if rem_args:
            update = True
            self.full_filename = str(rem_args[0])
        if update:
            self.check_obj_initated_properly()

        self.bat_.get_bat(self.file_id_)
        if "bl_data_nd_data" in self.bat_.block_names:
            bl, bl_idx = self.bat_.get_data_block("bl_data_nd_data", nargout=2)
            if not bl.dnd_info_defined:
                bl.read_dnd_info(self.file_id_)
                self.bat_.set_changed_block(bl, bl_idx)
            self.num_dim_ = int(bl.dimensions) if hasattr(bl, "dimensions") and bl.dnd_info_defined else self.num_dim_
        self.data_in_file_ = True
        return self

    def copy_contents(
        self, other_obj: "horace_binfile_interface", *args: Any
    ) -> Tuple["binfile_v4_common", List[str]]:
        write_mode = False
        for a in args:
            if isinstance(a, str) and a.strip().lower().lstrip("-").startswith("write"):
                write_mode = True

        flds = self.saveable_fields()
        missing_fields: List[str] = []
        self.do_check_combo_arg = False
        for fld in flds:
            try:
                val = getattr(other_obj, fld)
                setattr(self, fld, copy.deepcopy(val))
            except Exception:
                missing_fields.append(fld)
        self.do_check_combo_arg = True
        self.check_combo_arg()

        facc_mode = other_obj.io_mode
        if write_mode and (facc_mode == "rb" or (facc_mode == "wb+" and other_obj.data_in_file)):
            self.data_in_file_ = True
            facc_mode = "rb+"

        if not (not facc_mode and not self.full_filename):
            if self.full_filename and os.path.isfile(self.full_filename):
                self.file_id_ = sqw_fopen(self.full_filename, facc_mode or "rb")
                self.read_only_ = (facc_mode or "rb") in ("rb", "r")
        return self, missing_fields

    def do_class_dependent_updates(
        self, new_ldr: "horace_binfile_interface", *args: Any
    ) -> "horace_binfile_interface":
        if self.faccess_version != new_ldr.faccess_version:
            dnd_obj = self.get_dnd(nargout=1)
            new_ldr.sqw_holder = dnd_obj
            new_ldr.put_dnd()
            self.delete()
        return new_ldr

    # ------------------------------------------------------------------
    # Generic v4 Block Operations
    # ------------------------------------------------------------------

    def put_app_header(self, *args: Any) -> "binfile_v4_common":
        super().put_app_header(*args)
        self.bat_.put_bat(self.file_id_)
        return self

    def put_all_blocks(self, *args: Any) -> "binfile_v4_common":
        if len(args) > 0 and (self._is_sqw_or_dnd_obj(args[0]) or is_sqw_struct(args[0])):
            sqw_dnd_data = args[0]
            self.sqw_holder_ = sqw_dnd_data
            self.bat_.init_obj_info(sqw_dnd_data)
            rem_args = args[1:]
        else:
            sqw_dnd_data = self.sqw_holder
            rem_args = args

        ignored_blocks_list, _ = _extract_ignored_blocks_arg(*rem_args)
        self.put_app_header()

        bl_list = list(self.bat_.blocks_list)
        for i, bl in enumerate(bl_list):
            if bl.block_name in ignored_blocks_list:
                continue
            bl_list[i] = bl.put_sqw_block(self.file_id_, sqw_dnd_data)
        self.bat_.blocks_list = bl_list
        self.bat_.put_bat(self.file_id_)
        return self

    def get_all_blocks(self, *args: Any, nargout: Optional[int] = None) -> Any:
        from pyhorace.sqw.d0d import d0d
        from pyhorace.sqw.sqw import SQW

        ignored_blocks_list, argi = _extract_ignored_blocks_arg(*args)
        obj_to_set: Any = None
        if len(argi) > 0:
            if isinstance(argi[0], (str, os.PathLike)):
                self.init(str(argi[0]))
            else:
                obj_to_set = argi[0]
        if len(argi) > 1:
            if obj_to_set is not None:
                raise HoraceError(
                    "HORACE:binfile_v4_common:invalid_argument",
                    f"Two input parameters identified as object to modify: {type(argi[0]).__name__} and {type(argi[1]).__name__}",
                )
            obj_to_set = argi[1]
        elif obj_to_set is None:
            if isinstance(self.sqw_holder_, dict):
                obj_to_set = {}
            elif "dnd" in type(self).__name__.lower():
                obj_to_set = d0d()
            else:
                obj_to_set = SQW()

        is_ser = isinstance(obj_to_set, Serializable)
        if is_ser:
            obj_to_set.do_check_combo_arg = False

        for bl in self.bat_.blocks_list:
            if bl.block_name in ignored_blocks_list:
                continue
            _, obj_to_set = bl.get_sqw_block(self.file_id_, obj_to_set)

        if is_ser:
            obj_to_set.do_check_combo_arg = True
            obj_to_set.check_combo_arg()

        self.sqw_holder_ = obj_to_set
        if nargout == 1:
            return self
        return self, obj_to_set

    def get_sqw_block(
        self, block_name_or_class: Union[str, data_block], *args: Any, nargout: Optional[int] = None
    ) -> Any:
        if len(args) > 0:
            if len(args) > 1 or not isinstance(args[0], (str, os.PathLike)):
                raise HoraceError(
                    "HORACE:binfile_v4_common:invalid_argument",
                    f"Third argument of get_sqw_block can be only filename. You have provided: {args!r}",
                )
            self.init(str(args[0]))

        if not self.bat_.initialized:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                "Attempt to get sqw block using non-initialized file-accessor",
            )
        if self.file_id_ <= 0:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                f'Attempt to get sqw block using file-accessor with sqw file "{self.full_filename} " closed',
            )
        the_data_block = self.bat_.get_data_block(block_name_or_class, nargout=1)
        _, set_obj = the_data_block.get_sqw_block(self.file_id_, self.sqw_holder)
        if self.sqw_holder is not None and self._is_sqw_or_dnd_obj(set_obj):
            self.sqw_holder_ = set_obj
        if nargout == 1:
            return self
        return self, set_obj

    def put_sqw_block(
        self, block_name_or_class: Union[str, data_block], *args: Any, nargout: Optional[int] = None
    ) -> Any:
        no_init = False
        argi: List[Any] = []
        for a in args:
            if isinstance(a, str) and a.strip().lower().lstrip("-") == "noinit":
                no_init = True
            else:
                argi.append(a)

        if not no_init and not self.bat_.initialized and len(argi) > 0 and (self._is_sqw_or_dnd_obj(argi[0]) or is_sqw_struct(argi[0])):
            self.init(*argi)
            return self._put_sqw_block_impl(block_name_or_class, nargout=nargout)

        return self._put_sqw_block_impl(block_name_or_class, *argi, nargout=nargout)

    def _put_sqw_block_impl(
        self,
        block_name_or_instance: Union[str, data_block],
        obj_to_work_with: Any = None,
        *,
        nargout: Optional[int] = None,
    ) -> Any:
        if not self.bat_.initialized:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                "Attempt to put sqw block using non-initialized file-accessor",
            )
        if self.file_id_ <= 0:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                f'Attempt to put sqw block using file-accessor with closed or undefined sqw file: "{self.full_filename} "',
            )
        if obj_to_work_with is None:
            obj_to_work_with = self.sqw_holder

        the_data_block = self.bat_.get_data_block(block_name_or_instance, nargout=1)
        the_data_block.calc_obj_size(obj_to_work_with)
        bat, pos, compress = self.bat_.find_block_place(the_data_block)
        self.bat_ = bat
        the_data_block.position = pos
        the_data_block.put_sqw_block(self.file_id_)
        self.bat_.put_bat(self.file_id_)
        if compress:
            self.compress_file()
        return pack_io_outputs((self, the_data_block), nargout, default=1)

    def get_block_data(
        self, block_name_or_instance: Union[str, data_block], *args: Any, nargout: Optional[int] = None
    ) -> Any:
        if len(args) > 0:
            self._check_file_defined_and_exist("read", block_name_or_instance, *args)
            self.init(*args)
        if not self.bat_.initialized:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                "Attempt to get sqw block using non-initialized file-accessor",
            )
        if self.file_id_ <= 0:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                f'Attempt to get sqw block using file-accessor with sqw file "{self.full_filename} " closed',
            )
        the_data_block = self.bat_.get_data_block(block_name_or_instance, nargout=1)
        _, blk = the_data_block.get_sqw_block(self.file_id_, nargout=2)
        return pack_io_outputs((blk, self), nargout, default=1)

    def put_block_data(
        self, block_name_or_instance: Union[str, data_block], *args: Any
    ) -> "binfile_v4_common":
        if len(args) > 0:
            self._check_file_defined_and_exist("write", block_name_or_instance, *args)
            if not self.bat_.initialized:
                self.init(*args)
            else:
                old_obj = self.sqw_holder
                self.put_sqw_block(block_name_or_instance, args[0])
                self.sqw_holder_ = old_obj
                return self
        self.put_sqw_block(block_name_or_instance)
        return self

    def _check_file_defined_and_exist(
        self, io_mode: str, block_name_or_instance: Union[str, data_block], *args: Any
    ) -> None:
        file_4_io: Optional[str] = None
        for a in args:
            if isinstance(a, (str, os.PathLike)) and not str(a).startswith("-"):
                file_4_io = str(a)
                break
        if file_4_io is None:
            file_4_io = self.full_filename
        if not file_4_io or not os.path.isfile(file_4_io):
            bln = (
                block_name_or_instance.block_name
                if isinstance(block_name_or_instance, data_block)
                else str(block_name_or_instance)
            )
            raise HoraceError(
                "HORACE:faccess_dnd_v4:invalid_argument",
                f'File "{file_4_io} " to {io_mode} block {bln} is missing. Can not {io_mode} block. Incorrect filename?',
            )

    def compress_file(self) -> "binfile_v4_common":
        warnings.warn(
            "binary file compression have not been implemented yet",
            UserWarning,
            stacklevel=2,
        )
        return self

    def put_new_blocks_values(self, obj_to_write: Any, *args: Any) -> "binfile_v4_common":
        ignore_locks = False
        nocache = False
        rem_args: List[Any] = []
        for a in args:
            if isinstance(a, str):
                s = a.strip().lower()
                if s == "-ignore_locks":
                    ignore_locks = True
                    continue
                if s == "-nocache":
                    nocache = True
                    continue
            rem_args.append(a)
        return self._put_new_blocks_values_impl(
            obj_to_write, not ignore_locks, nocache, *rem_args
        )

    def _put_new_blocks_values_impl(
        self, obj_to_write: Any, keep_locked: bool, nocache: bool, *args: Any
    ) -> "binfile_v4_common":
        if not self.bat_.initialized:
            raise HoraceError(
                "HORACE:file_io:runtime_error",
                "attempting to put data using non-initialized file-accessor",
            )
        self.reopen_to_write()

        bl_names_to_change = self._parse_blocks_to_change_input(*args)
        BAT = self.bat_
        n_blocks = BAT.n_blocks
        all_bl_names = BAT.block_names
        was_locked = [False] * n_blocks

        for i in range(n_blocks):
            was_locked[i] = BAT.blocks_list[i].locked
            if keep_locked and was_locked[i]:
                continue
            if all_bl_names[i] in bl_names_to_change:
                BAT.blocks_list[i].locked = False
            else:
                BAT.blocks_list[i].locked = True

        BAT.clear_unlocked_blocks()
        BAT.place_undocked_blocks(obj_to_write, nocache)

        for i in range(n_blocks):
            locked = BAT.blocks_list[i].locked
            BAT.blocks_list[i].locked = was_locked[i]
            if locked:
                continue
            blk = BAT.blocks_list[i]
            if nocache:
                blk.put_sqw_block(self.file_id_, obj_to_write, True)
            else:
                blk.put_sqw_block(self.file_id_)
            BAT.blocks_list[i] = blk

        self.bat_ = BAT
        self.bat_.put_bat(self.file_id_)
        return self

    def _parse_blocks_to_change_input(self, *args: Any) -> List[str]:
        exclude_idx: Optional[int] = None
        update_idx: Optional[int] = None
        for i, a in enumerate(args):
            if isinstance(a, str):
                if a.startswith("exclude"):
                    exclude_idx = i
                elif a.startswith("update"):
                    update_idx = i
        if exclude_idx is not None and update_idx is not None:
            raise HoraceError(
                "HORACE:file_io:invalid_argument",
                '"excluded" and "update" keywords can not be specified together',
            )
        if exclude_idx is None and update_idx is None:
            return [bl.block_name for bl in self.bat_.blocks_list if not bl.locked]

        existing_bl_names = self.bat_.block_names
        kw_idx = exclude_idx if exclude_idx is not None else update_idx
        assert kw_idx is not None
        raw_list = args[kw_idx + 1]
        if isinstance(raw_list, str):
            list_provided = [raw_list]
        else:
            list_provided = [str(x) for x in raw_list]

        unknown = [nm for nm in list_provided if nm not in existing_bl_names]
        if unknown:
            raise HoraceError(
                "HORACE:file_io:invalid_argument",
                f'The block name(s): "{unknown}" are not familiar to faccessor {type(self).__name__}\n'
                f"Known block names are:\n {existing_bl_names}",
            )
        if exclude_idx is not None:
            return [nm for nm in existing_bl_names if nm not in list_provided]
        return list_provided

    # ------------------------------------------------------------------
    # DND Access Methods (Common to v4 DND and SQW)
    # ------------------------------------------------------------------

    def get_dnd_data(self, *args: Any, nargout: Optional[int] = None) -> Any:
        return self.get_block_data("bl_data_nd_data", *args, nargout=nargout)

    def get_dnd_metadata(self, *args: Any, nargout: Optional[int] = None) -> Any:
        return self.get_block_data("bl_data_metadata", *args, nargout=nargout)

    def get_dnd(self, *args: Any, nargout: Optional[int] = None) -> Any:
        from pyhorace.sqw.dnd_base import DnDBase

        header_only = False
        verbatim = False
        noclass = False
        rem_args: List[Any] = []
        for a in args:
            if isinstance(a, str) and a.startswith("-"):
                s = a.strip().lower().lstrip("-")
                if s in ("h", "he", "hea", "head", "heade", "header", "his", "history"):
                    header_only = True
                elif s in ("v", "ve", "ver", "verb", "verba", "verbat", "verbati", "verbatim"):
                    verbatim = True
                elif s in ("hv", "hve", "hver", "hverb", "hverba", "hverbat", "hverbati", "hverbatim"):
                    header_only = True
                    verbatim = True
                elif s.startswith("noc") or s in ("legacy", "sqw_struc"):
                    noclass = True
                else:
                    raise HoraceError(
                        "HORACE:binfile_v2_common:invalid_argument",
                        f"get_data::Error: Unrecognized option {a!r}",
                    )
            else:
                rem_args.append(a)

        if rem_args:
            self.init(*rem_args)

        if isinstance(self.num_dim, str) or self.num_dim is None:
            raise HoraceError(
                "HORACE:faccess_dnd_v4:runtime_error",
                "get_data method called on un-initialized loader",
            )

        dnd_obj_meta = self.get_dnd_metadata(nargout=1)
        if header_only:
            dnd_obj: Any = dnd_obj_meta
        else:
            dnd_obj_data = self.get_dnd_data(nargout=1)
            dnd_obj = DnDBase.dnd(dnd_obj_meta, dnd_obj_data)

        if not verbatim:
            dnd_obj.filename = self.filename
            dnd_obj.filepath = self.filepath

        if noclass:
            dnd_struc = dnd_obj.to_bare_struct()
            dnd_struc["title"] = dnd_obj.title
            dnd_struc["filename"] = dnd_obj.filename
            dnd_struc["filepath"] = dnd_obj.filepath
            dnd_struc["img_range"] = getattr(dnd_obj, "img_range", getattr(dnd_obj.axes, "img_range", None))
            dnd_obj = dnd_struc

        return pack_io_outputs((dnd_obj, self), nargout, default=1)

    def get_data(self, *args: Any, nargout: Optional[int] = None) -> Any:
        argi = list(args)
        has_key = any(isinstance(a, str) and a.startswith("-") for a in argi)
        if has_key and "-noclass" not in argi:
            argi.append("-noclass")
        return self.get_dnd(*argi, nargout=nargout)

    def get_se_npix(self, *args: Any, nargout: Optional[int] = None) -> Any:
        data_dnd = self.get_dnd_data(*args, nargout=1)
        data_str = {
            "s": np.asarray(data_dnd.sig, dtype=np.float64),
            "e": np.asarray(data_dnd.err, dtype=np.float64),
            "npix": np.asarray(data_dnd.npix, dtype=np.float64),
        }
        return pack_io_outputs((data_str, self), nargout, default=1)

    def get_img_db_range(self, *args: Any) -> np.ndarray:
        ds = self.get_dnd_metadata(nargout=1)
        return np.asarray(ds.axes.img_range, dtype=np.float64)

    def get_npix_block(self, bin_start: int, bin_end: int) -> np.ndarray:
        if not self.bat_.initialized:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                "Attempt to get npix data using non-initialized file-accessor",
            )
        if self.file_id_ <= 0:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                f'Attempt to get npix data using file-accessor with closed or undefined sqw file: "{self.full_filename} "',
            )
        n_elem = int(bin_end - bin_start + 1)
        img_acc_block = self.bat_.get_data_block("bl_data_nd_data", nargout=1)
        if not img_acc_block.dnd_info_defined:
            img_acc_block.read_dnd_info(self.file_id_)
        if bin_start > bin_end or n_elem > img_acc_block.data_size:
            raise HoraceError(
                "HORACE:binfile_v4_common:invalid_argument",
                f"Attempt to read npix data from pos: {bin_start} to pos: {bin_end}\n"
                f"The existing npix block contains {img_acc_block.data_size} elements.\n"
                "Read goes outside of the ranges of the npix block",
            )
        npix_pos = int(img_acc_block.npix_position) + (int(bin_start) - 1) * 8
        self.move_to_position(self.file_id_, npix_pos)
        npix, _ = do_fread(self.file_id_, n_elem, "uint64")
        self.check_read_error(self.file_id_, "npix")
        return npix.astype(np.float64)

    def put_dnd(self, *args: Any, **kwargs: Any) -> "binfile_v4_common":
        from pyhorace.sqw.sqw import SQW

        if len(args) > 0:
            init_args = list(args)
            has_obj = False
            for i, a in enumerate(init_args):
                if self._is_sqw_or_dnd_obj(a) or is_sqw_struct(a):
                    has_obj = True
                    if isinstance(a, SQW):
                        init_args[i] = a.data
            if has_obj:
                self.init(*init_args)

        if self.sqw_holder is None:
            raise HoraceError(
                "HORACE:faccess_dnd_v4:runtime_error",
                f"Attempt to store dnd object in file {self.full_filename} but the dnd object have not been defined",
            )
        sqh = self.sqw_holder
        if hasattr(sqh, "creation_date_defined") and not sqh.creation_date_defined:
            sqh.creation_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            self.sqw_holder_ = sqh
            self.bat_.init_obj_info(sqh)

        self.put_app_header()
        self.put_dnd_metadata()
        self.put_dnd_data()
        return self

    def put_dnd_data(self, dnd_obj: Any = None) -> "binfile_v4_common":
        from pyhorace.sqw.dnd_base import DnDBase
        from pyhorace.sqw.dnd_data import dnd_data

        if dnd_obj is not None:
            if not isinstance(dnd_obj, (DnDBase, dnd_data)):
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"This method accepts instance of DnD object or dnd_data as input. In fact you provided object of class {type(dnd_obj).__name__}",
                )
            self.put_sqw_block("bl_data_nd_data", dnd_obj, "-noinit")
        else:
            self.put_sqw_block("bl_data_nd_data")
        return self

    def put_dnd_metadata(self, dnd_obj: Any = None) -> "binfile_v4_common":
        from pyhorace.sqw.dnd_base import DnDBase
        from pyhorace.sqw.dnd_metadata import dnd_metadata

        if dnd_obj is not None:
            if not isinstance(dnd_obj, (DnDBase, dnd_metadata)):
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"This method accepts instance of DnD object as input. In fact you provided object of class {type(dnd_obj).__name__}",
                )
            self.put_sqw_block("bl_data_metadata", dnd_obj, "-noinit")
        else:
            self.put_sqw_block("bl_data_metadata")
        return self

    def put_senpix_block(self, img_block: Dict[str, Any], pos: int) -> "binfile_v4_common":
        if not self.bat_.initialized:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                "Attempt to put sqw block using non-initialized file-accessor",
            )
        if self.file_id_ <= 0:
            raise HoraceError(
                "HORACE:binfile_v4_common:runtime_error",
                f'Attempt to put sqw block using file-accessor with closed or undefined sqw file: "{self.full_filename} "',
            )
        if not (isinstance(img_block, dict) and all(k in img_block for k in ("s", "e", "npix"))):
            raise HoraceError(
                "HORACE:binfile_v4_common:invalid_argument",
                "Attempt to store img_block which does not contain necessary information",
            )
        s_arr = np.asarray(img_block["s"], dtype=np.float64).ravel(order="F")
        e_arr = np.asarray(img_block["e"], dtype=np.float64).ravel(order="F")
        npix_arr = np.asarray(img_block["npix"], dtype=np.uint64).ravel(order="F")
        n_elem = int(npix_arr.size)
        if not (s_arr.size == e_arr.size == n_elem):
            raise HoraceError(
                "HORACE:binfile_v4_common:invalid_argument",
                "number of elements for fields s,e,npix in the input image block must be equal",
            )

        img_acc_block = self.bat_.get_data_block("bl_data_nd_data", nargout=1)
        if not img_acc_block.dnd_info_defined:
            img_acc_block.read_dnd_info(self.file_id_)
        if pos + n_elem > img_acc_block.data_size:
            raise HoraceError(
                "HORACE:binfile_v4_common:invalid_argument",
                f"New image block will be written in pos {pos} and contains {n_elem} elements.\n"
                f"Total space allocated for image contains {img_acc_block.data_size} elements.\n"
                "Attempted to write new image block beyond existing image boundaries",
            )
        s_pos = int(img_acc_block.sig_position) + int(pos) * 8
        e_pos = int(img_acc_block.err_position) + int(pos) * 8
        npix_pos = int(img_acc_block.npix_position) + int(pos) * 8

        self.move_to_position(self.file_id_, s_pos)
        do_fwrite(self.file_id_, s_arr, "double")
        self.check_write_error(self.file_id_, "signal")

        self.move_to_position(self.file_id_, e_pos)
        do_fwrite(self.file_id_, e_arr, "double")
        self.check_write_error(self.file_id_, "error")

        self.move_to_position(self.file_id_, npix_pos)
        do_fwrite(self.file_id_, npix_arr, "uint64")
        self.check_write_error(self.file_id_, "npix")
        return self

    def head(self, *args: Any) -> Dict[str, Any]:
        from pyhorace.sqw.dnd_base import DnDBase

        full = False
        for a in args:
            if isinstance(a, str) and a.strip().lower().lstrip("-").startswith("f"):
                full = True
            else:
                raise HoraceError(
                    "HORACE:binfile_v4_common:invalid_argument",
                    f"Unrecognized option {a!r} in head",
                )
        if full:
            data = self.get_dnd("-verbatim", nargout=1)
        else:
            data = self.get_dnd_metadata(nargout=1)
        flds = DnDBase.head_form(full)
        hd: Dict[str, Any] = {}
        for fld in flds:
            hd[fld] = getattr(data, fld)
        return hd

    # ------------------------------------------------------------------
    # Serializable Interface
    # ------------------------------------------------------------------

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> List[str]:
        flds = list(self._v4_fields_to_save)
        if self.sqw_holder is not None:
            flds.append("sqw_holder")
        return flds

    def to_bare_struct(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return Serializable.to_bare_struct(self, *args, **kwargs)

    def from_bare_struct(self, indata: Dict[str, Any]) -> "binfile_v4_common":
        res = Serializable.from_bare_struct(self, indata)
        if isinstance(indata, dict) and "sqw_holder" in indata and indata["sqw_holder"] is not None:
            res.sqw_holder = self._restore_value_from_struct(indata["sqw_holder"])
        return res  # type: ignore[return-value]

