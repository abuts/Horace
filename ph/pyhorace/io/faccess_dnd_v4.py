"""DND binary file accessor for Horace format v4."""

from __future__ import annotations

from typing import Any, ClassVar, List, Optional
import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.io.binfile_v4_common import binfile_v4_common
from pyhorace.io.data_block import data_block
from pyhorace.io.dnd_data_block import dnd_data_block


class faccess_dnd_v4(binfile_v4_common):
    """Class to access Horace dnd files written by Horace v4."""

    dnd_blocks_list_: ClassVar[List[data_block]] = [
        data_block("data", "metadata"),
        dnd_data_block(),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data_type(self) -> str:
        return "b+"

    def get_sqw_type(self) -> bool:
        return False

    def get_faccess_version(self) -> float:
        return 4.0

    def get_data_blocks(self) -> List[data_block]:
        return [
            data_block("data", "metadata"),
            dnd_data_block(),
        ]

    def get_blocks_list(self) -> List[data_block]:
        return self.get_data_blocks()

    def get_pix_range(self, *args) -> np.ndarray:
        return np.empty((0, 4), dtype=np.float64)

    def get_sqw(self, *args, **kwargs) -> Any:
        return self.get_dnd(*args, **kwargs)

    def put_sqw(self, *args, **kwargs) -> "faccess_dnd_v4":
        return self.put_dnd(*args, **kwargs)

    def set_file_to_update(
        self, filename: Optional[str] = None, *args: Any
    ) -> "faccess_dnd_v4":
        if filename is None:
            filename = self.full_filename
        obj, _, old_ldr = self._set_file_to_update_impl(filename)
        if getattr(old_ldr, "sqw_type", False) is True:
            obj.fclose()
            raise HoraceError(
                "HORACE:faccess_dnd_v4:invalid_argument",
                f"File {self.full_filename} contains sqw data so you should not use loader for dnd folder to update it",
            )
        return obj

    def get_instrument(self, *args, **kwargs) -> Any:
        from pyhorace.data.instr_classes import IX_null_inst

        return IX_null_inst()

    def get_sample(self, *args, **kwargs) -> Any:
        from pyhorace.data.instr_classes import IX_samp

        mtd = self.get_dnd_metadata(nargout=1)
        return IX_samp(mtd.proj.alatt, mtd.proj.angdeg)

