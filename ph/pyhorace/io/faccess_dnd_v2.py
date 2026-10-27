"""DND binary file accessor for Horace format v2."""

from __future__ import annotations

from pyhorace.io.binfile_v2_common import binfile_v2_common


class faccess_dnd_v2(binfile_v2_common):
    """Class to access Horace dnd files written in binary format v1-v2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_faccess_version(self) -> float:
        return 2.0

