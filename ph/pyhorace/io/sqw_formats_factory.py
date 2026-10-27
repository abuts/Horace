"""Factory providing appropriate file accessors for Horace sqw and dnd binary files."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from pyhorace.exceptions import HoraceError
from pyhorace.io.faccess_dnd_v2 import faccess_dnd_v2
from pyhorace.io.faccess_dnd_v4 import faccess_dnd_v4
from pyhorace.io.faccess_sqw_prototype import faccess_sqw_prototype
from pyhorace.io.faccess_sqw_v2 import faccess_sqw_v2
from pyhorace.io.faccess_sqw_v3 import faccess_sqw_v3
from pyhorace.io.faccess_sqw_v3_2 import faccess_sqw_v3_2
from pyhorace.io.faccess_sqw_v3_21 import faccess_sqw_v3_21
from pyhorace.io.faccess_sqw_v3_3 import faccess_sqw_v3_3
from pyhorace.io.faccess_sqw_v4 import faccess_sqw_v4
from pyhorace.io.horace_binfile_interface import horace_binfile_interface


class _SqwFormatsFactoryMeta(type):
    """Metaclass providing singleton access via both `.instance` and `.instance()`."""

    _instance: "sqw_formats_factory | None" = None

    @property
    def instance(cls) -> "sqw_formats_factory":
        if cls._instance is None:
            cls._instance = super().__call__()
        return cls._instance


class sqw_formats_factory(metaclass=_SqwFormatsFactoryMeta):
    """Singleton factory providing registered Horace file accessors.

    Handles registration of file format accessors, automatic detection of
    file format versions via ``get_loader``, and selection of preferred
    writers via ``get_pref_access``.
    """

    def __init__(self) -> None:
        self.supported_accessors_: list[horace_binfile_interface] = [
            faccess_sqw_v4(),
            faccess_dnd_v4(),
            faccess_sqw_v3_3(),
            faccess_sqw_v3(),
            faccess_sqw_v3_21(),
            faccess_sqw_v2(),
            faccess_dnd_v2(),
            faccess_sqw_v3_2(),
            faccess_sqw_prototype(),
        ]
        self.preferred_accessor_num_: int = 0
        self.written_types_: list[str] = [
            "DnDBase",
            "sqw",
            "sqw2",
            "dnd",
            "d0d",
            "d1d",
            "d2d",
            "d3d",
            "d4d",
        ]
        self.types_map_: dict[str, int] = dict(
            zip(self.written_types_, [1, 0, 0, 1, 1, 1, 1, 1, 1])
        )

    def __call__(self) -> "sqw_formats_factory":
        """Allow both ``sqw_formats_factory.instance`` and ``sqw_formats_factory.instance()``."""
        return self

    @property
    def supported_accessors(self) -> list[horace_binfile_interface]:
        """Return fresh copies of all registered accessors."""
        return [type(acc)() for acc in self.supported_accessors_]

    def get_loader(
        self, sqw_file_name: Any, *args: Any
    ) -> horace_binfile_interface | list[horace_binfile_interface]:
        """Return an initialized loader appropriate to read the given file(s)."""
        if isinstance(sqw_file_name, (list, tuple)):
            return [self.get_loader(fname, *args) for fname in sqw_file_name]  # type: ignore[misc]

        if isinstance(sqw_file_name, (int, float, complex, bool, np.number)) or not isinstance(
            sqw_file_name, (str, os.PathLike)
        ):
            raise HoraceError("HORACE:file_io:runtime_error", "filename was not numeric")

        fname_str = os.fspath(sqw_file_name)
        if not os.path.isfile(fname_str):
            raise HoraceError(
                "HORACE:file_io:runtime_error",
                f"get_loader: File {fname_str} does not exist",
            )
        full_data_name = os.path.abspath(fname_str)

        head_struc, fh = horace_binfile_interface.get_file_header(full_data_name, *args)

        for acc in self.supported_accessors_:
            ok, objinit = acc.should_load_stream(head_struc, fh)
            if ok:
                loader = type(acc)()
                try:
                    loader = loader.init(objinit, *args)
                except Exception as err:
                    try:
                        fh.close()
                    except Exception:
                        pass
                    raise HoraceError(
                        "HORACE:file_io:runtime_error",
                        f"get_loader: Error: {err} initializing file: {fname_str}",
                    ) from err
                return loader

        try:
            fh.close()
        except Exception:
            pass

        if head_struc.get("name", "") == "horace":
            raise HoraceError(
                "HORACE:file_io:runtime_error",
                f"horace file version {head_struc.get('version')} for file {fname_str} is not currently supported.",
            )
        else:
            raise HoraceError(
                "HORACE:file_io:runtime_error",
                f" Can not interpret format of the file {fname_str} as a Horace file format.",
            )

    def get_pref_access(self, *args: Any) -> horace_binfile_interface:
        """Return the preferred uninitialized accessor for the given object or type."""
        if len(args) == 0:
            return type(self.supported_accessors_[self.preferred_accessor_num_])()

        var = args[0]
        if isinstance(var, str):
            the_type = var
        else:
            the_type, orig_type = self.get_sqw_type(var)
            if the_type == "none":
                the_type = orig_type

        if the_type in self.types_map_:
            ld_num = self.types_map_[the_type]
            return type(self.supported_accessors_[ld_num])()

        raise HoraceError(
            "HORACE:file_io:invalid_argument",
            f"Save called for unregistered Object of type '{the_type}'",
        )

    def set_pref_access(self, type_or_obj: Any, facc_name: Any = "") -> None:
        """Set preferred accessor for a given type (primarily for testing)."""
        if not isinstance(type_or_obj, str):
            type_name = type(type_or_obj).__name__
        else:
            type_name = type_or_obj

        if not isinstance(facc_name, str):
            facc_name_str = type(facc_name).__name__
        else:
            facc_name_str = facc_name

        if facc_name_str == "":
            self.types_map_ = dict(
                zip(self.written_types_, [1, 0, 0, 1, 1, 1, 1, 1, 1])
            )
            self.preferred_accessor_num_ = 0
            return

        if type_name not in self.written_types_:
            raise HoraceError(
                "HORACE:sqw_fromats_factory:invalid_argument",
                f"Registered keys to set factory format are 'sqw' or 'dnd' datasets. "
                f"Requested to set format for {type_name}",
            )

        acc_names = [type(x).__name__ for x in self.supported_accessors_]
        if facc_name_str not in acc_names:
            raise HoraceError(
                "HORACE:sqw_fromats_factory:invalid_argument",
                f"Only registered formatters names could be set up as preferential accessors. "
                f"Try to set: {facc_name_str}. Allowed formatters: {acc_names}",
            )

        facc_idx = acc_names.index(facc_name_str)
        self.preferred_accessor_num_ = facc_idx
        self.types_map_[type_name] = facc_idx

    def check_compatibility(
        self, obj1: horace_binfile_interface, obj2: horace_binfile_interface
    ) -> bool:
        """Return True if *obj1* can be upgraded into *obj2*."""
        if isinstance(obj2, type(obj1)):
            return True
        if isinstance(obj1, (faccess_sqw_v3, faccess_sqw_v3_3)) and isinstance(
            obj2, faccess_sqw_v3_2
        ):
            return False
        return True

    @staticmethod
    def get_sqw_type(obj: Any) -> tuple[str, str]:
        """Classify *obj* as ``'sqw'``, ``'dnd'``, or ``'none'`` alongside its class name."""
        orig_type = type(obj).__name__

        from pyhorace.dnd.dnd_base import DnDBase
        from pyhorace.sqw.sqw_dnd_base import SQWDnDBase

        if isinstance(obj, SQWDnDBase):
            if isinstance(obj, DnDBase):
                return ("dnd", orig_type)
            else:
                return ("sqw", orig_type)

        if orig_type in ("DnDBase", "d0d", "d1d", "d2d", "d3d", "d4d"):
            return ("dnd", orig_type)
        if orig_type in ("sqw", "sqw2"):
            return ("sqw", orig_type)

        if isinstance(obj, dict):
            fields = list(obj.keys())
            sqw_fields = ["main_header", "header", "detpar", "data", "pix"]
            dnd_fields = [
                "filename",
                "filepath",
                "title",
                "alatt",
                "angdeg",
                "uoffset",
                "u_to_rlu",
                "ulen",
                "label",
                "iax",
                "iint",
                "pax",
                "p",
                "dax",
                "s",
                "e",
                "npix",
            ]
            if len(fields) == len(sqw_fields) and all(f in sqw_fields for f in fields):
                return ("sqw", "struct")
            if len(fields) == len(dnd_fields) and all(f in dnd_fields for f in fields):
                return ("dnd", "struct")
            return ("none", "struct")

        return ("none", orig_type)

