"""Main header class of Horace SQW objects (main_header_cl) and filename parser.

Translated from:
- ``horace_core/sqw/@main_header_cl/main_header_cl.m``
- ``horace_core/sqw/@main_header_cl/private/set_filename_with_cdate_.m``
- ``herbert_core/utilities/files/parse_full_filename.m``
"""

from __future__ import annotations

import copy
import os
import re
import warnings
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.serializable import Serializable, compare_values, register_serializable


class _ReadOnlyPropertyError(AttributeError, HoraceError):
    """Exception raised when attempting to set a read-only property."""


def _fullfile(filepath: str, filename: str) -> str:
    """Join filepath and filename matching MATLAB ``fullfile`` semantics."""
    if not filepath and not filename:
        return ""
    if not filepath:
        return filename
    if not filename:
        return filepath
    return os.path.join(filepath, filename)


def parse_full_filename(full_filename: Any) -> Tuple[str, str]:
    """Transform an input filename into standard ``(filepath, filename)`` form.

    Used by Horace SQW objects to transform an input filename into the standard
    form used by the object. Also can be used to obtain this standard form for
    any input character array or string.

    Parameters
    ----------
    full_filename : str
        String or character array representing the full filename (with path).

    Returns
    -------
    filepath : str
        String representing path to the file.
    filename : str
        String representing name of the input file. If the input filename did
        not have an extension, the input name is modified to have extension
        ``'.sqw'``.
    """
    if isinstance(full_filename, (list, tuple, np.ndarray)):
        if len(full_filename) > 1 and all(
            isinstance(x, (str, np.str_)) for x in full_filename
        ):
            raise HoraceError(
                "HERBERT:utilities:invalid_argument",
                f'This utilities accepts single filename only.\n Provided {len(full_filename)} elements of class: "{type(full_filename).__name__}"',
            )
        raise HoraceError(
            "HERBERT:utilities:invalid_argument",
            f"full_filename should be a string, describing full name of the file on disk.\n It is: {full_filename!r}",
        )

    if not isinstance(full_filename, (str, np.str_)):
        raise HoraceError(
            "HERBERT:utilities:invalid_argument",
            f"full_filename should be a string, describing full name of the file on disk.\n It is: {full_filename!r}",
        )

    s = str(full_filename)
    if len(s) == 0:
        return "", ""

    fp, tail = os.path.split(s)
    fn, fe = os.path.splitext(tail)
    if tail.startswith(".") and fe == "":
        fn = ""
        fe = tail

    if len(fe) == 0:
        fe = ".sqw"
    filename = f"{fn}{fe}"
    return fp, filename


def _set_filename_with_cdate(obj: "main_header_cl", val: Any) -> "main_header_cl":
    """Separate mangled ``filename$creation_date`` string into properties on ``obj``.

    Take filename and file creation date as stored in a Horace binary file
    version 3.xxxx, separate it into parts, and set parts as appropriate
    properties of the ``main_header_cl`` class.

    If filename is mangled with file creation date, the file creation date
    becomes "known".
    """
    if not isinstance(val, (str, np.str_)) or isinstance(val, bool):
        raise HoraceError(
            "HORACE:main_header:invalid_argument",
            f"filename with creatrion data has to be a string Actually it is: {type(val).__name__}",
        )
    s = str(val)
    pos = s.find("$")
    if pos == -1:
        obj._creation_date_defined = False
        obj._filename = s
    else:
        # This also makes file creation date known within creation_date assignment
        obj.creation_date = s[pos + 1 :]
        obj._filename = s[:pos]
    return obj


@register_serializable
class main_header_cl(Serializable):
    """Class describing the main header of a Horace SQW object.

    Notes
    -----
    In addition to containing the file creation information, the class is
    designed to maintain the proper file creation date.
    To set up the file creation date, the actual creation date has to be
    assigned to the ``obj.creation_date`` property. Then the date becomes
    defined and maintained in various class load/save operations.

    Examples
    --------
    >>> obj = main_header_cl()
    >>> now = obj.creation_date
    >>> obj.creation_date = now

    This pattern should be used to set up the file creation date in all
    algorithms intended to create new SQW files.

    Construction
    ------------
    - ``obj = main_header_cl()``
    - ``obj = main_header_cl(filename)``
    - ``obj = main_header_cl(filename, filepath)``
    - ``obj = main_header_cl(filename, filepath, title)``
    - ``obj = main_header_cl(filename, filepath, title, nfiles)``
    - ``obj = main_header_cl(cl_struc)`` where ``cl_struc`` is a dictionary
      containing any set of public properties available to the constructor with
      their corresponding values.
    """

    dt_format: ClassVar[str] = "yyyy-mm-ddTHH:MM:SS"
    _py_dt_format: ClassVar[str] = "%Y-%m-%dT%H:%M:%S"

    # Fields used with serializable interface. Keep order of fields unchanged,
    # as setting creation_date sets also creation_date_defined_ and
    # creation_date_defined_privately sets/reads creation_date_defined_
    # (contrary to usual convention, but necessary for supporting old mat and
    # sqw files, which do not have these properties stored within them).
    fields_to_save_: ClassVar[List[str]] = [
        "full_filename",
        "title",
        "nfiles",
        "creation_date",
        "creation_date_defined_privately",
    ]

    _filename: str = PrivateAttr(default="")
    _filepath: str = PrivateAttr(default="")
    _title: str = PrivateAttr(default="")
    _nfiles: Union[int, float] = PrivateAttr(default=0)
    _creation_date: Union[str, datetime] = PrivateAttr(default="")
    _creation_date_defined: bool = PrivateAttr(default=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct an instance of main_header_cl.

        Calling forms
        -------------
        - ``main_header_cl()``
        - ``main_header_cl(filename)``
        - ``main_header_cl(filename, filepath)``
        - ``main_header_cl(filename, filepath, title)``
        - ``main_header_cl(filename, filepath, title, nfiles)``
        """
        super().__init__()
        self._filename = ""
        self._filepath = ""
        self._title = ""
        self._nfiles = 0
        self._creation_date = ""
        self._creation_date_defined = False

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            arg = args[0]
            if isinstance(arg, main_header_cl):
                self._filename = arg._filename
                self._filepath = arg._filepath
                self._title = arg._title
                self._nfiles = arg._nfiles
                self._creation_date = copy.deepcopy(arg._creation_date)
                self._creation_date_defined = arg._creation_date_defined
                return
            elif isinstance(arg, dict):
                loaded = Serializable.from_struct(arg, self)
                if isinstance(loaded, main_header_cl):
                    self._filename = loaded._filename
                    self._filepath = loaded._filepath
                    self._title = loaded._title
                    self._nfiles = loaded._nfiles
                    self._creation_date = copy.deepcopy(loaded._creation_date)
                    self._creation_date_defined = loaded._creation_date_defined
                if "filename_with_cdate" in arg:
                    self.filename_with_cdate = arg["filename_with_cdate"]
                return
            elif isinstance(arg, (str, np.str_)) and not isinstance(arg, bool):
                self.filename = str(arg)
                return
            else:
                raise HoraceError(
                    "HORACE:main_header:invalid_argument",
                    f"Can not construct main header from parameter: {arg!r}",
                )

        param_names_list = [
            "filename",
            "filepath",
            "title",
            "nfiles",
            "creation_date",
        ]
        _, remains = self.set_positional_and_key_val_arguments(
            param_names_list, False, *args, **kwargs
        )
        if remains:
            raise HoraceError(
                "HORACE:main_header:invalid_argument",
                f"Too many arguments provided on instantiation, excess args: {remains!r}",
            )

    # ------------------------------------------------------------------
    # Public Dependent Properties
    # ------------------------------------------------------------------
    @property
    def title(self) -> str:
        """Title (description) of the SQW object."""
        return self._title

    @title.setter
    def title(self, val: Any) -> None:
        if not isinstance(val, (str, np.str_)) or isinstance(val, bool):
            raise HoraceError(
                "HORACE:main_header:invalid_argument",
                f"Bad title ({val!r}). title must be string or char array.",
            )
        self._title = str(val)

    @property
    def filename(self) -> str:
        """The name of the file where the SQW object is stored."""
        return self._filename

    @filename.setter
    def filename(self, val: Any) -> None:
        if not isinstance(val, (str, np.str_)) or isinstance(val, bool):
            raise HoraceError(
                "HORACE:main_header:invalid_argument",
                f"Bad filename ({val!r}). filename must be string or char array.",
            )
        self._filename = str(val)

    @property
    def filepath(self) -> str:
        """The path where the file has been initially created."""
        return self._filepath

    @filepath.setter
    def filepath(self, val: Any) -> None:
        if not isinstance(val, (str, np.str_)) or isinstance(val, bool):
            raise HoraceError(
                "HORACE:main_header:invalid_argument",
                f"Bad filepath ({val!r}). filepath must be string or char array.",
            )
        self._filepath = str(val)

    @property
    def nfiles(self) -> Union[int, float]:
        """Number of runs contributing to the SQW file this header is responsible for."""
        return self._nfiles

    @nfiles.setter
    def nfiles(self, val: Any) -> None:
        if isinstance(val, np.ndarray):
            if val.size != 1 or not np.issubdtype(val.dtype, np.number) or np.issubdtype(val.dtype, np.bool_):
                raise HoraceError(
                    "HORACE:main_header:invalid_argument",
                    f"Bad nfiles ({val!r}). Number of files must be a non-negative scalar numeric value.",
                )
            val = val.item()
        if (
            not isinstance(val, (int, float, np.number))
            or isinstance(val, bool)
            or np.isnan(val)
            or val < 0
        ):
            raise HoraceError(
                "HORACE:main_header:invalid_argument",
                f"Bad nfiles ({val!r}). Number of files must be a non-negative scalar numeric value.",
            )
        if float(val).is_integer():
            self._nfiles = int(val)
        else:
            self._nfiles = float(val)

    @property
    def creation_date(self) -> str:
        """Retrieve file creation date either from stored value or from system file date."""
        if self._creation_date_defined:
            dt = self._creation_date
        else:
            # Assume that creation date is unknown and will be set as creation
            # date of the file later and explicitly.
            # Return either file date if file exists or actual date if it does not.
            file = _fullfile(self.filepath, self.filename)
            if not file or not os.path.isfile(file):
                dt = datetime.now()
            else:
                mtime = os.path.getmtime(file)
                dt = datetime.fromtimestamp(mtime)
        return self.DT_out_transf_(dt)

    @creation_date.setter
    def creation_date(self, val: Any) -> None:
        """Explicitly set up creation date and make it 'known'."""
        dt = main_header_cl.check_datetime_valid(val)
        self._creation_date = dt
        self._creation_date_defined = True

    @property
    def creation_date_defined(self) -> bool:
        """Return False if the file creation date is not stored together with binary data in old binary files."""
        return self._creation_date_defined

    @creation_date_defined.setter
    def creation_date_defined(self, val: Any) -> None:
        raise _ReadOnlyPropertyError(
            "MATLAB:class:noSetMethod",
            "You cannot set the read-only property 'creation_date_defined' of main_header_cl.",
        )

    # ------------------------------------------------------------------
    # Hidden Dependent Properties
    # ------------------------------------------------------------------
    @property
    def creation_date_defined_privately(self) -> bool:
        """Hidden property allowing save/restore of creation_date_defined used by loadobj/saveobj only."""
        return self._creation_date_defined

    @creation_date_defined_privately.setter
    def creation_date_defined_privately(self, val: Any) -> None:
        self._creation_date_defined = bool(val)

    @property
    def filename_with_cdate(self) -> str:
        """Return filename mangled with creation date for old-style v3 binary Horace files."""
        if self._creation_date_defined:
            return f"{self.filename}${self.creation_date}"
        return self.filename

    @filename_with_cdate.setter
    def filename_with_cdate(self, val: Any) -> None:
        """Separate mangled filename and creation date and set appropriate properties."""
        _set_filename_with_cdate(self, val)

    @property
    def full_filename(self) -> str:
        """Convenience property providing full file path and file name."""
        return _fullfile(self._filepath, self._filename)

    @full_filename.setter
    def full_filename(self, val: Any) -> None:
        flpth, flnm = parse_full_filename(val)
        self._filepath = flpth
        self._filename = flnm

    # ------------------------------------------------------------------
    # Static Utility Methods
    # ------------------------------------------------------------------
    @staticmethod
    def DT_out_transf_(dt: Any) -> str:
        """Transform datetime into the requested string."""
        if not isinstance(dt, datetime):
            warnings.warn(
                "call to class-protected function with invalid argument. Something wrong with class usave",
                UserWarning,
                stacklevel=2,
            )
            dt = datetime.now()
        return dt.strftime(main_header_cl._py_dt_format)

    @staticmethod
    def convert_datetime_from_str(in_str: str) -> datetime:
        """Convert main_header_cl's generated time string into datetime format."""
        if not isinstance(in_str, (str, np.str_)) or isinstance(in_str, bool):
            raise HoraceError(
                "HORACE:main_header:invalid_argument",
                f'Bad creation date: "{in_str!r}".\n File creation date must be either datetime class or a string, convertible to datetime according to format {main_header_cl.dt_format}.',
            )
        s = str(in_str).strip()
        for fmt in (
            main_header_cl._py_dt_format,
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise HoraceError(
            "HORACE:main_header:invalid_argument",
            f'Bad creation date: "{s}".\n File creation date must be either datetime class or a string, convertible to datetime according to format {main_header_cl.dt_format}.',
        )

    @staticmethod
    def convert_datetime_to_str(date_time: Any) -> str:
        """Convert datetime instance into main_header_cl's specific string containing date and time."""
        return main_header_cl.DT_out_transf_(date_time)

    @staticmethod
    def check_datetime_valid(val: Any) -> datetime:
        """Check if the input value represents a valid datetime and convert it to datetime if it does."""
        if isinstance(val, (str, np.str_)) and not isinstance(val, bool):
            return main_header_cl.convert_datetime_from_str(str(val))
        elif isinstance(val, datetime):
            return val
        else:
            raise HoraceError(
                "HORACE:main_header:invalid_argument",
                f'Bad creation date: "{val!r}".\n File creation date must be either datetime class or a string, convertible to datetime according to format {main_header_cl.dt_format}.',
            )

    # ------------------------------------------------------------------
    # Serializable Interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        """Define version of the class to store in mat-files and nxsqw data format."""
        return 2

    def saveable_fields(self) -> Sequence[str]:
        """Return fields used with serializable interface."""
        return list(main_header_cl.fields_to_save_)

    def convert_old_struct(
        self, S: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], "main_header_cl"]:
        """Convert old header structure into the new (version 2) form."""
        S = dict(S)
        if "filepath" in S and isinstance(S["filepath"], str):
            S["filepath"] = re.sub(r"[\\/]$", "", S["filepath"])
        filepath = S.get("filepath", "")
        if "filename" in S:
            S["full_filename"] = _fullfile(filepath, S["filename"])
        elif "filename_with_cdate" in S:
            fparts = str(S["filename_with_cdate"]).split("$")
            S["full_filename"] = _fullfile(filepath, fparts[0])
            if len(fparts) > 1:
                S["creation_date"] = fparts[1]
                S.pop("filename_with_cdate", None)
        return S, self

    def equal_to_tol(
        self,
        other: Any,
        tol: Union[float, Sequence[float]] = (0.0, 0.0),
        nan_equal: bool = True,
        ignore_str: bool = False,
        name_a: str = "obj1",
        name_b: str = "obj2",
    ) -> Tuple[bool, str]:
        """Check equality of two main_header_cl objects within tolerance."""
        if type(self) is not type(other):
            return (
                False,
                f"Objects are of different classes: '{type(self).__name__}' and '{type(other).__name__}'",
            )
        for fld in self.saveable_fields():
            if (
                fld == "creation_date"
                and not self._creation_date_defined
                and not other._creation_date_defined
            ):
                continue
            val1 = getattr(self, fld, None)
            val2 = getattr(other, fld, None)
            iseq, mess = compare_values(
                val1,
                val2,
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=f"{name_a}.{fld}",
                name_b=f"{name_b}.{fld}",
            )
            if not iseq:
                return False, mess
        return True, ""

