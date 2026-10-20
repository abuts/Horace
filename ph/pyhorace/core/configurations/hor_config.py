"""Main Horace configuration class for PyHorace.

Translated from `horace_core/configuration/@hor_config/hor_config.m`.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from typing import Any, List, Optional, Union

import numpy as np

from .config_base import config_base
from .config_store import (
    HoraceConfigError,
    _sys_memory,
    config_store,
    warn_with_id,
)
from .hpc_config import hpc_config
from .parallel_config import parallel_config


class hor_config(config_base):
    """Create the Horace configuration that sets memory options and some other defaults.

    To see the list of current configuration option values:
        >>> hor_config()

    To set values:
        >>> hc = hor_config()
        >>> hc.name1 = val1
    or:
        >>> set(hor_config, 'name1', val1, 'name2', val2, ...)

    To fetch values:
        >>> val1 = hor_config().name1
    or:
        >>> val1, val2, ... = get(hor_config, 'name1', 'name2', ...)

    `hor_config` properties are:
    ----------------------------
    - `mem_chunk_size`       - Maximum length of buffer array to accumulate pixels
                               from an input file.
    - `fb_scale_factor`      - The product of `fb_scale_factor` and `mem_chunk_size`
                               defines maximal number of pixels to put in memory.
                               If number of pixels in sqw object exceeds this product,
                               sqw object by default becomes filebacked.
    - `ignore_nan`           - Ignore NaN data when making cuts.
    - `ignore_inf`           - Ignore Inf data when making cuts.
    - `log_level`            - Set verbosity of informational output.
    - `use_mex`              - Use mex/C++ extensions for time-consuming operations,
                               if available.
    - `delete_tmp`           - Automatically delete temporary files after generating
                               sqw files.
    - `working_directory`    - The folder to write tmp files.
    - `store_src_in_plots`   - If True, each figure stores in `fig.UserData` field
                               list of objects the figure displays.
    - `hpc_config`           - Helper/compatibility property to access high performance
                               computing settings. Use `hpc_config` to modify hpc
                               settings themselves.
    - `parallel_config`      - Helper/compatibility property to access parallel
                               computing settings. Use `parallel_config` to modify
                               parallel computing settings themselves.
    - `force_mex_if_use_mex` - Fail if mex cannot be used. Used in mex files debugging.
    - `init_tests`           - Enable Horace specific unit test functions and access
                               to Horace unit tests folders. Works for Horace
                               downloaded as github repository only.
    """

    FILE_PIX_SIZE: int = 36

    _SAVED_PROPERTIES: List[str] = [
        "mem_chunk_size",
        "fb_scale_factor",
        "ignore_nan",
        "ignore_inf",
        "use_mex",
        "delete_tmp",
        "store_src_in_plots",
        "force_mex_if_use_mex",
        "log_level",
        "init_tests",
    ]

    def __init__(self) -> None:
        super().__init__("hor_config")
        self.mem_chunk_size_: int = int(1e7)
        self.fb_scale_factor_: int = 10
        self.ignore_nan_: bool = True
        self.ignore_inf_: bool = False
        self.use_mex_: bool = True
        self.delete_tmp_: bool = True
        self.force_mex_if_use_mex_: bool = False
        self.log_level_: int = 1
        self.store_src_in_plots_: bool = True
        self.init_tests_: bool = False
        self.spe_file_en_transf_field_width_: int = 10

    # ------------------------------------------------------------------
    # Getters & Setters
    # ------------------------------------------------------------------

    @property
    def mem_chunk_size(self) -> int:
        """Maximum number of pixels that are processed at one go during cuts.
        On a usual machine with 16GB of RAM it is `10^7` (higher value does
        not provide obvious performance benefits) but on older machines
        with ~4GB it has to be reduced to `10^6`.
        """
        return int(self.get_or_restore_field("mem_chunk_size"))

    @mem_chunk_size.setter
    def mem_chunk_size(self, val: Union[int, float]) -> None:
        val_int = int(math.ceil(val))
        if val_int < 1000:
            warn_with_id(
                "HOR_CONFIG:set_mem_chunk_size",
                " mem chunk size should not be too small at least 1M is recommended",
            )
        free_mem, _ = _sys_memory()
        if val_int * self.FILE_PIX_SIZE >= free_mem / 3:
            if val_int <= free_mem * 0.8:
                warn_with_id(
                    "HORACE:invalid_argument",
                    "Buffer chunk size exceeds 1/3 of available physical memory. HORACE may got unstable, trying to use such chunk in calculations",
                )
            else:
                raise HoraceConfigError(
                    "HORACE:hor_config:invalid_argument",
                    f"attempt to set up mem_chunk_size exceeding 0.8 of total physical memory available (Evaluated to: {free_mem // 1024}kB). This would not work",
                )
        config_store.instance().store_config(self, "mem_chunk_size", val_int)

    @property
    def fb_scale_factor(self) -> float:
        """Factor for determining whether to do certain operations as
        file-backed or memory-backed. The value is roughly equivalent to the
        number of "pages" which would need to be produced before considering
        using file-backed algorithms (slower, but work for any sized object),
        i.e. if `num_pixels > mem_chunk_size * fb_scale_factor` => do file-backed.
        """
        return float(self.get_or_restore_field("fb_scale_factor"))

    @fb_scale_factor.setter
    def fb_scale_factor(self, val: Union[int, float]) -> None:
        if val < 3:
            warn_with_id(
                "HOR_CONFIG:set_fb_scale_factor",
                f" fb_scale_factor in general should not be less than 3.\nSetting it to {val} may cause performance degradation",
            )
        config_store.instance().store_config(self, "fb_scale_factor", val)

    @property
    def pixel_page_size(self) -> int:
        """Maximum memory size of pixel data array in file-backed algorithms
        (units of bytes).
        """
        return int(self.mem_chunk_size * self.FILE_PIX_SIZE)

    @property
    def threads(self) -> int:
        """Number of computational threads to use (delegates to `parallel_config`)."""
        return parallel_config().threads

    @threads.setter
    def threads(self, val: int) -> None:
        pc = parallel_config()
        pc.threads = val

    @property
    def ignore_nan(self) -> bool:
        """Ignore NaN values if pixels have them (default: True)."""
        return bool(self.get_or_restore_field("ignore_nan"))

    @ignore_nan.setter
    def ignore_nan(self, val: Union[bool, int]) -> None:
        config_store.instance().store_config(
            self,
            "ignore_nan",
            bool(val > 0 if isinstance(val, (int, float)) else val),
        )

    @property
    def ignore_inf(self) -> bool:
        """Ignore Inf values if pixels have them (default: False)."""
        return bool(self.get_or_restore_field("ignore_inf"))

    @ignore_inf.setter
    def ignore_inf(self, val: Union[bool, int]) -> None:
        config_store.instance().store_config(
            self,
            "ignore_inf",
            bool(val > 0 if isinstance(val, (int, float)) else val),
        )

    @property
    def log_level(self) -> Union[int, float]:
        """The verbosity of the log messages.
        The larger the value, the more information is printed, e.g.:
        - `-1`: No information messages printed
        - `0`: Major information messages printed
        - `1`: Minor information messages printed in addition
        - `2`: Time of the run measured and printed as well.
        """
        return self.get_or_restore_field("log_level")

    @log_level.setter
    def log_level(self, val: Union[int, float]) -> None:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise HoraceConfigError(
                "HOR_CONFIG:set_log_level", " log level must be a number"
            )
        config_store.instance().store_config(self, "log_level", val)

    @property
    def horace_info_level(self) -> Union[int, float]:
        """Old implementation of `log_level` property indicating how verbose
        Horace would be. See `log_level` for more details.
        """
        return self.log_level

    @horace_info_level.setter
    def horace_info_level(self, val: Union[int, float]) -> None:
        self.log_level = val

    @property
    def use_mex(self) -> bool:
        """Use mex/C++ code for time-consuming operations (default: True if
        mex/C++ files are compiled).
        """
        return bool(self.get_or_restore_field("use_mex"))

    @use_mex.setter
    def use_mex(self, val: Union[bool, int]) -> None:
        use = bool(val > 0 if isinstance(val, (int, float)) else val)
        config_store.instance().store_config(self, "use_mex", use)

    @property
    def store_src_in_plots(self) -> bool:
        """If True, plotted figures contain in their `fig.UserData` field list
        of objects the figure displays.
        """
        return bool(self.get_or_restore_field("store_src_in_plots"))

    @store_src_in_plots.setter
    def store_src_in_plots(self, val: Union[bool, int]) -> None:
        do = bool(val > 0 if isinstance(val, (int, float)) else val)
        config_store.instance().store_config(self, "store_src_in_plots", do)

    @property
    def force_mex_if_use_mex(self) -> bool:
        """Testing and debugging option -- fail if mex cannot be used.
        By default if mex file fails, program tries to use fallback code, but
        if this option is set to True, the whole operation may fail.
        """
        return bool(self.get_or_restore_field("force_mex_if_use_mex"))

    @force_mex_if_use_mex.setter
    def force_mex_if_use_mex(self, val: Union[bool, int]) -> None:
        use = bool(val > 0 if isinstance(val, (int, float)) else val)
        config_store.instance().store_config(self, "force_mex_if_use_mex", use)

    @property
    def delete_tmp(self) -> bool:
        """Automatically delete temporary files after generating sqw files.
        By default it is True, but you may set it to False to keep files for
        later operations.
        """
        return bool(self.get_or_restore_field("delete_tmp"))

    @delete_tmp.setter
    def delete_tmp(self, val: Union[bool, int]) -> None:
        del_tmp = bool(val > 0 if isinstance(val, (int, float)) else val)
        config_store.instance().store_config(self, "delete_tmp", del_tmp)

    @property
    def working_directory(self) -> str:
        """The folder where tmp files should be stored.
        By default `gen_sqw` sets this value to the place where spe files are
        located. If you never did `gen_sqw` on a given machine, system tmp
        directory is used.
        Change this value to point to a fast & large disk or to a parallel
        file system. Assign empty value to restore it to default (system tmp
        directory).
        """
        work_dir = config_store.instance().get_config_field(
            "parallel_config", "working_directory"
        )
        if not work_dir:
            return tempfile.gettempdir()
        return str(work_dir)

    @working_directory.setter
    def working_directory(self, val: Optional[str]) -> None:
        pc = parallel_config()
        pc.working_directory = val

    @property
    def wkdir_is_default(self) -> bool:
        """Return True if working directory has not been set and refers
        to default (system tmp) directory.
        """
        work_dir = config_store.instance().get_config_field(
            "parallel_config", "working_directory"
        )
        return not bool(work_dir)

    @property
    def init_tests(self) -> bool:
        """Enable Horace unit test framework and add unit test folders to
        search path (option for Horace testing). Works for Horace retrieved
        from repository only.
        """
        return bool(self.get_or_restore_field("init_tests"))

    @init_tests.setter
    def init_tests(self, val: Union[bool, int]) -> None:
        init = bool(val > 0 if isinstance(val, (int, float)) else val)
        path = self._process_unit_test_path(init, set_path=True)
        if (init and path) or not init:
            config_store.instance().store_config(self, "init_tests", init)

    @property
    def unit_test_folder(self) -> Optional[str]:
        """Location of the folder with unit tests."""
        if self.init_tests:
            return self._process_unit_test_path(True, set_path=False)
        return None

    def set_unit_test_path(self) -> "hor_config":
        """Add unit test path to search path."""
        self._process_unit_test_path(True, set_path=True)
        return self

    @staticmethod
    def _process_unit_test_path(init: bool, set_path: bool = False) -> str:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        tests_path = os.path.join(repo_root, "tests")
        if not os.path.isdir(tests_path):
            if init:
                warn_with_id(
                    "HERBERT_INIT:invalid_setup",
                    f"Can not set-up access to the unit tests as no unit tests at {tests_path} are available",
                )
            return ""
        if set_path:
            if init:
                if tests_path not in sys.path:
                    sys.path.insert(0, tests_path)
            else:
                while tests_path in sys.path:
                    sys.path.remove(tests_path)
                return ""
        return tests_path

    @property
    def hpc_config(self) -> hpc_config:
        """Property related to high performance computing settings.
        Provided for information only while changes to this property should
        be made through `hpc_config` class setters directly.
        """
        return hpc_config()

    @property
    def parallel_config(self) -> parallel_config:
        """Property exposing access to Horace parallel computing settings.
        Provided for information only while changes to this property should
        be made through `parallel_config` class setters directly.
        """
        return parallel_config()

    def get_storage_field_names(self) -> List[str]:
        """Helper function returns the list of the public names of the fields
        which should be saved.
        """
        return list(self._SAVED_PROPERTIES)

    def get_default_value(self, field_name: str) -> Any:
        """Method gets internal field value bypassing standard get/set
        methods interface. Relies on assumption that each public field has a
        private field with name different by underscore.
        """
        return getattr(self, f"{field_name}_")


__all__ = ["hor_config"]
