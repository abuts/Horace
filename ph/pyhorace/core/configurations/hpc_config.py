"""High-performance computing configuration for PyHorace.

Translated from `horace_core/configuration/@hpc_config/hpc_config.m`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np

from .config_base import config_base
from .config_store import (
    HoraceConfigError,
    _sys_memory,
    config_store,
    warn_with_id,
)
from .parallel_config import parallel_config


class hpc_config(config_base):
    """Class responsible for settings related to the algorithms which may
    run in parallel. Defines which parallel algorithms should run in
    parallel and what parallel options to use to achieve the best
    performance on a given machine.

    The settings optimal for a given machine (or rather given machine
    configuration) should be enabled/disabled running `hpc('on')`/`hpc('off')`
    helper method.

    To see the list of current configuration option values:
        >>> hpc_config()

    To set values:
        >>> hc = hpc_config()
        >>> hc.name1 = val1
    or:
        >>> set(hpc_config, 'name1', val1, 'name2', val2, ...)

    To fetch values:
        >>> val1 = hpc_config().name1
    or:
        >>> val1, val2, ... = get(hpc_config, 'name1', 'name2', ...)

    `hpc_config` properties & methods are:
    --------------------------------------
    - `build_sqw_in_parallel`   - If True, use parallel framework to generate tmp files
                                  and do other computationally expensive tasks
                                  benefiting from parallelization.
    - `parallel_workers_number` - Number of sessions (MPI workers) to launch to
                                  run parallel code.
    - `combine_sqw_using`       - What type of sub-algorithm to use for combining
                                  sqw/tmp files together.
    - `combine_sqw_options`     - Helper property providing options available for
                                  `combine_sqw_using` property (`'matlab'`,
                                  `'mex_code'`, `'mpi_code'`).
    - `mex_combine_thread_mode` - Various thread modes deployed when combining sqw
                                  files using mex code.
    - `mex_combine_buffer_size` - Size of buffer used by mex code while combining
                                  files per each contributing file.
    - `parallel_multifit`       - Enable or disable computation of fitting in parallel.
    - `parallel_cluster`        - What parallel cluster type to use to perform
                                  parallel tasks (`'herbert'`, `'parpool'`,
                                  `'mpiexec_mpi'`, or `'slurm_mpi'`).
    - `phys_mem_available`      - Return real size (in bytes) of physical memory
                                  available to use by algorithms.
    - `sort_pix_in_binary_op`   - Pixels are usually randomly distributed within an
                                  image cell. To perform correct binary operations
                                  between two sqw files with pixel data, pixels within
                                  each image cell have to be sorted, which is a
                                  relatively expensive operation. If you are sure that
                                  pixels in the operation come from the same data, this
                                  property may be set to False. Not saveable in
                                  configuration files (memory-only).
    """

    _SAVED_PROPERTIES: List[str] = [
        "build_sqw_in_parallel",
        "combine_sqw_using",
        "mex_combine_thread_mode",
        "mex_combine_buffer_size",
        "parallel_multifit",
    ]
    _COMBINE_SQW_OPTIONS: List[str] = ["matlab", "mex_code", "mpi_code"]
    DEFAULT_PIX_SIZE: int = 36  # 9 floats * 4 bytes

    def __init__(self) -> None:
        super().__init__("hpc_config")
        self.build_sqw_in_parallel_: bool = False
        self.parallel_multifit_: bool = False
        self.combine_sqw_using_: str = "matlab"
        self.mex_combine_thread_mode_: int = 0
        self.mex_combine_buffer_size_: int = 1024 * 64
        self.sort_pix_in_binary_op_: bool = True
        self.mem_only_prop_list_ = ["sort_pix_in_binary_op", "phys_mem_available"]

    @staticmethod
    def calc_free_memory() -> int:
        """Function tries to estimate actual free memory available to Horace.
        Result is expressed in bytes.
        """
        free_mem, phys_mem = _sys_memory()
        mem = int(0.9 * free_mem) if free_mem > 0 else int(phys_mem)
        return max(1, mem)

    # ------------------------------------------------------------------
    # Getters & Setters
    # ------------------------------------------------------------------

    @property
    def combine_sqw_using(self) -> str:
        """Set-up algorithm to use for combining multiple sqw(tmp) files."""
        return str(self.get_or_restore_field("combine_sqw_using"))

    @combine_sqw_using.setter
    def combine_sqw_using(self, val: str) -> None:
        if not isinstance(val, str):
            raise HoraceConfigError(
                "HORACE:hpc_config:invalid_argument",
                f"combine_sqw_using must be a string, got {type(val).__name__}",
            )
        matches = [
            o for o in self._COMBINE_SQW_OPTIONS if o.lower().startswith(val.lower())
        ]
        if len(matches) != 1:
            raise HoraceConfigError(
                "HORACE:hpc_config:invalid_argument",
                f"Unrecognized option: {val}. Only 'matlab','mex_code' or 'mpi_code' can be used",
            )
        config_store.instance().store_config(self, "combine_sqw_using", matches[0])

    @property
    def combine_sqw_options(self) -> List[str]:
        """Helper read-only property displaying possible codes to use to
        combine sqw (`combine_sqw_using`) available options, namely:
        - `'matlab'`: uses initial code to combine multiple tmp files. Slowest
          but most reliable method, enabled by default.
        - `'mex_code'`: uses multi-threaded compiled C++ mex code to combine
          multiple files.
        - `'mpi_code'`: can be enabled if parallel computing is present and
          system supports MPI.
        To select one of the options above, one can provide only first
        distinctive prefix for any option (e.g. `'ma'`, `'me'`, or `'mp'`).
        """
        return list(self._COMBINE_SQW_OPTIONS)

    @property
    def use_mex_for_combine(self) -> bool:
        """Deprecated property: use multi-threaded mex code to combine various
        sqw/tmp files together. Replaced by `combine_sqw_using`.
        """
        return self.combine_sqw_using.lower() == "mex_code"

    @use_mex_for_combine.setter
    def use_mex_for_combine(self, val: Union[bool, int]) -> None:
        mode = "mex_code" if bool(val) else "matlab"
        config_store.instance().store_config(self, "combine_sqw_using", mode)

    @property
    def mex_combine_buffer_size(self) -> int:
        """Size of buffer used by mex code while combining files per each file."""
        return int(self.get_or_restore_field("mex_combine_buffer_size"))

    @mex_combine_buffer_size.setter
    def mex_combine_buffer_size(self, val: int) -> None:
        if val == 0:
            self.use_mex_for_combine = False
            return
        if val < 64:
            raise HoraceConfigError(
                "HORACE:hpc_config:invalid_argument",
                " mex_combine_buffer_size should be bigger then 64, and better >1024",
            )
        config_store.instance().store_config(self, "mex_combine_buffer_size", int(val))

    @property
    def mex_combine_thread_mode(self) -> int:
        """If mex code is used for combining tmp files, various thread modes
        can be deployed for this operation:
        - `0` - one thread reads all tmp files and another one writes combined
          information into the target file.
        - `1` - one thread writes combined sqw file and two threads are launched
          for each contributing file to read necessary information (combination
          of modes 2 and 3).
        - `2` - a thread per contributing file is launched to read bin information
          while a common thread for all contributing files is used to read pixel
          information.
        - `3` - a thread is launched per contributing file to read pixel
          information while a common thread is used to read bin information.
        """
        return int(self.get_or_restore_field("mex_combine_thread_mode"))

    @mex_combine_thread_mode.setter
    def mex_combine_thread_mode(self, val: int) -> None:
        if val < 0 or val > 3:
            raise HoraceConfigError(
                "HORACE:hpc_config:invalid_argument",
                " mex_combine_multithreaded should be a number in the range fromn 0 to 3\n "
                "  meaning:\n"
                " 0 -- minor multithreading "
                " 1 -- full multithreading"
                " and two debug options:\n"
                " 2 -- only bin numbers are read by separate thread"
                " 3 -- only pixels are read by separate thread",
            )
        config_store.instance().store_config(self, "mex_combine_thread_mode", int(val))

    @property
    def build_sqw_in_parallel(self) -> bool:
        """If True, launch separate session(s) or parallel job to generate tmp files."""
        return bool(self.get_or_restore_field("build_sqw_in_parallel"))

    @build_sqw_in_parallel.setter
    def build_sqw_in_parallel(self, val: Union[bool, int]) -> None:
        accum = bool(val)
        if accum:
            pc = parallel_config()
            if not pc._is_worker_available(pc.worker):
                warn_with_id(
                    "HORACE:hpc_config:invalid_argument",
                    "Can not start accumulating in separate process. Can not find worker on a data search path; "
                    "See: https://pace-neutrons.github.io/Horace/unstable/manual/Parallel.html#running-horace-in-parallel "
                    "for the details on how to set it up",
                )
                accum = False
        config_store.instance().store_config(self, "build_sqw_in_parallel", accum)

    @property
    def accum_in_separate_process(self) -> bool:
        """Deprecated alias for `build_sqw_in_parallel`."""
        return self.build_sqw_in_parallel

    @accum_in_separate_process.setter
    def accum_in_separate_process(self, val: Union[bool, int]) -> None:
        self.build_sqw_in_parallel = val

    @property
    def parallel_multifit(self) -> bool:
        """Enable or disable computation of fitting in parallel."""
        return bool(self.get_or_restore_field("parallel_multifit"))

    @parallel_multifit.setter
    def parallel_multifit(self, val: Union[bool, int]) -> None:
        p_mf = bool(val)
        if p_mf:
            pc = parallel_config()
            if not pc._is_worker_available(pc.worker):
                warn_with_id(
                    "HPC_CONFIG:invalid_argument",
                    " Can not start accumulating in separate process as: worker not found",
                )
                p_mf = False
        config_store.instance().store_config(self, "parallel_multifit", p_mf)

    @property
    def sort_pix_in_binary_op(self) -> bool:
        """Pixels are usually randomly distributed within an image cell.
        To perform correct binary operations between two sqw files with
        pixel data, pixels within each image cell have to be sorted,
        which is a relatively expensive operation. If you are sure that
        pixels in operation come from the same data (e.g. calculating
        foreground and background out of the same dataset and adding them
        together), this property may be set to False.
        The property is not saveable in configuration files so it returns
        to default each time memory is cleared up.
        """
        return bool(self.get_or_restore_field("sort_pix_in_binary_op"))

    @sort_pix_in_binary_op.setter
    def sort_pix_in_binary_op(self, val: Union[bool, int]) -> None:
        config_store.instance().store_config(self, "sort_pix_in_binary_op", bool(val))

    @property
    def parallel_workers_number(self) -> int:
        """Number of workers to deploy in parallel jobs."""
        return int(
            config_store.instance().get_value(
                "parallel_config", "parallel_workers_number"
            )
        )

    @parallel_workers_number.setter
    def parallel_workers_number(self, val: int) -> None:
        pc = parallel_config()
        pc.parallel_workers_number = val

    @property
    def parallel_cluster(self) -> str:
        """What parallel framework to use for parallel tasks. Defined in
        `parallel_config` and exposed here for clarity.
        """
        return str(
            config_store.instance().get_value("parallel_config", "parallel_cluster")
        )

    @parallel_cluster.setter
    def parallel_cluster(self, val: str) -> None:
        pc = parallel_config()
        pc.parallel_cluster = val

    @property
    def remote_folder(self) -> str:
        """Exposes the folder used by `parallel_config` for storing/reading job data."""
        return str(
            config_store.instance().get_value(
                "parallel_config", "shared_folder_on_remote"
            )
        )

    @remote_folder.setter
    def remote_folder(self, val: str) -> None:
        pc = parallel_config()
        pc.remote_folder = val

    @property
    def parallel_config(self) -> parallel_config:
        """Immutable reference to the class which describes the parallel
        configuration. To change the parallel configuration, work with
        the `parallel_config` class itself.
        """
        return parallel_config()

    @property
    def hpc_options(self) -> List[str]:
        """Helper read-only property returning list of options which define
        hpc configuration.
        """
        return list(self._SAVED_PROPERTIES)

    @property
    def phys_mem_available(self) -> int:
        """Return real size (in bytes) of physical memory available to use
        by algorithms. Tries to estimate free memory at first call and stores
        result for future usage, so first call should be tried on a clean system.
        """
        return self._get_phys_mem_available()

    @phys_mem_available.setter
    def phys_mem_available(
        self, val: Optional[Union[int, float, Sequence[Any]]]
    ) -> None:
        self._set_phys_mem_available(val, True)

    def _get_phys_mem_available(self) -> int:
        if self.is_field_configured("phys_mem_available"):
            mem = config_store.instance().get_value(self, "phys_mem_available")
            if isinstance(mem, (int, float, np.number)) and mem > 0:
                return int(mem)
        return self._set_phys_mem_available(None, False)

    def _set_phys_mem_available(
        self,
        mem: Optional[Union[int, float, Sequence[Any]]],
        warn_on_settings: bool,
    ) -> int:
        if mem is None or (
            isinstance(mem, (list, tuple, np.ndarray)) and len(mem) == 0
        ):
            mem = self.calc_free_memory()
            warn_on_settings = False

        if not isinstance(mem, (int, float, np.number)) or mem <= 0:
            raise HoraceConfigError(
                "HORACE:hpc_config:invalid_argument",
                "Physical memory available should be a value larger than 0",
            )
        mem_int = int(mem)
        mchs, fbs = config_store.instance().get_value(
            "hor_config", "mem_chunk_size", "fb_scale_factor"
        )
        def_size = int(mchs) * int(fbs) * self.DEFAULT_PIX_SIZE
        enable_warning = not self.disable_setup_warnings
        if mem_int < def_size and enable_warning:
            warn_with_id(
                "HORACE:insufficient_physical_memory",
                f"Estimated free physical memory ({mem_int // (1024 * 1024)}MB) is smaller than "
                f"size of default memory-based sqw object ({def_size // (1024 * 1024)}MB)",
            )
        if warn_on_settings and enable_warning:
            warn_with_id(
                "HORACE:physical_memory_configured",
                "You have specified the value for free physical memory.",
            )
        config_store.instance().store_config(self, "phys_mem_available", mem_int)
        return mem_int

    def get_storage_field_names(self) -> List[str]:
        """Helper function returns the list of the public names of the fields
        which should be saved.
        """
        return list(self._SAVED_PROPERTIES)

    def get_default_value(self, field_name: str) -> Any:
        """Method gets internal field value bypassing standard get/set
        methods interface.
        """
        if field_name == "phys_mem_available":
            return self._get_phys_mem_available()
        return getattr(self, f"{field_name}_")


__all__ = ["hpc_config"]
