"""Parallel & cluster configuration for PyHorace.

Translated from `herbert_core/classes/MPIFramework/@parallel_config/parallel_config.m`.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .config_base import config_base
from .config_store import (
    ConfigData,
    HoraceConfigError,
    config_store,
    warn_with_id,
)


class parallel_config(config_base):
    """The config class contains the information about the parallel cluster
    and the parallel cluster available to run parallel Horace/Herbert jobs.

    To see the list of current configuration option values type:
        >>> parallel_config()

    To set values:
        >>> pc = parallel_config()
        >>> pc.name1 = val1
    or just:
        >>> set(parallel_config, 'name1', val1, 'name2', val2, ...)

    To fetch values:
        >>> pc = parallel_config()
        >>> val1 = pc.name1
    or just:
        >>> val1, val2, ... = get(parallel_config, 'name1', 'name2', ...)

    `parallel_config` properties & methods:
    ---------------------------------------------------------------------
    - `worker`             - The name of the script or program to run on cluster
                             in parallel using parallel workers.
    - `is_compiled`        - False if the worker is a MATLAB/Python script and
                             True if this script is a compiled application.
    - `parallel_cluster`   - The name of a cluster to use. Currently defined are
                             `h[erbert]`, `p[arpool]`, `[m]piexec_mpi` and
                             `[s]lurm_mpi` clusters, or `none`.
    - `cluster_config`     - The configuration describing the parallel cluster
                             running the selected cluster.
    - `parallel_workers_number` - Number of workers to deploy in parallel jobs.
    - `threads`            - How many computational threads to use in parallel
                             and in MEX/C++ OMP.
    - `min_npix_for_omp_cut` - Running OMP on cuts with very small number of
                             pixels is inefficient. If number of pixels in
                             source object is smaller than specified, code
                             reverts to single-threaded execution. Setting it to
                             0 or negative runs with OMP where available.
    - `dynamic_omp_npixels_stride` - Set this value to negative to disable dynamic
                             scheduling OMP pixels cuts.
    - `parallel_threads`   - Number of computational threads to use on remote
                             workers.
    ---------------------------------------------------------------------
    - `shared_folder_on_local`  - The folder on your working machine containing
                                  the job input and output data.
    - `shared_folder_on_remote` - The place where job input and output data
                                  should be found on a remote worker.
    - `working_directory`       - The folder containing input data for the job
                                  and where tmp and output results should be stored
                                  (view from a remote worker).
    ---------------------------------------------------------------------
    - `external_mpiexec`   - If `cpp_communicator` is compiled with MPI installed
                             on system rather than the one provided with Herbert,
                             the full name (with path) to `mpiexec` program used to
                             run parallel job. Used only when `parallel_cluster == 'mpiexec_mpi'`.
    - `slurm_commands`     - Commands to be passed to Slurm, provided as a dict,
                             string, list of strings, or loaded from an sbatch-like file.
    =====================================================================
    - `known_clusters`     - Information property returning the list of the parallel
                             clusters known to Herbert.
    - `known_cluster_configs` - Information property returning the list of the
                             configurations available for the selected cluster.
    """

    _SAVED_PROPERTIES: List[str] = [
        "worker",
        "is_compiled_",
        "parallel_cluster",
        "cluster_config",
        "parallel_workers_number",
        "threads",
        "min_npix_for_omp_cut",
        "dynamic_omp_npixels_stride",
        "parallel_threads",
        "shared_folder_on_local",
        "shared_folder_on_remote",
        "working_directory",
        "external_mpiexec",
        "slurm_commands",
        "debug",
    ]
    _KNOWN_CLUSTERS: List[str] = [
        "herbert",
        "parpool",
        "mpiexec_mpi",
        "slurm_mpi",
        "none",
    ]

    n_cores: int = os.cpu_count() or 1

    def __init__(self) -> None:
        super().__init__("parallel_config")
        self.worker_: str = "worker_v4"
        self.is_compiled__: bool = False
        self.parallel_cluster_: str = "herbert"
        self.cluster_config_: str = "local"
        self.parallel_workers_number_: int = 2
        self.threads_: int = 0
        self.min_npix_for_omp_cut_: int = 100000
        self.dynamic_omp_npixels_stride_: int = 0
        self.parallel_threads_: int = 0
        self.shared_folder_on_local_: str = ""
        self.shared_folder_on_remote_: str = ""
        self.working_directory_: str = ""
        self.external_mpiexec_: str = ""
        self.slurm_commands_: Dict[str, str] = {}
        self.debug_: bool = False
        self._trial_cluster_config: str = ""

    # ------------------------------------------------------------------
    # Getters & Setters
    # ------------------------------------------------------------------

    @property
    def known_clusters(self) -> List[str]:
        """Information property returning the list of the parallel clusters
        known to Herbert. You cannot add or change a cluster using this method;
        the cluster has to be defined and subscribed via the clusters factory.
        """
        return list(self._KNOWN_CLUSTERS)

    @property
    def known_cluster_configs(self) -> List[str]:
        """Information property returning list of the known configurations
        available to run the selected cluster.
        """
        fram = self.parallel_cluster
        if fram.lower() == "none":
            return ["none"]
        return ["local", "default"]

    @property
    def worker(self) -> str:
        """The name of the script or program to run on cluster in parallel
        using parallel workers. The script has to be on the search path for all
        running sessions.
        """
        return str(self.get_or_restore_field("worker"))

    @worker.setter
    def worker(self, new_wrkr: str) -> None:
        self._check_and_set_worker(new_wrkr)

    def _is_worker_available(self, wrkr: str) -> bool:
        if not wrkr:
            return False
        if os.path.isfile(wrkr) or shutil.which(wrkr) is not None:
            return True
        return False

    def _check_and_set_worker(self, new_wrkr: str) -> None:
        if not isinstance(new_wrkr, str):
            raise HoraceConfigError(
                "HORACE:parallel_config:invalid_argument",
                "The worker property needs the executable script name",
            )
        store = config_store.instance()
        if self._is_worker_available(new_wrkr):
            store.store_config(self, "worker", new_wrkr)
            self.is_compiled_ = not new_wrkr.endswith(".m") and not new_wrkr.endswith(
                ".py"
            )
            return

        def_wrkr = self.worker_
        if new_wrkr == def_wrkr:
            cur_fmw = self.get_or_restore_field("parallel_cluster")
            if str(cur_fmw).lower() != "none":
                warn_with_id(
                    "HORACE:parallel_config:invalid_argument",
                    f"The script to run in parallel ({new_wrkr}) should be available "
                    "to all running sessions but parallel config can not find it."
                    " Parallel extensions are disabled",
                )
                store.store_config(
                    self, "parallel_cluster", "none", "cluster_config", "none"
                )
        else:
            store.store_config(self, "worker", def_wrkr)
            raise HoraceConfigError(
                "HORACE:parallel_config:invalid_argument",
                f"The script to run in parallel ({new_wrkr}) should be available "
                f"to all running sessions but parallel config can not find it. "
                f"keeping default worker {def_wrkr}",
            )

    @property
    def is_compiled(self) -> bool:
        """False if the worker is a script; True if the worker is compiled."""
        return self.is_compiled_

    @property
    def is_compiled_(self) -> bool:
        """Hidden property used to store `is_compiled` value without allowing
        setting `is_compiled` directly.
        """
        return bool(self.get_or_restore_field("is_compiled_"))

    @is_compiled_.setter
    def is_compiled_(self, val: bool) -> None:
        config_store.instance().store_config(self, "is_compiled_", bool(val))

    @property
    def parallel_cluster(self) -> str:
        """The name of a cluster to use for messages exchange.
        Currently available are `h[erbert]`, `p[arpool]`, `[m]piexec_mpi`,
        `[s]lurm_mpi`, and `none`:
        - `[h]erbert` -- Single-node file-based message cluster.
        - `[p]arpool` -- Parallel computing pool cluster.
        - `[m]piexec_mpi` -- C++ wrapped MPI libraries and `mpiexec`.
        - `[s]lurm_mpi` -- C++ wrapped MPI libraries submitted to Slurm job queues.
        - `none` -- Not available (when worker cannot be found on path).
        """
        wrkr = config_store.instance().get_value(self, "worker")
        if self._is_worker_available(wrkr):
            return str(self.get_or_restore_field("parallel_cluster"))
        return "none"

    @parallel_cluster.setter
    def parallel_cluster(self, cluster_name: Union[str, int]) -> None:
        selected = self.select_option(self._KNOWN_CLUSTERS, cluster_name)
        config_store.instance().store_config(self, "parallel_cluster", selected)

    @property
    def cluster_config(self) -> str:
        """The configuration describing the parallel cluster running the selected cluster.
        For `herbert` cluster, the configuration name can only be `'local'`.
        A `parpool` cluster accepts `'default'`. `'mpiexec_mpi'` accepts `'local'`
        or host file configurations in `herbert_core/admin/mpi_cluster_configs`.
        """
        return str(self.get_or_restore_field("cluster_config"))

    @cluster_config.setter
    def cluster_config(self, val: str) -> None:
        config_store.instance().store_config(self, "cluster_config", str(val))

    @property
    def parallel_workers_number(self) -> int:
        """Number of workers to deploy in parallel jobs."""
        return int(self.get_or_restore_field("parallel_workers_number"))

    @parallel_workers_number.setter
    def parallel_workers_number(self, val: int) -> None:
        if val < 1:
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                "Number of parallel workers must be more then 1",
            )
        config_store.instance().store_config(self, "parallel_workers_number", int(val))

    @property
    def threads(self) -> int:
        """Number of OMP threads to use."""
        n_threads = int(self.get_or_restore_field("threads"))
        if n_threads == 0:
            return self.n_cores
        return n_threads

    @threads.setter
    def threads(self, n_threads: int) -> None:
        n_threads = int(math.floor(n_threads))
        if n_threads < 0:
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                "threads must be positive or 0 (automatic)",
            )
        if n_threads > self.n_cores:
            warn_with_id(
                "HERBERT:parallel_config:threads",
                f"Number of threads ({n_threads}) might exceed computer capacity ({self.n_cores})",
            )
        config_store.instance().store_config(self, "threads", n_threads)

    @property
    def min_npix_for_omp_cut(self) -> int:
        """Running OMP on cuts with very small number of pixels is inefficient.
        If number of pixels in source object is smaller than specified,
        code reverts to single-threaded execution.
        """
        return int(self.get_or_restore_field("min_npix_for_omp_cut"))

    @min_npix_for_omp_cut.setter
    def min_npix_for_omp_cut(self, n_pixels: int) -> None:
        config_store.instance().store_config(
            self, "min_npix_for_omp_cut", int(math.floor(n_pixels))
        )

    @property
    def dynamic_omp_npixels_stride(self) -> int:
        """`bin_pixels` with OMP processes pixels in chunks of size
        `dynamic_omp_npixels_stride`, which is big enough to provide decent
        performance but does not cover whole pixels range divided into number
        of threads, as some chunks of work may hit areas with no contributing
        pixels. Set this value to -1 to make scheduling static.
        """
        return int(self.get_or_restore_field("dynamic_omp_npixels_stride"))

    @dynamic_omp_npixels_stride.setter
    def dynamic_omp_npixels_stride(self, stride: int) -> None:
        if not isinstance(stride, (int, float, np.number)) or isinstance(stride, bool):
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                f"Stride size must be numeric. Provided class {type(stride).__name__}",
            )
        config_store.instance().store_config(
            self, "dynamic_omp_npixels_stride", int(stride)
        )

    @property
    def parallel_threads(self) -> int:
        """Number of threads to use in MPIFramework on remote workers."""
        n_threads = int(self.get_or_restore_field("parallel_threads"))
        n_workers = max(1, int(self.get_or_restore_field("parallel_workers_number")))
        if n_threads == 0:
            return max(1, self.n_cores // n_workers)
        return n_threads

    @parallel_threads.setter
    def parallel_threads(self, n_threads: int) -> None:
        n_threads = int(math.floor(n_threads))
        n_workers = max(1, int(self.get_or_restore_field("parallel_workers_number")))
        mult = 2 if os.name == "nt" else 1
        n_poss_threads = max(1, (mult * self.n_cores) // n_workers)
        if n_threads < 0:
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                "parallel_threads must be positive or 0 (automatic)",
            )
        if n_threads > n_poss_threads:
            warn_with_id(
                "HERBERT:parallel_config:parallel_threads",
                f"Total number of threads used by all parallel workers n_threads*n_workers = ({n_threads * n_workers}) "
                f"might exceed computer capacity ({n_poss_threads} threads)",
            )
        config_store.instance().store_config(self, "parallel_threads", n_threads)

    @property
    def is_auto_parallel_threads(self) -> bool:
        """Threading using auto-calculated threads; used in Slurm because
        remote machine probably doesn't have same number of cores as local machine.
        """
        return int(self.get_or_restore_field("parallel_threads")) < 1

    @property
    def debug(self) -> bool:
        """Redirect IO to host and other debug features."""
        return bool(self.get_or_restore_field("debug"))

    @debug.setter
    def debug(self, val: Union[bool, int]) -> None:
        config_store.instance().store_config(self, "debug", bool(val))

    @property
    def shared_folder_on_local(self) -> str:
        """The folder on your working machine containing the job input and
        output data mounted on local machine and available from the remote
        machines. Must have read/write permissions for all machines.
        If empty, assumed that the local machine filesystem is shared
        with remote machine filesystem and have the same mounting points.
        """
        return str(self.get_or_restore_field("shared_folder_on_local"))

    @shared_folder_on_local.setter
    def shared_folder_on_local(self, val: Optional[str]) -> None:
        if not val:
            val = ""
        if not isinstance(val, str):
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                "The remote folder value should be a text string, describing the location of the input/output files on a remote machine",
            )
        config_store.instance().store_config(self, "shared_folder_on_local", val)

    @property
    def shared_folder_on_remote(self) -> str:
        """The place where a job data should be found on a remote worker.
        Must have read/write permissions for all machines.
        On a MPI worker should point to the physical location equal to
        that specified by `shared_folder_on_local` property.
        If empty, assumed to be equal to `shared_folder_on_local`.
        """
        folder = str(self.get_or_restore_field("shared_folder_on_remote"))
        if not folder:
            return self.shared_folder_on_local
        return folder

    @shared_folder_on_remote.setter
    def shared_folder_on_remote(self, val: Optional[str]) -> None:
        if not val:
            val = ""
        if not isinstance(val, str):
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                "The remote folder value should be a text string, describing the location of the input/output files on a remote machine",
            )
        config_store.instance().store_config(self, "shared_folder_on_remote", val)

    @property
    def remote_folder(self) -> str:
        """Alias for `shared_folder_on_remote`."""
        return self.shared_folder_on_remote

    @remote_folder.setter
    def remote_folder(self, val: Optional[str]) -> None:
        self.shared_folder_on_remote = val

    @property
    def working_directory(self) -> str:
        """Used as the folder where tmp files should be stored in parallel and
        non-parallel configuration.
        By default `gen_sqw` sets this value to the place where spe files are
        located. If you never did `gen_sqw` on a given machine, system tmp
        directory is used.
        Change this value to point to a fast & large disk or to a parallel
        file system. Assign empty value to restore it to default (system tmp
        directory).
        """
        work_dir = str(self.get_or_restore_field("working_directory"))
        if not work_dir:
            return tempfile.gettempdir()
        return work_dir

    @working_directory.setter
    def working_directory(self, val: Optional[str]) -> None:
        if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
            val = ""
        if not isinstance(val, str):
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                "working directory value should be a string",
            )
        if val:
            if os.path.normcase(os.path.normpath(val)) == os.path.normcase(
                os.path.normpath(tempfile.gettempdir())
            ):
                val = ""
            else:
                test_dir = os.path.join(val, "horace_test_write_directory")
                try:
                    os.makedirs(test_dir, exist_ok=True)
                    shutil.rmtree(test_dir, ignore_errors=True)
                except OSError:
                    warn_with_id(
                        "HERBERT:parallel_config:invalid_argument",
                        f"working directory {val} does not have write permissions. "
                        f"Changing it to {tempfile.gettempdir()} directory",
                    )
                    val = ""
        config_store.instance().store_config(self, "working_directory", val)

    @property
    def wkdir_is_default(self) -> bool:
        """Information field: True if `working_directory` has not been set
        (points to system tmpdir).
        """
        work_dir = str(self.get_or_restore_field("working_directory"))
        return len(work_dir) == 0

    @property
    def external_mpiexec(self) -> str:
        """If set up, specifies the `mpiexec` program with full path to it,
        used to launch parallel jobs instead of internal `mpiexec` program
        provided with Horace. Also accepts `True`/`False` values: `False`
        disables external `mpiexec` and `True` tries to identify `mpiexec`
        on system PATH.
        """
        return str(self.get_or_restore_field("external_mpiexec"))

    @external_mpiexec.setter
    def external_mpiexec(self, val: Union[str, bool, int, None]) -> None:
        if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
            val = ""
        elif isinstance(val, (bool, int, np.bool_, np.integer)):
            if bool(val):
                val = shutil.which("mpiexec") or ""
            else:
                val = ""
        if not isinstance(val, str):
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                "the value has to be a string specifying the program with full path to it to run mpi job",
            )
        config_store.instance().store_config(self, "external_mpiexec", val)

    @property
    def slurm_commands(self) -> Dict[str, str]:
        """Extra Slurm commands to be passed through to Slurm when initialising
        Slurm job. Can be provided as a dict, string, list of strings, or
        loaded from an sbatch-like file.
        """
        orig = self.get_or_restore_field("slurm_commands")
        if not orig:
            return {}
        return dict(orig)

    @slurm_commands.setter
    def slurm_commands(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, dict, str)) and len(val) == 0
        ):
            cmd_map: Dict[str, str] = {}
        elif isinstance(val, dict):
            cmd_map = {str(k): str(v) for k, v in val.items()}
        else:
            keys, vals = self.parse_slurm_commands(val)
            cmd_map = dict(zip(keys, vals))
        config_store.instance().store_config(self, "slurm_commands", cmd_map)

    def update_slurm_commands(
        self, val: Any, append: bool = False
    ) -> "parallel_config":
        """Update or replace `slurm_commands` dictionary."""
        cmd_map = dict(self.slurm_commands) if append else {}
        keys, vals = self.parse_slurm_commands(val)
        for k, v in zip(keys, vals):
            cmd_map[k] = v
        self.slurm_commands = cmd_map
        return self

    def load_slurm_commands_from_file(
        self, filename: str, append: bool = False
    ) -> "parallel_config":
        """Load `#SBATCH` directives from a Slurm batch script file."""
        if not os.path.isfile(filename):
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                f"File ({filename}) does not exist",
            )
        cmd_map = dict(self.slurm_commands) if append else {}
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^#SBATCH\s+([^#]+)", line)
                if m:
                    tokens = [t for t in re.split(r"[ \t=]+", m.group(1).strip()) if t]
                    if len(tokens) >= 2:
                        cmd_map[tokens[0]] = tokens[1]
        self.slurm_commands = cmd_map
        return self

    @staticmethod
    def parse_slurm_commands(val: Any) -> Tuple[List[str], List[str]]:
        """Parse Slurm commands into keys and values for building a map or updating one."""
        if isinstance(val, str):
            val = [t for t in re.split(r"[ \t=]+", val.strip()) if t]

        if not val:
            return [], []
        if isinstance(val, dict):
            return [str(k) for k in val.keys()], [str(v) for v in val.values()]
        if isinstance(val, (list, tuple)):
            if all(isinstance(x, str) for x in val):
                keys = [str(x) for x in val[0::2]]
                vals = [str(x) for x in val[1::2]]
                return keys, vals
            if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in val):
                keys = [str(x[0]) for x in val]
                vals = [str(x[1]) for x in val]
                return keys, vals
        raise HoraceConfigError(
            "HERBERT:parallel_config:invalid_argument",
            "slurm_commands must be string or cell array of strings",
        )

    @staticmethod
    def select_option(opt: Sequence[str], arg: Union[str, int]) -> str:
        """Select single-valued option from the list of available options.

        Inputs:
        - `opt` -- Sequence of available options.
        - `arg` -- Either string which uniquely defines one of the options by prefix,
          or the 1-based number selecting the option by index.
        """
        if isinstance(arg, (int, np.integer)) and not isinstance(arg, bool):
            if arg < 0 or arg > len(opt):
                raise HoraceConfigError(
                    "HERBERT:parallel_config:invalid_argument",
                    f"Option index {arg} out of range 0..{len(opt)}",
                )
            return opt[0] if arg == 0 else opt[arg - 1]
        if not isinstance(arg, str) or not arg:
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                "Selected framework must be a non-empty string or integer index",
            )
        matches = [o for o in opt if o.lower().startswith(arg.lower())]
        if not matches:
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                f"Unknown option: {arg}. Available options: {list(opt)}",
            )
        if len(matches) > 1:
            raise HoraceConfigError(
                "HERBERT:parallel_config:invalid_argument",
                f"Input name: {arg} is an ambiguous abbreviation of: {matches}",
            )
        return matches[0]

    def get_data_to_store(self) -> ConfigData:
        """Return the dictionary of savable fields and their values to store."""
        data = super().get_data_to_store()
        working_dir = str(data.get("working_directory", ""))
        tdr = tempfile.gettempdir()
        if working_dir and tdr.lower().startswith(working_dir.lower()):
            data["working_directory"] = ""
        return data

    def get_storage_field_names(self) -> List[str]:
        """Helper function returns the list of the names of the structure
        `get_data_to_store` returns.
        """
        return list(self._SAVED_PROPERTIES)

    def get_default_value(self, field_name: str) -> Any:
        """Method gets internal field value bypassing standard get/set
        methods interface.
        """
        val = getattr(self, f"{field_name}_")
        if isinstance(val, dict):
            return dict(val)
        return val


__all__ = ["parallel_config"]
