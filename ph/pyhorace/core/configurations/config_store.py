"""Singleton JSON-backed configuration store for PyHorace.

Translated from `herbert_core/configuration/@config_store` and modified to
persist configurations as standard `.json` files via Python's built-in `json`
module instead of MATLAB `.mat` files.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

if TYPE_CHECKING:
    from .config_base import config_base


# ============================================================================
# Exceptions, Warnings, and MATLAB-compatible Warning Tracking
# ============================================================================


class HoraceConfigError(Exception):
    """Exception raised for invalid configuration arguments or operations."""

    def __init__(self, err_id: str, message: str = "") -> None:
        self.err_id = err_id
        self.msg = message
        full_msg = f"{err_id}: {message}" if message else err_id
        super().__init__(full_msg)


class HoraceConfigWarning(UserWarning):
    """Warning category for configuration warnings."""


_LAST_WARNING: Tuple[str, str] = ("", "")


def warn_with_id(warn_id: str, message: str) -> None:
    """Issue a warning with a MATLAB-style warning ID and record it for lastwarn()."""
    global _LAST_WARNING
    _LAST_WARNING = (message, warn_id)
    warnings.warn(f"[{warn_id}] {message}", HoraceConfigWarning, stacklevel=2)


def lastwarn(
    new_msg: Optional[str] = None, new_id: Optional[str] = None
) -> Tuple[str, str]:
    """Return (or reset) the last issued (message, warning_id) pair."""
    global _LAST_WARNING
    if new_msg is not None or new_id is not None:
        _LAST_WARNING = (new_msg or "", new_id or "")
    return _LAST_WARNING


# ============================================================================
# Helper Data Structures & Option Parsers
# ============================================================================


class ConfigData(dict):
    """Dictionary supporting both key (`d['a']`) and attribute (`d.a`) access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'ConfigData' object has no attribute '{name}'"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'ConfigData' object has no attribute '{name}'"
            ) from exc

    def copy(self) -> "ConfigData":
        return ConfigData(copy.deepcopy(dict(self)))


CONFIG_CLASS_REGISTRY: Dict[str, Type["config_base"]] = {}


def _resolve_config_class(
    class_or_name: Union[str, "config_base", Type["config_base"]]
) -> Tuple[str, "config_base"]:
    """Return `(class_name, class_instance)` from a name, class, or instance."""
    from .config_base import config_base

    if isinstance(class_or_name, config_base):
        return class_or_name.class_name, class_or_name
    if isinstance(class_or_name, type) and issubclass(class_or_name, config_base):
        inst = class_or_name()
        return inst.class_name, inst
    if isinstance(class_or_name, str):
        cls = CONFIG_CLASS_REGISTRY.get(class_or_name)
        if cls is None:
            try:
                importlib.import_module(
                    f".{class_or_name}", package=__package__ or "pyhorace.core.configurations"
                )
                cls = CONFIG_CLASS_REGISTRY.get(class_or_name)
            except ImportError:
                pass
        if cls is None:
            raise HoraceConfigError(
                "HERBERT:config_store:invalid_argument",
                f"Unknown configuration class '{class_or_name}'",
            )
        inst = cls()
        return class_or_name, inst
    raise HoraceConfigError(
        "HERBERT:config_store:invalid_argument",
        f"Invalid config class specification: {class_or_name!r}",
    )


def _parse_char_options(
    args: Sequence[Any], known_options: Sequence[str]
) -> Tuple[bool, str, List[bool], List[Any]]:
    """Parse prefix-matched flag options (like `'-files'`, `'-in_mem'`, `'-buffer'`)."""
    flags = [False] * len(known_options)
    remaining: List[Any] = []
    for arg in args:
        if isinstance(arg, str) and arg.startswith("-"):
            matches = [
                idx
                for idx, opt in enumerate(known_options)
                if opt.lower().startswith(arg.lower())
            ]
            if len(matches) == 1:
                flags[matches[0]] = True
            elif len(matches) > 1:
                return (
                    False,
                    f"Ambiguous option '{arg}' matches {[known_options[i] for i in matches]}",
                    flags,
                    remaining,
                )
            else:
                return False, f"Unrecognized option '{arg}'", flags, remaining
        else:
            remaining.append(arg)
    return True, "", flags, remaining


def _json_default_serializer(obj: Any) -> Any:
    """Serialize numpy arrays/scalars or custom containers into standard JSON types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (frozenset,)) or type(obj).__name__ == "set":
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _values_equal(v1: Any, v2: Any) -> bool:
    """Deep equality check for config values (including lists/numpy arrays/dicts)."""
    if type(v1) is not type(v2) and not (
        isinstance(v1, (int, float, np.number, list, np.ndarray))
        and isinstance(v2, (int, float, np.number, list, np.ndarray))
    ):
        return False
    if isinstance(v1, (list, tuple, np.ndarray)) and isinstance(
        v2, (list, tuple, np.ndarray)
    ):
        try:
            a1 = np.asarray(v1)
            a2 = np.asarray(v2)
            if a1.shape != a2.shape:
                return False
            return bool(np.array_equal(a1, a2))
        except Exception:
            return list(v1) == list(v2)
    if isinstance(v1, dict) and isinstance(v2, dict):
        if list(v1.keys()) != list(v2.keys()):
            return False
        return all(_values_equal(v1[k], v2[k]) for k in v1)
    return bool(v1 == v2)


def _sys_memory() -> Tuple[int, int]:
    """Return `(free_phys_mem_bytes, total_phys_mem_bytes)`."""
    try:
        import ctypes

        if os.name == "nt":

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys), int(stat.ullTotalPhys)
    except Exception:
        pass
    # Fallback default: 8 GiB free, 16 GiB total
    return 8 * 1024**3, 16 * 1024**3


def _default_config_root() -> str:
    """Return default parent directory for `mprogs_config`."""
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(appdata, "pyhorace")


# ============================================================================
# config_store: Singleton JSON-backed Configuration Store
# ============================================================================


class config_store:
    """Class provides single storage point for various configuration classes.

    It stores/restores the classes inheriting from `config_base` class.
    Persists configurations in `.json` files inside `config_folder` using
    Python's standard `json` module.

    Properties
    ----------
    config_folder : str
        The full path to the folder where the configuration is stored.
    config_classes : list of str
        Property to observe config classes currently initiated in the singleton
        (mainly for testing and debug purposes).
    config_folder_name : str
        The name of the folder where the configuration is stored.
    """

    _unique_store: Optional["config_store"] = None
    DEFAULT_CONFIG_FOLDER_NAME: str = "mprogs_config"

    def __init__(self, config_folder_path: Optional[str] = None) -> None:
        self._config_folder: str = ""
        self._config_folder_name: str = self.DEFAULT_CONFIG_FOLDER_NAME
        self._config_storage: Dict[str, ConfigData] = {}
        self._saveable: Dict[str, bool] = {}

        if config_folder_path:
            self.set_config_path(config_folder_path)
        else:
            self.set_config_path(_default_config_root())

    @classmethod
    def instance(cls, *args: Any) -> Optional["config_store"]:
        """Get instance of unique `config_store` implementation.

        Usage:
            `con = config_store.instance([a_config_folder_name | 'clear'])`

        Where optional parameter does the following:
        - `'clear'` -- removes all configurations from memory.
        - `a_config_folder_name` -- if present, sets the location of the
          current config store in memory to the folder provided.
        """
        if len(args) > 0:
            arg0 = args[0]
            if isinstance(arg0, str) and arg0.lower() == "clear":
                cls._unique_store = None
                return None
            cls._unique_store = None
            cls._unique_store = cls(arg0)
            return cls._unique_store

        if cls._unique_store is None:
            cls._unique_store = cls()
        return cls._unique_store

    @classmethod
    def set_config_folder(cls, config_folder_name: str) -> None:
        """Set the location for a folder with configuration to a location provided as input.

        If the folder does not exist, it is created. The configurations
        currently in memory are retained but will be saved to the new
        location on request only.
        """
        if not isinstance(config_folder_name, str) or not config_folder_name:
            raise HoraceConfigError(
                "HERBERT:config_store:invalid_argument",
                f"Config folder name has to be a non-empty string, got {type(config_folder_name).__name__}",
            )
        if config_folder_name.lower() == "clear":
            raise HoraceConfigError(
                "HERBERT:config_store:invalid_argument",
                "Config folder name cannot be 'clear'",
            )
        cls.instance(config_folder_name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config_folder(self) -> str:
        """The full path to the folder where the configuration is stored."""
        return self._config_folder

    @config_folder.setter
    def config_folder(self, val: str) -> None:
        """Simplified setter for config path.

        Never copies configuration to new place. Assumes that it is either
        newly configured or has some configuration there. Removes all existing
        configurations from memory to pick up defaults or load configurations
        from the new folder.
        """
        self.set_config_path(val, "-clean")

    @property
    def config_folder_name(self) -> str:
        """The name of the folder where the configuration is stored."""
        return self._config_folder_name

    @property
    def config_classes(self) -> List[str]:
        """Property to observe config classes currently initiated in the singleton
        (mainly for testing and debug purposes).
        """
        return list(self._config_storage.keys())

    def set_config_path(self, new_path: str, *options: str) -> None:
        """Set new config store path.

        Usage:
            `config_store.instance().set_config_path(new_path, ['-clean'])`

        Parameters
        ----------
        new_path : str
            The path to store new configuration.
        options : str, optional
            `'-clean'` -- existing configuration is unloaded from memory.
            This option should be used within MPI workers to ensure they do not
            keep any spurious configuration in memory, and load shared
            configuration from config folder.
        """
        if not isinstance(new_path, str) or not new_path:
            raise HoraceConfigError(
                "HERBERT:config_store:invalid_argument",
                "new_path must be a non-empty string",
            )
        norm_path = os.path.normpath(new_path)
        folder_name = os.path.basename(norm_path)
        if self.DEFAULT_CONFIG_FOLDER_NAME in folder_name:
            self._config_folder_name = folder_name
            target_dir = norm_path
        else:
            self._config_folder_name = self.DEFAULT_CONFIG_FOLDER_NAME
            target_dir = os.path.join(norm_path, self._config_folder_name)

        os.makedirs(target_dir, exist_ok=True)
        self._config_folder = target_dir

        if any(isinstance(o, str) and o.lower().startswith("-clean") for o in options):
            self._config_storage.clear()
            self._saveable.clear()

    # ------------------------------------------------------------------
    # JSON File I/O Helpers (replacing MATLAB .mat with .json)
    # ------------------------------------------------------------------

    def _config_file_path(self, class_name: str) -> str:
        """Return full path to `<class_name>.json`."""
        if os.path.isabs(class_name) and class_name.endswith(".json"):
            return class_name
        return os.path.join(self._config_folder, f"{class_name}.json")

    def save_config(self, class_or_filename: str, data_to_save: Dict[str, Any]) -> None:
        """Save config class information to the appropriate `.json` config file."""
        filepath = self._config_file_path(class_or_filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {"config_data": dict(data_to_save)}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default_serializer)

    @staticmethod
    def load_config_from_file(
        filename: str, class_field_names: Sequence[str]
    ) -> Tuple[Optional[ConfigData], int, str]:
        """Load configuration from JSON file and verify its field structure.

        Returns
        -------
        config_data : ConfigData or None
        result : int
            1 on success, 0 if outdated/mismatched fields, -1 if missing/corrupt.
        mess : str
        """
        if not os.path.isfile(filename):
            return None, -1, f"File: {filename} does not exist"
        try:
            with open(filename, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as exc:
            return None, -1, f"Failed to load JSON from {filename}: {exc}"

        if not isinstance(loaded, dict) or "config_data" not in loaded:
            return (
                None,
                -1,
                f"File: {filename} does not contain valid 'config_data' field",
            )

        raw_data = loaded["config_data"]
        if not isinstance(raw_data, dict):
            return None, -1, f"File: {filename} has non-object 'config_data'"

        if list(raw_data.keys()) != list(class_field_names):
            return (
                None,
                0,
                f"Stored configuration for file: {filename} does not match the structure of the configuration class",
            )

        return ConfigData(raw_data), 1, ""

    # ------------------------------------------------------------------
    # Core Store / Restore Methods
    # ------------------------------------------------------------------

    def store_config(
        self,
        config_class: Union[str, "config_base", Type["config_base"]],
        *args: Any,
    ) -> None:
        """Store configuration in memory and on file if requested.

        `config_class` instances whose property `saveable == True` are saved
        both into memory and to disk (`.json` file).

        If option `'-forcesave'` (or `'-force'`) is provided, the file is saved
        to disk regardless of its change status in memory.
        """
        ok, mess, flags, remaining = _parse_char_options(args, ["-forcesave"])
        if not ok:
            raise HoraceConfigError("HERBERT:config_store:invalid_argument", mess)
        force_save = any(flags)

        _, class_inst = _resolve_config_class(config_class)
        self._store_config_internal(class_inst, remaining, force_save)

    def _store_config_internal(
        self,
        config_instance: "config_base",
        fields_to_store: Sequence[Any],
        force_save: bool,
    ) -> None:
        class_name = config_instance.class_name
        non_saveable = list(config_instance.mem_only_prop_list)
        do_not_save = False

        if not fields_to_store:
            data_to_store = ConfigData(config_instance.get_data_to_store())
            for fld in non_saveable:
                val = self._get_config_field_value(config_instance, fld)
                data_to_store[fld] = val
        else:
            data_to_store, _ = self._get_config_internal(config_instance, True)
            data_to_store = data_to_store.copy()
            if len(fields_to_store) % 2 != 0:
                raise HoraceConfigError(
                    "HERBERT:config_store:invalid_argument",
                    "Incomplete Pair: Arguments of store_config have to be 'field_name', field_value pairs",
                )
            for i in range(0, len(fields_to_store), 2):
                field_name = fields_to_store[i]
                if not isinstance(field_name, str):
                    raise HoraceConfigError(
                        "HERBERT:config_store:invalid_argument",
                        "Invalid Parameter: First element in pair has to be string field_name",
                    )
                field_val = fields_to_store[i + 1]
                data_to_store[field_name] = field_val
                if len(fields_to_store) == 2 and field_name in non_saveable:
                    do_not_save = True

        # Check if anything changed vs in-memory storage
        data_changed = True
        if class_name in self._config_storage:
            existing = self._config_storage[class_name]
            if _values_equal(existing, data_to_store):
                data_changed = False

        self._config_storage[class_name] = data_to_store.copy()
        saveable = config_instance.saveable
        self._saveable[class_name] = saveable

        if do_not_save:
            return

        if (saveable and data_changed) or force_save:
            disk_data = data_to_store.copy()
            for fld in non_saveable:
                disk_data.pop(fld, None)
            self.save_config(class_name, disk_data)

    def _get_config_internal(
        self,
        class_to_restore: Union[str, "config_base", Type["config_base"]],
        use_defaults_for_missing: bool,
    ) -> Tuple[ConfigData, bool]:
        class_name, class_inst = _resolve_config_class(class_to_restore)

        if class_name in self._config_storage:
            return self._config_storage[class_name].copy(), False

        filename = self._config_file_path(class_name)
        class_fields = class_inst.get_storage_field_names()
        config_data, result, _ = self.load_config_from_file(filename, class_fields)
        got_from_file = True

        if result != 1:
            if class_inst.warn_if_missing_config:
                if result == 0:
                    warn_with_id(
                        "HERBERT:config_store:outdated_configuration",
                        f"Stored configuration for class: {class_name} is outdated\n"
                        " The configuration has been reset to defaults ",
                    )
                elif use_defaults_for_missing:
                    warn_with_id(
                        "HERBERT:config_store:default_configuration",
                        f"Custom configuration for class: {class_name} does not exist\n"
                        f" The configuration has been set to defaults. Type:\n"
                        f">>{class_name}\n   to check if defaults are correct",
                    )
            if os.path.isfile(filename):
                try:
                    os.unlink(filename)
                except OSError:
                    pass
            got_from_file = False

        if config_data is None:
            if use_defaults_for_missing:
                config_data = ConfigData(class_inst.get_defaults())
                got_from_file = False
            else:
                return ConfigData(), False

        self._config_storage[class_name] = config_data.copy()
        if class_name not in self._saveable:
            self._saveable[class_name] = class_inst.get_saveable_default()

        return config_data.copy(), got_from_file

    def get_config(
        self, class_to_restore: Union[str, "config_base", Type["config_base"]]
    ) -> ConfigData:
        """Return configuration from memory or load it from a file if such
        configuration exists on file and not in memory.

        Parameters
        ----------
        class_to_restore : str or config_base
            The class instance which should be restored from disk or the name
            of this class. If `class_to_restore` has `returns_defaults == True`,
            default class configuration is returned.
        """
        _, class_inst = _resolve_config_class(class_to_restore)
        config_data, read_from_file = self._get_config_internal(class_inst, True)
        if read_from_file:
            ss = class_inst.saveable
            class_inst.saveable = False
            try:
                class_inst.set_stored_data(config_data)
            finally:
                class_inst.saveable = ss
        return config_data

    def _get_config_field_value(
        self,
        class_to_restore: Union[str, "config_base", Type["config_base"]],
        *field_names: str,
    ) -> Any:
        _, class_inst = _resolve_config_class(class_to_restore)
        config_data = self.get_config(class_inst)
        results: List[Any] = []
        for field_name in field_names:
            if field_name in config_data:
                results.append(config_data[field_name])
            else:
                if field_name not in class_inst.mem_only_prop_list:
                    warn_with_id(
                        "HERBERT:config_store:default_field_value",
                        f"Field {field_name} is not stored in the configuration {class_inst.class_name}. Returning default value",
                    )
                val = class_inst.get_default_value(field_name)
                results.append(val)
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def get_config_field(
        self,
        class_to_restore: Union[str, "config_base", Type["config_base"]],
        *field_names: str,
    ) -> Any:
        """Return the values of the requested field(s) from the specified configuration.

        Loads configuration from file if it is present there and has not been
        loaded in memory before.

        Parameters
        ----------
        class_to_restore : str or config_base
            The configuration class or its name to get values from.
        *field_names : str
            The name(s) of the configuration field(s) to return values for.

        Returns
        -------
        Any or tuple of Any
            The value of the requested configuration field (or tuple of values
            when multiple fields are requested).
        """
        return self._get_config_field_value(class_to_restore, *field_names)

    def get_value(
        self,
        class_to_restore: Union[str, "config_base", Type["config_base"]],
        *field_names: str,
    ) -> Any:
        """Return specific config property value or list of values from a config class.

        Loads configuration from file if it is present there and has not been
        loaded in memory before. Identical to `get_config_field`.
        """
        return self._get_config_field_value(class_to_restore, *field_names)

    def set_value(
        self,
        class_to_Update: Union[str, "config_base", Type["config_base"]],
        *field_val_pairs: Any,
    ) -> None:
        """Set value of config field directly in memory.

        Unlike the usual way of setting value using property assignment,
        changing value this way does not affect values stored on disk. The
        consequence is much faster change of config values, but the changes
        become transient. If memory is cleared, these changes are not restored.
        """
        class_name, class_inst = _resolve_config_class(class_to_Update)
        config_data = self.get_config(class_inst)
        for i in range(0, len(field_val_pairs), 2):
            fld = field_val_pairs[i]
            val = field_val_pairs[i + 1]
            if fld not in config_data:
                warn_with_id(
                    "HERBERT:config_store:unknown_field",
                    f"Setting field {fld} which has not been previously defined in class {class_name}",
                )
            config_data[fld] = val
        self._config_storage[class_name] = config_data

    def has_config(self, class_name: Union[str, "config_base"]) -> bool:
        """Check if the class with given name has configuration stored in a file.

        In other words, checks whether a configuration has ever been changed
        from defaults and saved to disk.
        """
        from .config_base import config_base

        name = (
            class_name.class_name
            if isinstance(class_name, config_base)
            else str(class_name)
        )
        return os.path.isfile(self._config_file_path(name))

    def get_saveable(
        self, class_to_check: Union[str, "config_base", Type["config_base"]]
    ) -> bool:
        """Return True if a configuration class requested as input is saveable,
        i.e. changes in configuration are stored on disk.

        Usage:
            `is_sav = config_store.instance().get_saveable('class_name')`
            or
            `is_sav = config_store.instance().get_saveable(class_instance)`
        """
        class_name, class_inst = _resolve_config_class(class_to_check)
        if class_name in self._saveable:
            return self._saveable[class_name]
        is_sav = class_inst.get_saveable_default()
        self._saveable[class_name] = is_sav
        return is_sav

    def set_saveable(
        self,
        class_instance: Union[str, "config_base", Type["config_base"]],
        is_it: Union[bool, int],
    ) -> None:
        """Set or clear the property which defines if changes in the class
        configuration are stored on disk.

        Usage:
            `config_store.instance().set_saveable('class_name', to_save)`
            or
            `config_store.instance().set_saveable(class_instance, to_save)`
        """
        class_name, _ = _resolve_config_class(class_instance)
        self._saveable[class_name] = bool(is_it)

    def clear_config(
        self,
        class_instance: Union[str, "config_base", Type["config_base"]],
        *options: str,
    ) -> None:
        """Clear configuration from memory.

        If option `'-files'` (or `'-file'`) exists, it also deletes the file
        where this configuration is stored.
        """
        from .config_base import config_base

        ok, mess, flags, _ = _parse_char_options(options, ["-files"])
        if not ok:
            raise HoraceConfigError("HERBERT:config_store:invalid_argument", mess)
        clear_file = flags[0]

        if isinstance(class_instance, str):
            class_name = class_instance
        elif isinstance(class_instance, config_base):
            class_name = class_instance.class_name
        elif isinstance(class_instance, type) and issubclass(
            class_instance, config_base
        ):
            class_name = class_instance().class_name
        else:
            raise HoraceConfigError(
                "HERBERT:config_store:invalid_argument",
                f"Invalid class_instance: {class_instance!r}",
            )

        self._config_storage.pop(class_name, None)
        self._saveable.pop(class_name, None)

        if clear_file:
            cfg_file = self._config_file_path(class_name)
            if os.path.isfile(cfg_file):
                try:
                    os.unlink(cfg_file)
                except OSError:
                    pass

    def delete_all_files(self) -> None:
        """Delete all `.json` files corresponding to configurations in memory."""
        for class_name in list(self._config_storage.keys()):
            cfg_file = self._config_file_path(class_name)
            if os.path.isfile(cfg_file):
                try:
                    os.unlink(cfg_file)
                except OSError:
                    pass

    def clear_all(self, *options: str) -> None:
        """Clear all configurations stored in memory.

        If option `'-files'` (or `'-file'`) exists, it also deletes all files
        related to configurations stored in memory.
        """
        ok, mess, flags, _ = _parse_char_options(options, ["-files"])
        if not ok:
            raise HoraceConfigError("HERBERT:config_store:invalid_argument", mess)
        if flags[0]:
            self.delete_all_files()
        config_store.instance("clear")

    def is_configured(
        self,
        class_instance: Union[str, "config_base", Type["config_base"]],
        *options: str,
    ) -> bool:
        """Check if `class_instance` is stored within the `config_store`.

        If option `'-in_mem'` is provided, it checks only if such configuration
        is loaded in memory.
        """
        from .config_base import config_base

        ok, mess, flags, _ = _parse_char_options(options, ["-in_mem"])
        if not ok:
            raise HoraceConfigError("HERBERT:config_store:invalid_argument", mess)
        check_mem_only = flags[0]

        if isinstance(class_instance, str):
            class_name = class_instance
        elif isinstance(class_instance, config_base):
            class_name = class_instance.class_name
        else:
            class_name = class_instance().class_name

        in_mem = class_name in self._config_storage
        if check_mem_only or in_mem:
            return in_mem

        cfg_file = self._config_file_path(class_name)
        return os.path.isfile(cfg_file)

    def is_field_configured(
        self,
        class_inst_or_name: Union[str, "config_base", Type["config_base"]],
        field_name: str,
    ) -> bool:
        """Check if the specified field from the specified configuration class
        has a configured value in memory or on disk.
        """
        config_data, _ = self._get_config_internal(class_inst_or_name, False)
        return field_name in config_data

    def get_all_configs(self) -> Dict[str, Any]:
        """Return all config info currently loaded in memory."""
        snapshot: Dict[str, Any] = {}
        for name, data in self._config_storage.items():
            conf = data.copy()
            conf["saveable"] = self.get_saveable(name)
            snapshot[name] = conf
        snapshot["config_folder"] = self.config_folder
        return snapshot

    def set_all_configs(self, config_struct: Dict[str, Any]) -> None:
        """Set up config info previously retrieved by `get_all_configs` into
        memory as current configuration.
        """
        data_copy = copy.deepcopy(config_struct)
        folder = data_copy.pop("config_folder", None)
        if folder:
            self._config_folder = folder
            os.makedirs(self._config_folder, exist_ok=True)

        for name, cfdata in data_copy.items():
            _, cfcl = _resolve_config_class(name)
            saveable = cfdata.pop("saveable", True)
            cfcl.saveable = False
            for k, v in cfdata.items():
                setattr(cfcl, k, v)
            cfcl.saveable = saveable
            if saveable:
                self.save_config(name, cfcl.get_data_to_store())


__all__ = [
    "CONFIG_CLASS_REGISTRY",
    "ConfigData",
    "HoraceConfigError",
    "HoraceConfigWarning",
    "config_store",
    "lastwarn",
    "warn_with_id",
]
