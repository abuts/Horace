"""Abstract base class for all PyHorace configuration classes.

Translated from `herbert_core/configuration/@config_base`.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config_store import (
    CONFIG_CLASS_REGISTRY,
    ConfigData,
    HoraceConfigError,
    _parse_char_options,
    config_store,
    warn_with_id,
)


class _ClassOrInstanceMethod:
    """Descriptor allowing a method on `config_base` to be called either on an
    instance (`cfg.get(...)`) or directly on the class (`graph_config.get(...)`),
    in which case a transient instance `cls()` is used.
    """

    def __init__(self, func: Any) -> None:
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        if instance is None:
            instance = owner()
        return self.func.__get__(instance, owner)


class config_base(ABC):
    """Base class for configurations, which have a single instance for the whole
    package and can be automatically stored/restored from disk using `config_store`.

    All derived classes used with configuration have to define two abstract
    methods of this class:

    1. Helper function returning the list of the public properties whose values
       need to be stored:
       `fields = obj.get_storage_field_names()`
    2. Method returning the default property value defined by the config class
       instance, ignoring the current value stored in common configuration and
       returned by the usual property getter:
       `value = obj.get_default_value(field_name)`

    And specify the setters and getters for all stored properties in the
    following form:

    a) The property itself is defined as a `@property` on the derived class.
    b) It has a default value stored on an internal attribute whose name differs
       from the public property name by a trailing underscore (e.g.
       `self.stored_property_ = default_value`).
    c) Its getter calls `self.get_or_restore_field('stored_property')`.
    d) Its setter calls `config_store.instance().store_config(self, 'stored_property', val)`.

    Properties
    ----------
    class_name : str
        Defines the name of the derived storage class. The storage knows the
        stored configuration under this name.
    saveable : bool
        Specifies if changes to the class should be stored on disk to restore
        them later.
    returns_defaults : bool
        If this property is set to True, class getters return default
        configurations instead of saved configurations.
    config_folder : str
        The folder where the configuration data are stored (defined by
        `config_store` class, and provided here as an interface to it).
    do_check_combo_arg : bool
        Similarly to `Serializable`, allows disabling the check for
        interdependent properties until they all have been set up.
    mem_only_prop_list : list of str
        Defines the list of properties which are never stored on disk and exist
        in memory only. To define such properties, the child class should add
        the name of the property to `self.mem_only_prop_list_`.
    warn_if_missing_config : bool
        If True, issue warning if class has never been configured and its
        values are chosen from defaults.
    disable_setup_warnings : bool
        If True, disables warnings issued during loading when certain
        properties are set to a new, potentially invalid value. Default is False.
    """

    saveable_default_: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        CONFIG_CLASS_REGISTRY[cls.__name__] = cls

    def __init__(self, class_name: Optional[str] = None) -> None:
        """Constructor accepts input parameter which should be the derived class name.

        Parameters
        ----------
        class_name : str, optional
            The string which defines the stored class name.
        """
        if class_name is not None and not isinstance(class_name, str):
            raise HoraceConfigError(
                "HERBERT:config_base:constructor",
                "first config_base variable has to be a string, providing the derived class name",
            )
        self.class_name_: str = class_name or type(self).__name__
        self.returns_defaults_: bool = False
        self.do_check_combo_arg_: bool = True
        self.mem_only_prop_list_: List[str] = []
        self.warn_if_missing_config_: bool = True
        self.disable_setup_warnings_: bool = False
        CONFIG_CLASS_REGISTRY[self.class_name_] = type(self)

    # ------------------------------------------------------------------
    # Abstract Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_storage_field_names(self) -> List[str]:
        """Helper function returns the list of the public properties whose
        values one needs to store.
        """

    @abstractmethod
    def get_default_value(self, field_name: str) -> Any:
        """Get default value of any public configuration property.

        Relies on the assumption that each public field has a private field
        with a name differing by a trailing underscore (`field_name + '_'`).
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_name(self) -> str:
        """Property defines the name of the derived storage class.

        The storage knows the stored configuration under this name.
        """
        return self.class_name_

    @property
    def saveable(self) -> bool:
        """Property specifies if changes to the class should be stored on disk
        to restore them later.
        """
        return config_store.instance().get_saveable(self)

    @saveable.setter
    def saveable(self, val: Union[bool, int]) -> None:
        if not isinstance(val, (bool, int, np.bool_, np.integer)):
            raise HoraceConfigError(
                "HERBERT:config_base:invalid_argument",
                "saveable property can be only boolean or convertible to boolean",
            )
        config_store.instance().set_saveable(self, bool(val))

    @property
    def returns_defaults(self) -> bool:
        """If this property is set to True, class getters return default
        configurations instead of saved configurations.
        """
        return self.returns_defaults_

    @returns_defaults.setter
    def returns_defaults(self, val: Union[bool, int]) -> None:
        self.returns_defaults_ = bool(val)

    @property
    def config_folder(self) -> str:
        """The folder where the configuration data are stored (defined by
        `config_store` class, and provided here as an interface to it).
        """
        return config_store.instance().config_folder

    @config_folder.setter
    def config_folder(self, val: str) -> None:
        warn_with_id(
            "HERBERT:temporary_config_path",
            " Defining configuration path from the configuration is valid for single session only.\n"
            " Mention `config_store.set_config_folder(path)` in setup to keep this location constant",
        )
        config_store.instance().set_config_path(val)

    @property
    def is_default(self) -> bool:
        """Check if a configuration has been changed by user or still has its
        default values.
        """
        return not config_store.instance().is_configured(self)

    @property
    def do_check_combo_arg(self) -> bool:
        """Similarly to `Serializable`, allows disabling the check for
        interdependent properties until they all have been set up.
        """
        return self.do_check_combo_arg_

    @do_check_combo_arg.setter
    def do_check_combo_arg(self, val: bool) -> None:
        self.do_check_combo_arg_ = bool(val)

    @property
    def mem_only_prop_list(self) -> List[str]:
        """Defines the list of properties which are never stored on disk and
        exist in memory only.

        To define such properties, the child class should add the name of the
        property to this list.
        """
        return list(self.mem_only_prop_list_)

    @property
    def warn_if_missing_config(self) -> bool:
        """If True, issue warning if class has never been configured and its
        values are chosen from defaults.
        """
        return self.warn_if_missing_config_

    @warn_if_missing_config.setter
    def warn_if_missing_config(self, val: bool) -> None:
        self.warn_if_missing_config_ = bool(val)

    @property
    def disable_setup_warnings(self) -> bool:
        """If True, disables warnings issued during loading when certain
        properties are set to a new, potentially invalid value. Default is False.
        """
        return self.disable_setup_warnings_

    @disable_setup_warnings.setter
    def disable_setup_warnings(self, val: bool) -> None:
        self.disable_setup_warnings_ = bool(val)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def is_field_configured(self, field_name: str) -> bool:
        """Check if `field_name` is configured in memory or on disk."""
        return config_store.instance().is_field_configured(self, field_name)

    def get_or_restore_field(self, field_name: str) -> Any:
        """Method to restore value from `config_store` if available or take
        default value from the class defaults if not.

        The method is used as part of a standard derived class getter.
        """
        if self.returns_defaults_:
            return self.get_default_value(field_name)
        return config_store.instance().get_config_field(self, field_name)

    def get_defaults(self) -> ConfigData:
        """Return the structure (`ConfigData`) with default class data."""
        fields = self.get_storage_field_names()
        data = ConfigData()
        for field_name in fields:
            data[field_name] = copy.deepcopy(self.get_default_value(field_name))
        return data

    def get_data_to_store(self) -> ConfigData:
        """Return the structure (`ConfigData`) with the data expected to be
        stored in configuration.
        """
        fields = self.get_storage_field_names()
        data = ConfigData()
        for field_name in fields:
            data[field_name] = getattr(self, field_name)
        return data

    def get_all_configured(self) -> Tuple[ConfigData, ConfigData]:
        """Return all configurable fields stored in memory as
        `(saveable_data, mem_only_data)`.
        """
        data = self.get_data_to_store()
        mem_fields = self.mem_only_prop_list
        mem_data = ConfigData()
        for field_name in mem_fields:
            mem_data[field_name] = getattr(self, field_name)
        return data, mem_data

    def set_stored_data(self, data: Dict[str, Any]) -> "config_base":
        """Execute class setters for the config `class_instance` using `data`
        structure provided as argument.

        `data` structure has to have fields with names equal to the names of
        the class setters. It can be used to load data from `config_store` to
        config class instance though such usage is not standard and should be
        used for testing and debugging purposes only.
        """
        self.do_check_combo_arg = False
        try:
            for field_name, val in data.items():
                setattr(self, field_name, val)
        finally:
            self.do_check_combo_arg = True
        self.check_combo_arg()
        return self

    def get_saveable_default(self) -> bool:
        """Return the default saveable state for the particular object.

        If object is not saveable by default, this method should be overloaded.
        """
        return self.saveable_default_

    def check_combo_arg(self) -> "config_base":
        """Do validation of interdependent properties. Subclasses may override."""
        return self

    def get_all(self, public_only: bool = True) -> ConfigData:
        """Return structure of all public properties (`get(obj)`)."""
        fields = self.get_storage_field_names()
        res = ConfigData()
        for fld in fields:
            res[fld] = getattr(self, fld)
        if not public_only:
            for fld in self.mem_only_prop_list:
                res[fld] = getattr(self, fld)
        return res

    @classmethod
    def instance(cls) -> "config_base":
        """Return an instance of this configuration class pointing to the
        shared `config_store` singleton.
        """
        return cls()

    @_ClassOrInstanceMethod
    def reset(self) -> "config_base":
        """Reset all configuration properties of this class back to their
        default values.
        """
        return self.set("defaults")

    @_ClassOrInstanceMethod
    def get(self, *args: Any) -> Any:
        """Retrieve configuration properties (`get(obj, ...)`).

        Supports:
        - `obj.get()` / `cls.get()` -> `ConfigData` of all stored public properties
        - `obj.get('defaults')` -> `ConfigData` of default properties
        - `obj.get('field1', 'field2', ...)` -> value or tuple of values
        """
        ok, _, flags, rem = _parse_char_options(args, ["-public"])
        public_only = flags[0] if ok and any(flags) else True
        if ok:
            args = tuple(rem)

        if len(args) == 0:
            return self.get_all(public_only=public_only)

        if (
            len(args) == 1
            and isinstance(args[0], str)
            and "defaults".startswith(args[0].lower())
            and len(args[0]) >= 3
        ):
            saved_rd = self.returns_defaults
            self.returns_defaults = True
            try:
                return self.get_all(public_only=public_only)
            finally:
                self.returns_defaults = saved_rd

        known_fields = {*self.get_storage_field_names(), *self.mem_only_prop_list}
        results: List[Any] = []
        for name in args:
            if not isinstance(name, str) or (
                name not in known_fields and not hasattr(type(self), name)
            ):
                raise HoraceConfigError(
                    "HERBERT:config_base:invalid_argument",
                    f"Property '{name}' is not recognized on {self.class_name}",
                )
            results.append(getattr(self, name))

        if len(results) == 1:
            return results[0]
        return tuple(results)

    @_ClassOrInstanceMethod
    def set(self, *args: Any, **kwargs: Any) -> "config_base":
        """Set configuration properties (`set(obj, ...)`).

        Supports:
        - `'defaults'` / `'default'` / `'def'` -> reset to defaults
        - `'saved'` / `'save'` -> restore from saved `.json` file (or defaults)
        - `'-buffer'` / `'-save'` options
        - `dict` / `ConfigData`, list of pairs, or `'field', value` pairs.
        """
        ok, mess, flags, rem = _parse_char_options(args, ["-buffer", "-save"])
        if not ok:
            raise HoraceConfigError("HERBERT:config_base:invalid_argument", mess)
        opt_buffer, opt_save = flags
        if opt_buffer and opt_save:
            raise HoraceConfigError(
                "HERBERT:config_base:invalid_argument",
                "Options '-buffer' and '-save' cannot be specified simultaneously",
            )
        if opt_buffer or opt_save:
            save_changes = opt_save
        else:
            save_changes = self.saveable

        change_to_default = False
        change_to_saved = False
        S: Dict[str, Any] = {}

        if kwargs and not rem:
            S = dict(kwargs)
        elif len(rem) == 1:
            arg0 = rem[0]
            if isinstance(arg0, dict):
                S = dict(arg0)
            elif isinstance(arg0, (list, tuple)):
                if len(arg0) % 2 != 0:
                    raise HoraceConfigError(
                        "HERBERT:config_base:invalid_argument",
                        "Cell/list of property/value pairs must have even length",
                    )
                for i in range(0, len(arg0), 2):
                    S[str(arg0[i])] = arg0[i + 1]
            elif isinstance(arg0, str) and arg0:
                low = arg0.lower()
                if "defaults".startswith(low):
                    change_to_default = True
                elif "saved".startswith(low):
                    change_to_saved = True
                else:
                    raise HoraceConfigError(
                        "HERBERT:config_base:invalid_argument",
                        f"Single option string must be 'defaults' or 'saved', got '{arg0}'",
                    )
            else:
                raise HoraceConfigError(
                    "HERBERT:config_base:invalid_argument",
                    f"Invalid argument to set(): {arg0!r}",
                )
        elif len(rem) > 1:
            if len(rem) % 2 != 0:
                raise HoraceConfigError(
                    "HERBERT:config_base:invalid_argument",
                    "Incomplete Pair: Arguments of set method have to be 'field_name', field_value pairs",
                )
            for i in range(0, len(rem), 2):
                S[str(rem[i])] = rem[i + 1]
            S.update(kwargs)

        if change_to_default:
            S = dict(self.get_defaults())
        elif change_to_saved:
            config_store.instance().clear_config(self)
            S = dict(config_store.instance().get_config(self))

        if S:
            prev_state = self.get_all(public_only=False)
            current_saveable = self.saveable
            self.saveable = False
            self.do_check_combo_arg = False
            self.disable_setup_warnings = True
            try:
                for fn, fv in S.items():
                    if not hasattr(type(self), fn) and not hasattr(self, fn):
                        raise HoraceConfigError(
                            "HERBERT:config_base:invalid_argument",
                            f"Unknown property '{fn}' on {self.class_name}",
                        )
                    setattr(self, fn, fv)
                self.do_check_combo_arg = True
                self.check_combo_arg()
                self.disable_setup_warnings = False
            except Exception:
                for fn, fv in prev_state.items():
                    config_store.instance().set_value(self, fn, fv)
                raise
            finally:
                self.do_check_combo_arg = True
                self.disable_setup_warnings = False
                self.saveable = current_saveable

            if save_changes:
                config_store.instance().store_config(self, "-forcesave")

        return self


def get(cfg: config_base, *args: Any) -> Any:
    """MATLAB-style `get(cfg, ...)` functional wrapper."""
    return cfg.get(*args)


def set(cfg: config_base, *args: Any, **kwargs: Any) -> config_base:
    """MATLAB-style `set(cfg, ...)` functional wrapper."""
    return cfg.set(*args, **kwargs)


__all__ = ["config_base", "get", "set"]
