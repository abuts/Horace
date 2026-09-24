"""Serializable base class for versioned serialization, validation, and flexible construction.

Ported from Herbert/Horace MATLAB @serializable class.
"""

from __future__ import annotations

import copy
import math
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

SERIALIZABLE_REGISTRY: Dict[str, Type["Serializable"]] = {}


def register_serializable(cls: Type["Serializable"]) -> Type["Serializable"]:
    """Register a Serializable subclass for deserialization lookup."""
    SERIALIZABLE_REGISTRY[cls.__name__] = cls
    return cls


class Serializable(ABC):
    """
    Abstract base class for versioned serialization, property validation,
    and flexible construction matching MATLAB's @serializable.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._do_check_combo_arg: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        SERIALIZABLE_REGISTRY[cls.__name__] = cls

    @property
    def do_check_combo_arg(self) -> bool:
        """Flag controlling interdependent property validation."""
        return getattr(self, "_do_check_combo_arg", True)

    @do_check_combo_arg.setter
    def do_check_combo_arg(self, val: bool) -> None:
        self._do_check_combo_arg = bool(val)

    @property
    def do_check_combo_arg_(self) -> bool:
        """Alias matching MATLAB protected property name."""
        return self.do_check_combo_arg

    @do_check_combo_arg_.setter
    def do_check_combo_arg_(self, val: bool) -> None:
        self.do_check_combo_arg = val

    @abstractmethod
    def class_version(self) -> int:
        """Return integer class version number."""
        pass

    def classVersion(self) -> int:
        """MATLAB camelCase alias for class_version."""
        return self.class_version()

    @abstractmethod
    def saveable_fields(self) -> Sequence[str]:
        """Return list of property names that fully define the state."""
        pass

    def saveableFields(self) -> Sequence[str]:
        """MATLAB camelCase alias for saveable_fields."""
        return self.saveable_fields()

    def check_combo_arg(self) -> "Serializable":
        """
        Validate interdependent properties. Subclasses override to enforce invariants.
        Should return self.
        """
        return self

    # -------------------------------------------------------------------------
    # Structure Conversion (to_bare_struct / to_struct)
    # -------------------------------------------------------------------------

    def to_bare_struct(
        self, recursive_bare: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Convert object properties in saveable_fields to a bare dictionary.

        If recursive_bare is False, nested Serializable properties are converted
        using to_struct() (containing class name and version metadata).
        If recursive_bare is True, nested Serializable properties are converted
        using to_bare_struct().
        """
        flds = self.saveable_fields()
        res: Dict[str, Any] = {}
        for fld in flds:
            val = getattr(self, fld, None)
            res[fld] = self._convert_value_to_bare(val, recursive_bare)
        return res

    @staticmethod
    def _convert_value_to_bare(val: Any, recursive_bare: bool) -> Any:
        if isinstance(val, Serializable):
            return val.to_bare_struct(recursive_bare) if recursive_bare else val.to_struct()
        elif isinstance(val, list):
            return [Serializable._convert_value_to_bare(item, recursive_bare) for item in val]
        elif isinstance(val, tuple):
            return tuple(Serializable._convert_value_to_bare(item, recursive_bare) for item in val)
        elif isinstance(val, np.ndarray) and val.dtype == object:
            converted = [Serializable._convert_value_to_bare(item, recursive_bare) for item in val.flat]
            return np.array(converted, dtype=object).reshape(val.shape)
        return val

    def to_struct(self) -> Dict[str, Any]:
        """
        Convert object to a structured dictionary including 'serial_name' and 'version'.
        """
        flds = self.saveable_fields()
        if "serial_name" in flds or "version" in flds:
            raise ValueError(
                "The input object cannot have properties with the protected names: 'serial_name' and 'version'"
            )

        bare = self.to_bare_struct(recursive_bare=False)
        assert isinstance(bare, dict)
        res: Dict[str, Any] = {
            "serial_name": self.__class__.__name__,
            "version": self.class_version(),
        }
        res.update(bare)
        return res

    @classmethod
    def to_struct_array(
        cls, objs: Sequence["Serializable"]
    ) -> Dict[str, Any]:
        """Convert a sequence/array of Serializable objects to a struct representation with array_dat."""
        if len(objs) == 0:
            raise ValueError("Cannot serialize empty array of objects without type information")
        first = objs[0]
        return {
            "serial_name": first.__class__.__name__,
            "version": first.class_version(),
            "array_dat": [o.to_bare_struct(recursive_bare=False) for o in objs],
        }

    # Aliases
    to_bare_dict = to_bare_struct
    to_dict = to_struct

    # -------------------------------------------------------------------------
    # Structure Restoration (from_bare_struct / from_struct)
    # -------------------------------------------------------------------------

    def from_bare_struct(
        self, S: Union[Dict[str, Any], Sequence[Dict[str, Any]]]
    ) -> Union["Serializable", List["Serializable"]]:
        """
        Populate object (or return list of objects) from bare dictionary/dictionaries.
        """
        if isinstance(S, (list, tuple, np.ndarray)):
            if len(S) == 0:
                return self
            # Sequence of structs -> sequence of objects
            result = []
            for item in S:
                obj_item = copy.copy(self)
                obj_item = obj_item._populate_single_bare(item)
                result.append(obj_item)
            return result

        if not isinstance(S, dict):
            raise TypeError(f"Expected dict or sequence of dicts, got {type(S).__name__}")

        if len(S) == 0:
            return self

        return self._populate_single_bare(S)

    def _populate_single_bare(self, S: Dict[str, Any]) -> "Serializable":
        flds_to_set = set(self.saveable_fields())
        flds_present = set(S.keys())
        common_flds = flds_to_set.intersection(flds_present)
        if not common_flds:
            return self

        old_combo_check = self.do_check_combo_arg
        try:
            self.do_check_combo_arg = False
            for fld in self.saveable_fields():
                if fld in S:
                    val = S[fld]
                    val = self._restore_value_from_struct(val)
                    setattr(self, fld, val)
        finally:
            self.do_check_combo_arg = old_combo_check

        if self.do_check_combo_arg:
            self.check_combo_arg()
        return self

    @classmethod
    def _restore_value_from_struct(cls, val: Any) -> Any:
        if isinstance(val, dict):
            if "serial_name" in val:
                return cls.from_struct(val)
            else:
                return {k: cls._restore_value_from_struct(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [cls._restore_value_from_struct(item) for item in val]
        elif isinstance(val, tuple):
            return tuple(cls._restore_value_from_struct(item) for item in val)
        return val

    @classmethod
    def from_struct(
        cls,
        S: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        obj_template: Optional["Serializable"] = None,
    ) -> Union["Serializable", List["Serializable"]]:
        """
        Restore an object or list of objects from a structured dictionary created by to_struct.
        """
        if isinstance(S, (list, tuple, np.ndarray)):
            if obj_template is None:
                raise ValueError("Array of bare structures requires obj_template to determine class")
            return obj_template.from_old_struct(S)

        if not isinstance(S, dict):
            raise TypeError(f"Expected dict, got {type(S).__name__}")

        # Resolve instance / class
        if obj_template is None:
            if "serial_name" not in S:
                raise ValueError(
                    'Class has not been converted into a structure using serializable '
                    'class "to_struct" operation. This method cannot restore the class'
                )
            class_name = S["serial_name"]
            target_cls = SERIALIZABLE_REGISTRY.get(class_name)
            if target_cls is None:
                for sub in cls._all_subclasses():
                    if sub.__name__ == class_name:
                        target_cls = sub
                        SERIALIZABLE_REGISTRY[class_name] = sub
                        break
            if target_cls is None:
                raise ValueError(f"Unknown serializable class: '{class_name}'")
            obj = target_cls()
        else:
            obj = copy.copy(obj_template)

        # Check version
        version = S.get("version")
        if version is not None and version == obj.class_version():
            if "array_dat" in S:
                return obj.from_bare_struct(S["array_dat"])
            else:
                return obj.from_bare_struct(S)
        else:
            # Different or missing version -> invoke from_old_struct
            return obj.from_old_struct(S)

    @classmethod
    def _all_subclasses(cls) -> set[Type["Serializable"]]:
        subclasses = set(cls.__subclasses__())
        for sub in list(subclasses):
            subclasses.update(sub._all_subclasses())
        return subclasses

    # Aliases
    from_bare_dict = from_bare_struct
    from_dict = from_struct

    # -------------------------------------------------------------------------
    # Old Version Migration
    # -------------------------------------------------------------------------

    def from_old_struct(
        self, S: Union[Dict[str, Any], Sequence[Dict[str, Any]]]
    ) -> Union["Serializable", List["Serializable"]]:
        """
        Restore object or object array from structure created by prior class versions.
        """
        if isinstance(S, dict):
            ver = S.get("version", float("nan"))
            dat = S.get("array_dat", S)
        else:
            ver = float("nan")
            dat = S

        if isinstance(dat, (list, tuple, np.ndarray)):
            result = []
            for item in dat:
                s_updated, _ = self.convert_old_struct(item, ver)
                item_obj = copy.copy(self)
                item_obj = item_obj.from_bare_struct(s_updated)
                result.append(item_obj)
            return result
        else:
            s_updated, _ = self.convert_old_struct(dat, ver)
            return self.from_bare_struct(s_updated)

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "Serializable"]:
        """
        Update the structure created from earlier class versions to the current version.
        Default returns S unaltered. Subclasses override to provide custom migration.
        """
        return S, self

    # -------------------------------------------------------------------------
    # Byte Serialization (serialize / deserialize / serial_size)
    # -------------------------------------------------------------------------

    def serialize(self) -> bytes:
        """Serialize object to bytes via to_struct."""
        S = self.to_struct()
        return pickle.dumps(S, protocol=5)

    @classmethod
    def serialize_array(cls, objs: Sequence["Serializable"]) -> bytes:
        """Serialize sequence of objects to bytes."""
        S = cls.to_struct_array(objs)
        return pickle.dumps(S, protocol=5)

    def serial_size(self) -> int:
        """Return byte length of serialized object."""
        return len(self.serialize())

    @classmethod
    def deserialize(
        cls, byte_array: bytes, pos: int = 0
    ) -> Tuple[Union["Serializable", List["Serializable"]], int]:
        """
        Recover object or list of objects from serialized byte array.
        Returns (obj_or_objs, nbytes).
        """
        sub_bytes = byte_array[pos:]
        S = pickle.loads(sub_bytes)
        nbytes = len(sub_bytes)
        obj = cls.from_struct(S)
        return obj, nbytes

    # -------------------------------------------------------------------------
    # Saveobj / Loadobj
    # -------------------------------------------------------------------------

    def saveobj(self) -> Dict[str, Any]:
        """Return dictionary representation for saving."""
        return self.to_struct()

    @classmethod
    def loadobj(
        cls, S: Dict[str, Any], obj_template: Optional["Serializable"] = None
    ) -> Union["Serializable", List["Serializable"]]:
        """Load object from dictionary representation."""
        if obj_template is None:
            obj_template = cls()
        return cls.from_struct(S, obj_template=obj_template)

    # -------------------------------------------------------------------------
    # Constructor Argument Parser
    # -------------------------------------------------------------------------

    def set_positional_and_key_val_arguments(
        self,
        positional_arg_names: Sequence[str],
        options: Union[bool, Dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple["Serializable", List[Any]]:
        """
        Flexible constructor argument parser supporting:
        - Positional arguments mapping to positional_arg_names
        - Key-value pairs (with partial prefix matching against positional_arg_names)
        - Optional '-' prefix for keys with deprecation warning
        - Returns (self, remains)
        """
        if isinstance(options, bool):
            support_dash_option = options
            mandatory_properties: Optional[List[bool]] = None
        elif isinstance(options, dict):
            support_dash_option = options.get("key_dash", False)
            mandatory_properties = options.get("mandatory_props", None)
        else:
            support_dash_option = False
            mandatory_properties = None

        # Build list of all input tokens
        input_args = list(args)
        for k, v in kwargs.items():
            input_args.extend([k, v])

        if len(input_args) == 0:
            return self, []

        self.do_check_combo_arg = False

        try:
            (
                remains,
                key_pos,
                val_pos,
                is_positional,
                argi,
            ) = self._parse_keyval_argi(
                positional_arg_names, support_dash_option, input_args
            )

            n_positional_arg_names = len(positional_arg_names)
            n_positional = sum(is_positional)

            if n_positional > n_positional_arg_names:
                if all(is_positional):
                    pos_remains = list(is_positional)
                    for j in range(n_positional_arg_names):
                        pos_remains[j] = False
                    remains = [
                        argi[idx] for idx, is_pos in enumerate(pos_remains) if is_pos
                    ] + remains
                    argi = [
                        argi[idx]
                        for idx, is_pos in enumerate(is_positional)
                        if not pos_remains[idx]
                    ]
                    n_positional = n_positional_arg_names
                else:
                    raise ValueError(
                        f"More positional arguments identified: ({n_positional}) than "
                        f"positional values allowed: ({n_positional_arg_names}). "
                        f"Looks like some keys from key-value pairs have been identified as property values"
                    )

            # Check mandatory properties
            if mandatory_properties is not None:
                property_set = [False] * n_positional_arg_names
                for i in range(n_positional):
                    property_set[i] = True
                for kp in key_pos:
                    k_name = argi[kp]
                    if k_name in positional_arg_names:
                        idx = positional_arg_names.index(k_name)
                        property_set[idx] = True

                for i, is_mand in enumerate(mandatory_properties):
                    if is_mand and not property_set[i]:
                        raise ValueError(
                            f"One or more class {self.__class__.__name__} constructor mandatory properties "
                            f"have not been provided: '{positional_arg_names[i]}'"
                        )

            # Assign positional arguments
            if any(is_positional):
                pos_vals = [argi[idx] for idx, is_pos in enumerate(is_positional) if is_pos]
                for i in range(min(n_positional, len(pos_vals))):
                    setattr(self, positional_arg_names[i], pos_vals[i])

            # Assign keyword-value arguments
            for kp, vp in zip(key_pos, val_pos):
                setattr(self, argi[kp], argi[vp])

        finally:
            self.do_check_combo_arg = True

        self.check_combo_arg()
        return self, remains

    def _parse_keyval_argi(
        self,
        arg_names: Sequence[str],
        support_dash_option: bool,
        args_list: List[Any],
    ) -> Tuple[List[Any], List[int], List[int], List[bool], List[Any]]:
        (
            is_key,
            deprecated_fields,
            argi,
        ) = self._is_char_key_member(arg_names, support_dash_option, args_list)

        if support_dash_option and deprecated_fields:
            warnings.warn(
                f"Some class {self.__class__.__name__} constructor key-value inputs have been provided "
                f"with '-' prefix: {deprecated_fields}. This syntax is deprecated. "
                f"Provide these keys without '-' prefix.",
                category=DeprecationWarning,
                stacklevel=3,
            )

        n = len(args_list)
        is_positional = [not k for k in is_key]

        if any(is_key):
            first_key = is_key.index(True)
            for j in range(first_key, n):
                is_positional[j] = False

            key_pos = [i for i, k in enumerate(is_key) if k]
            val_pos = [i + 1 for i in key_pos]

            if len(val_pos) > 0 and (val_pos[-1] >= n or any(p in is_key for p in val_pos if is_key[p])):
                raise ValueError(
                    "should be even number of key-value pairs, but some keys do not have corresponding value argument"
                )

            consumed = [False] * n
            for p in is_positional:
                pass
            for i, is_pos in enumerate(is_positional):
                if is_pos:
                    consumed[i] = True
            for kp, vp in zip(key_pos, val_pos):
                consumed[kp] = True
                consumed[vp] = True

            remains = [args_list[i] for i in range(n) if not consumed[i]]
        else:
            key_pos = []
            val_pos = []
            remains = []

        return remains, key_pos, val_pos, is_positional, argi

    def _is_char_key_member(
        self,
        key_list: Sequence[str],
        support_dash_option: bool,
        args_list: List[Any],
    ) -> Tuple[List[bool], List[str], List[Any]]:
        n = len(args_list)
        is_key = [False] * n
        is_deprecated = [False] * n
        deprecated_fields = []
        argi = list(args_list)

        in_pos_parameters = True
        prev_input_is_key = False

        for i, arg in enumerate(args_list):
            min_comp_base = 4
            if in_pos_parameters:
                in_pos_parameters = i < len(key_list)

            if not isinstance(arg, str):
                prev_input_is_key = False
                continue

            if prev_input_is_key:
                prev_input_is_key = False
                continue

            arg_str = arg
            if in_pos_parameters:
                curr_val = getattr(self, key_list[i], None)
                if isinstance(curr_val, str):
                    if arg_str.startswith("-"):
                        arg_str = arg_str[1:]
                    else:
                        min_comp_base = math.inf

            # Match against key_list
            matches = []
            matches_depr = []
            for k in key_list:
                m = self._compare_par(arg_str, k, min_comp_base)
                matches.append(m)
                if support_dash_option:
                    m_dep = self._compare_par(arg, f"-{k}", min_comp_base + 1)
                    matches_depr.append(m_dep)
                else:
                    matches_depr.append(False)

            if support_dash_option:
                any_depr = [d for d in matches_depr if d]
                if any_depr:
                    is_deprecated[i] = True
                combined_matches = [m or d for m, d in zip(matches, matches_depr)]
            else:
                combined_matches = matches

            found = sum(combined_matches)
            if found > 1:
                matched_names = [k for k, m in zip(key_list, combined_matches) if m]
                raise ValueError(
                    f"Input key N{i+1} ({arg}) can non-uniquely define more then one possible property: "
                    f"{matched_names}. Can not interpret this key"
                )
            elif found == 1:
                if prev_input_is_key:
                    matched_k = [k for k, m in zip(key_list, combined_matches) if m][0]
                    raise ValueError(
                        f"Two adjacent input parameters N{i} and N{i+1} are identified as keys: "
                        f"({argi[i-1]} and {matched_k}). Something is wrong"
                    )
                prev_input_is_key = True
                in_pos_parameters = False
                is_key[i] = True
                matched_name = [k for k, m in zip(key_list, combined_matches) if m][0]
                argi[i] = matched_name
                if support_dash_option and is_deprecated[i]:
                    deprecated_fields.append(arg)
            else:
                prev_input_is_key = False

        return is_key, deprecated_fields, argi

    @staticmethod
    def _compare_par(par: str, key: str, min_comp: float) -> bool:
        if math.isinf(min_comp):
            comp_base = len(key)
        else:
            comp_base = min(int(min_comp), len(key))
        if len(par) < comp_base:
            return False
        return key.startswith(par)

    # -------------------------------------------------------------------------
    # Equality and Tolerance Checking
    # -------------------------------------------------------------------------

    def equal_to_tol(
        self,
        other: Any,
        tol: Union[float, Sequence[float]] = (0.0, 0.0),
        nan_equal: bool = True,
        ignore_str: bool = False,
        name_a: str = "obj1",
        name_b: str = "obj2",
    ) -> Tuple[bool, str]:
        """
        Check equality of two Serializable objects within tolerance across all saveable_fields.
        Returns (is_equal, message).
        """
        if type(self) is not type(other):
            return (
                False,
                f"Objects are of different classes: '{type(self).__name__}' and '{type(other).__name__}'",
            )

        flds = self.saveable_fields()
        for fld in flds:
            val1 = getattr(self, fld, None)
            val2 = getattr(other, fld, None)
            fld_name_a = f"{name_a}.{fld}"
            fld_name_b = f"{name_b}.{fld}"

            if ignore_str and isinstance(val1, str) and isinstance(val2, str):
                continue

            iseq, mess = compare_values(
                val1,
                val2,
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=fld_name_a,
                name_b=fld_name_b,
            )
            if not iseq:
                return False, mess

        return True, ""

    def eq(self, other: Any) -> bool:
        """MATLAB eq method alias."""
        return self.__eq__(other)

    def ne(self, other: Any) -> bool:
        """MATLAB ne method alias."""
        return self.__ne__(other)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Serializable):
            return False
        iseq, _ = self.equal_to_tol(other)
        return iseq

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


def compare_values(
    val1: Any,
    val2: Any,
    tol: Union[float, Sequence[float]] = (0.0, 0.0),
    nan_equal: bool = True,
    ignore_str: bool = False,
    name_a: str = "a",
    name_b: str = "b",
) -> Tuple[bool, str]:
    """
    Compare two values (scalar, array, Serializable, list, dict) with tolerance and nan_equal.
    """
    if isinstance(tol, (int, float)):
        atol = float(tol) if tol >= 0 else 0.0
        rtol = abs(float(tol)) if tol < 0 else 0.0
    else:
        atol = float(tol[0]) if len(tol) > 0 else 0.0
        rtol = float(tol[1]) if len(tol) > 1 else 0.0

    # None comparison
    if val1 is None or val2 is None:
        if val1 is None and val2 is None:
            return True, ""
        return False, f"{name_a} and {name_b} differ: one is None and the other is not"

    # Serializable comparison
    if isinstance(val1, Serializable) and isinstance(val2, Serializable):
        return val1.equal_to_tol(
            val2,
            tol=tol,
            nan_equal=nan_equal,
            ignore_str=ignore_str,
            name_a=name_a,
            name_b=name_b,
        )

    # String comparison
    if isinstance(val1, str) and isinstance(val2, str):
        if ignore_str or val1 == val2:
            return True, ""
        return False, f"{name_a} and {name_b} strings differ: '{val1}' != '{val2}'"

    # Lists / Tuples
    if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
        if len(val1) != len(val2):
            return False, f"{name_a} and {name_b}: Lengths differ ({len(val1)} vs {len(val2)})"
        for i, (item1, item2) in enumerate(zip(val1, val2)):
            iseq, mess = compare_values(
                item1,
                item2,
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=f"{name_a}[{i}]",
                name_b=f"{name_b}[{i}]",
            )
            if not iseq:
                return False, mess
        return True, ""

    # Dictionaries
    if isinstance(val1, dict) and isinstance(val2, dict):
        if set(val1.keys()) != set(val2.keys()):
            return False, f"{name_a} and {name_b}: Dictionary keys differ"
        for k in val1:
            iseq, mess = compare_values(
                val1[k],
                val2[k],
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=f"{name_a}['{k}']",
                name_b=f"{name_b}['{k}']",
            )
            if not iseq:
                return False, mess
        return True, ""

    # Convert numeric types / lists to numpy arrays if either is numpy array
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        arr1 = np.asarray(val1)
        arr2 = np.asarray(val2)

        if arr1.shape != arr2.shape:
            return False, f"{name_a} and {name_b}: Array shapes differ: {arr1.shape} vs {arr2.shape}"

        if arr1.dtype == object or arr2.dtype == object:
            for idx, (x, y) in enumerate(zip(arr1.flat, arr2.flat)):
                iseq, mess = compare_values(
                    x,
                    y,
                    tol=tol,
                    nan_equal=nan_equal,
                    ignore_str=ignore_str,
                    name_a=f"{name_a}[{idx}]",
                    name_b=f"{name_b}[{idx}]",
                )
                if not iseq:
                    return False, mess
            return True, ""

        if np.issubdtype(arr1.dtype, np.number) and np.issubdtype(arr2.dtype, np.number):
            diff = np.abs(arr1 - arr2)
            if nan_equal:
                nan_mask1 = np.isnan(arr1)
                nan_mask2 = np.isnan(arr2)
                if not np.array_equal(nan_mask1, nan_mask2):
                    return False, f"{name_a} and {name_b}: NaN patterns differ"
                valid_mask = ~nan_mask1
            else:
                valid_mask = np.ones(arr1.shape, dtype=bool)

            denom = np.maximum(np.abs(arr1), np.abs(arr2))
            allowed_err = atol + rtol * denom
            violating = valid_mask & (diff > allowed_err)

            if np.any(violating):
                max_err = float(np.max(diff[violating]))
                flat_idx = int(np.argmax(diff * violating))
                # 1-indexed to match MATLAB message format
                return (
                    False,
                    f"{name_a} and {name_b}: Not all elements are equal; max. error = {max_err:g} at element ({flat_idx + 1})",
                )
            return True, ""
        else:
            if not np.array_equal(arr1, arr2):
                return False, f"{name_a} and {name_b}: Array elements differ"
            return True, ""

    # Scalar numbers
    if isinstance(val1, (int, float, complex, np.number)) and isinstance(
        val2, (int, float, complex, np.number)
    ):
        v1 = float(val1)
        v2 = float(val2)
        if nan_equal and math.isnan(v1) and math.isnan(v2):
            return True, ""
        if math.isnan(v1) or math.isnan(v2):
            return False, f"{name_a} and {name_b}: One value is NaN"

        err = abs(v1 - v2)
        denom = max(abs(v1), abs(v2))
        allowed_err = atol + rtol * denom
        if err > allowed_err:
            return (
                False,
                f"{name_a} and {name_b}: Not all elements are equal; max. error = {err:g} at element (1)",
            )
        return True, ""

    # Direct fallback comparison
    if val1 != val2:
        return False, f"{name_a} and {name_b} differ: {val1} != {val2}"

    return True, ""
