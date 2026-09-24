"""Test mock classes translated from MATLAB _test/test_serializers/*.m."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.serializable import Serializable, register_serializable


@register_serializable
class SerializableTester1(Serializable):
    """Mock class 1 supporting version changes and custom from_old_struct."""

    _class_version_override: Optional[int] = None
    _fields_to_save: Sequence[str] = (
        "Prop_class1_1",
        "Prop_class1_2",
        "Prop_class1_3",
    )

    def __init__(
        self,
        prop1: Any = 10,
        prop2: Any = 20,
        prop3: Any = "new_value",
    ) -> None:
        super().__init__()
        self.Prop_class1_1 = prop1
        self.Prop_class1_2 = prop2
        self.Prop_class1_3 = prop3

    @classmethod
    def ver_holder(cls, new_version: Optional[int] = None) -> int:
        if new_version is not None:
            cls._class_version_override = new_version
        if cls._class_version_override is None:
            cls._class_version_override = 2
        return cls._class_version_override

    @classmethod
    def version_holder(cls, new_version: Optional[int] = None) -> int:
        return cls.ver_holder(new_version)

    def class_version(self) -> int:
        return self.ver_holder()

    def saveable_fields(self) -> Sequence[str]:
        if self.class_version() == 1:
            return self._fields_to_save[:2]
        return self._fields_to_save

    def from_old_struct(
        self, S: Union[Dict[str, Any], Sequence[Dict[str, Any]]]
    ) -> Union[Serializable, List[Serializable]]:
        if isinstance(S, dict):
            ver = S.get("version", None)
            if ver != 2:
                dat = S.get("array_dat", S)
                if isinstance(dat, (list, tuple)):
                    recovered = [SerializableTester1().from_bare_struct(item) for item in dat]
                    for item in recovered:
                        item.Prop_class1_3 = "recovered_new_from_old_value"
                    return recovered
                else:
                    recovered = self.from_bare_struct(dat)
                    if isinstance(recovered, list):
                        for item in recovered:
                            item.Prop_class1_3 = "recovered_new_from_old_value"
                    else:
                        recovered.Prop_class1_3 = "recovered_new_from_old_value"
                    return recovered

        return super().from_old_struct(S)


@register_serializable
class SerializableTester2(Serializable):
    """Mock class 2 using set_positional_and_key_val_arguments."""

    _version_override: int = 1
    _fields_to_save: Sequence[str] = (
        "Prop_class2_1",
        "Prop_class2_2",
        "Prop_class2_3",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.Prop_class2_1: Any = None
        self.Prop_class2_2: Any = None
        self.Prop_class2_3: Any = None
        self.remains: List[Any] = []

        if len(args) > 0 or len(kwargs) > 0:
            pos_names = self.saveable_fields()
            _, self.remains = self.set_positional_and_key_val_arguments(
                pos_names, False, *args, **kwargs
            )

    @classmethod
    def version_holder(cls, ver: Optional[int] = None) -> int:
        if ver is not None:
            cls._version_override = ver
        return cls._version_override

    def class_version(self) -> int:
        return self.version_holder()

    def saveable_fields(self) -> Sequence[str]:
        return self._fields_to_save


def make_tester2(*args: Any, **kwargs: Any) -> Tuple[SerializableTester2, List[Any]]:
    """Helper returning (obj, remains) matching MATLAB [tc, rem] = serializableTester2(...)."""
    obj = SerializableTester2(*args, **kwargs)
    return obj, obj.remains


@register_serializable
class SerializableTester3(Serializable):
    """
    Mock class 3 based on IX_det_He3tube.
    Exercises dependent properties, check_combo_arg, and convert_old_struct.
    """

    _version_override: int = 2

    def __init__(
        self,
        dia: Any = None,
        height: Any = None,
        wall: Any = None,
        atms: Any = None,
    ) -> None:
        super().__init__()
        if dia is not None and height is not None and wall is not None and atms is not None:
            self.do_check_combo_arg = False
            self.dia = dia
            self.height = height
            self.wall = wall
            self.atms = atms
            self.do_check_combo_arg = True
            self.check_combo_arg()
        else:
            ver = self.version_holder()
            if ver == 2:
                self.dia = 0.1
                self.height = 0.3
                self.wall = 0.01
                self.atms = 1.0
            elif ver == 1:
                self.dia = 0.1
                self.height = 0.3
                self.wall = 0.02
                self.atms = 1.0
            else:
                self.dia = 0.1
                self.height = 0.3
                self.wall = 0.03
                self.atms = 1.0

    @classmethod
    def version_holder(cls, ver: Optional[int] = None) -> int:
        if ver is not None:
            cls._version_override = ver
        return cls._version_override

    def class_version(self) -> int:
        return self.version_holder()

    def saveable_fields(self) -> Sequence[str]:
        ver = self.class_version()
        if ver == 2:
            return ["dia", "height", "wall", "atms"]
        elif ver == 1:
            return ["dia", "height", "atms"]
        else:
            return ["dia", "height"]

    @property
    def inner_rad(self) -> Any:
        return 0.5 * (np.asarray(self.dia) - 2 * np.asarray(self.wall))

    @property
    def ndet(self) -> int:
        return int(np.size(self.dia))

    def check_combo_arg(self) -> "SerializableTester3":
        dia_arr = np.asarray(self.dia)
        wall_arr = np.asarray(self.wall)
        if np.any(dia_arr < 2 * wall_arr):
            raise ValueError("Tube diameter(s) must be greater or equal to twice the wall thickness(es)")
        return self

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "Serializable"]:
        S_updated = dict(S)
        if ver == 1:
            S_updated["wall"] = 1e-6
        elif math.isnan(ver):
            S_updated["wall"] = 1e-7
            S_updated["atms"] = 27.0
        else:
            raise ValueError(f"Unrecognised class version: {ver}")
        return S_updated, self


@register_serializable
class SerializableTesterWithInterdepProp(Serializable):
    """Mock class testing interdependent properties and partial keyword matching."""

    _fields_to_save: Sequence[str] = (
        "Prop_class2_1",
        "Prop_class2_2",
        "Prop_class2_3",
        "partial_match_1_blue",
        "partial_match_2_green",
        "partial_match_3_yellow",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._Prop_class2_1: Any = None
        self._Prop_class2_2: Any = None
        self._Prop_class2_3: Any = None
        self.partial_match_1_blue: Any = None
        self.partial_match_2_green: Any = None
        self.partial_match_3_yellow: Any = None
        self.remains: List[Any] = []

        if len(args) > 0 or len(kwargs) > 0:
            pos_names = self.saveable_fields()
            _, self.remains = self.set_positional_and_key_val_arguments(
                pos_names, True, *args, **kwargs
            )

    @property
    def Prop_class2_1(self) -> Any:
        return self._Prop_class2_1

    @Prop_class2_1.setter
    def Prop_class2_1(self, val: Any) -> None:
        self._Prop_class2_1 = val
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def Prop_class2_2(self) -> Any:
        return self._Prop_class2_2

    @Prop_class2_2.setter
    def Prop_class2_2(self, val: Any) -> None:
        self._Prop_class2_2 = val
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def Prop_class2_3(self) -> Any:
        return self._Prop_class2_3

    @Prop_class2_3.setter
    def Prop_class2_3(self, val: Any) -> None:
        self._Prop_class2_3 = val
        if self.do_check_combo_arg:
            self.check_combo_arg()

    def check_combo_arg(self) -> "SerializableTesterWithInterdepProp":
        p1 = self.Prop_class2_1
        p2 = self.Prop_class2_2
        p3 = self.Prop_class2_3

        if (
            (p1 is not None and p2 is None)
            or (p3 is None)
            or (p1 is not None and p2 is not None and p1 > p2)
        ):
            raise ValueError("inconsistent interdependent properties")
        return self

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Sequence[str]:
        return self._fields_to_save


@register_serializable
class SerializableTester4SetKeyValConstructor(Serializable):
    """Mock class testing string validation and dash options in set_positional_and_key_val_arguments."""

    _fields_to_save: Sequence[str] = ("prop1_char", "prop2_char", "prop3_char")

    def __init__(self, old_keyval_compat: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._prop1_char: str = ""
        self._prop2_char: str = ""
        self._prop3_char: str = ""

        pos_names = self.saveable_fields()
        _, remains = self.set_positional_and_key_val_arguments(
            pos_names, old_keyval_compat, *args, **kwargs
        )
        if len(remains) > 0:
            raise ValueError(f"unrecognized property provided as input: {remains}")

    @property
    def prop1_char(self) -> str:
        return self._prop1_char

    @prop1_char.setter
    def prop1_char(self, val: Any) -> None:
        if not isinstance(val, str):
            raise TypeError("This property accepts only char value")
        self._prop1_char = val

    @property
    def prop2_char(self) -> str:
        return self._prop2_char

    @prop2_char.setter
    def prop2_char(self, val: Any) -> None:
        if not isinstance(val, str):
            raise TypeError("This property accepts only char value")
        self._prop2_char = val

    @property
    def prop3_char(self) -> str:
        return self._prop3_char

    @prop3_char.setter
    def prop3_char(self, val: Any) -> None:
        if not isinstance(val, str):
            raise TypeError("This property accepts only char value")
        self._prop3_char = val

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Sequence[str]:
        return self._fields_to_save
