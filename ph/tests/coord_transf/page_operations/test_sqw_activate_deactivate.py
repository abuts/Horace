"""Unit tests for ``sqw.activate``, ``sqw.deactivate``, and ``sqw.set_as_tmp_obj``.

Translated from ``_test/test_sqw_pageOpMethods/test_sqw_activate_deactivate.m``.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile

import pytest

from pyhorace.core.serializable import compare_values
from pyhorace.sqw import sqw


@pytest.fixture(scope="module")
def act_deact_env():
    sqww = sqw.generate_cube_sqw(10)
    work_dir = tempfile.mkdtemp(prefix="test_sqw_activation_")
    sqw_file = os.path.join(work_dir, "test_sqw_activation.sqw")
    sqww.save(sqw_file)
    yield {
        "sqw_obj": sqww,
        "sqw_file": sqw_file,
        "work_dir": work_dir,
    }
    shutil.rmtree(work_dir, ignore_errors=True)


class TestSqwActivateDeactivate:
    """Test suite translated from ``test_sqw_activate_deactivate.m``."""

    def test_deactivated_copied_tmp_file_deleted(self, act_deact_env):
        new_file = os.path.join(
            act_deact_env["work_dir"], "test_deact_copied_tmp_deleted.tmp_xxxx"
        )
        shutil.copyfile(act_deact_env["sqw_file"], new_file)

        sq_obj = sqw(new_file, file_backed=True)
        sq_obj = sq_obj.set_as_tmp_obj()
        sq2_obj = sq_obj.copy()
        sq_obj = sq_obj.deactivate()
        assert os.path.isfile(new_file)
        del sq_obj
        gc.collect()
        assert os.path.isfile(new_file)
        del sq2_obj
        gc.collect()
        assert not os.path.isfile(new_file)

    def test_copied_tmp_file_deleted(self, act_deact_env):
        new_file = os.path.join(
            act_deact_env["work_dir"], "test_copied_tmp_deleted.tmp_xxxx"
        )
        shutil.copyfile(act_deact_env["sqw_file"], new_file)

        sq_obj = sqw(new_file, file_backed=True)
        sq_obj = sq_obj.set_as_tmp_obj()
        sq2_obj = sq_obj.copy()
        assert os.path.isfile(new_file)
        del sq_obj
        gc.collect()
        assert os.path.isfile(new_file)
        del sq2_obj
        gc.collect()
        assert not os.path.isfile(new_file)

    def test_deactivated_activated_tmp_file_deleted(self, act_deact_env):
        new_file = os.path.join(
            act_deact_env["work_dir"], "test_deact_act_tmp_deleted.tmp_xxxx"
        )
        shutil.copyfile(act_deact_env["sqw_file"], new_file)

        sq_obj = sqw(new_file, file_backed=True)
        sq_obj = sq_obj.set_as_tmp_obj()
        sq_obj = sq_obj.deactivate()
        assert os.path.isfile(new_file)
        sq_obj = sq_obj.activate()
        del sq_obj
        gc.collect()
        assert not os.path.isfile(new_file)

    def test_deactivated_activated_non_tmp_file_kept(self, act_deact_env):
        new_file = os.path.join(
            act_deact_env["work_dir"], "test_deactivated_act_nontmp_kept.sqw"
        )
        shutil.copyfile(act_deact_env["sqw_file"], new_file)
        try:
            sq_obj = sqw(new_file, file_backed=True)
            sq_obj = sq_obj.set_as_tmp_obj()
            sq_obj = sq_obj.deactivate()
            assert os.path.isfile(new_file)
            sq_obj = sq_obj.activate()
            sq_obj.deactivate()
            del sq_obj
            gc.collect()
            assert os.path.isfile(new_file)
        finally:
            if os.path.isfile(new_file):
                try:
                    os.remove(new_file)
                except OSError:
                    pass

    def test_deactivated_file_kept(self, act_deact_env):
        new_file = os.path.join(
            act_deact_env["work_dir"], "test_deactivated_tmp_kept.sqw"
        )
        shutil.copyfile(act_deact_env["sqw_file"], new_file)
        try:
            sq_obj = sqw(new_file, file_backed=True)
            sq_obj = sq_obj.set_as_tmp_obj()
            sq_obj = sq_obj.deactivate()
            assert os.path.isfile(new_file)
            del sq_obj
            gc.collect()
            assert os.path.isfile(new_file)
        finally:
            if os.path.isfile(new_file):
                try:
                    os.remove(new_file)
                except OSError:
                    pass

    def test_based_file_deleted(self, act_deact_env):
        new_file = os.path.join(
            act_deact_env["work_dir"], "test_activation_tmp_deleted.sqw"
        )
        shutil.copyfile(act_deact_env["sqw_file"], new_file)

        sq_obj = sqw(new_file, file_backed=True)
        sq_obj = sq_obj.set_as_tmp_obj()
        assert os.path.isfile(new_file)
        del sq_obj
        gc.collect()
        assert not os.path.isfile(new_file)

    def test_sqw_activate_deactivate_does_nothing_in_mem(self, act_deact_env):
        sqw_obj = act_deact_env["sqw_obj"]
        sdec = sqw_obj.deactivate()
        sdec = sdec.activate()
        ok, mess = compare_values(sdec, sqw_obj)
        assert ok, mess
