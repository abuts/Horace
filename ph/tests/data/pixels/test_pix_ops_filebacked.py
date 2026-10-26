"""File-backed multi-page pixel operations test runner.

Translated from `_test/test_sqw_pixels/test_pix_ops_filebacked.m`
(uses small `page_size = 11` so 30-pixel test objects span 3 pages).
"""

from __future__ import annotations

from tests.data.pixels.test_pix_ops_in_memory import PixOpsCommonSuite


class TestPixOpsFileBacked(PixOpsCommonSuite):
    """Run PixOpsCommonSuite with file-backed PixelDataFileBacked (`page_size = 11`)."""

    file_backed = True
    page_size = 11

