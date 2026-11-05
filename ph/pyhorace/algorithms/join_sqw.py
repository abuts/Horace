"""Algorithms for joining and combining SQW objects and files onto a common grid.

Ported from:
- ``horace_core/sqw/@sqw/join.m``
- ``horace_core/algorithms/collect_sqw_metadata.m``
- ``horace_core/algorithms/write_nsqw_to_sqw.m``
- ``horace_core/algorithms/check_img_consistency.m``
- ``horace_core/algorithms/private/parse_additional_input4_join_sqw_.m``
- ``horace_core/algorithms/private/get_pix_comb_info_.m``
- ``horace_core/algorithms/private/combine_exper_and_img_.m``
- ``horace_core/algorithms/private/check_img_consistency_.m``
- ``horace_core/algorithms/private/build_combined_img_.m``
- ``horace_core/sqw/@accumulate_headers_job/private/read_input_headers_.m``
- ``horace_core/utilities/normalize_signal.m``
"""

from __future__ import annotations

import datetime
import inspect
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.page_operations._utils import (
    minmax_ranges,
    parse_char_options,
)
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.configurations.hpc_config import hpc_config
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data.instr_classes import IX_detector_array
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixfile_combine_info import pixfile_combine_info
from pyhorace.data.pixels.pixobj_combine_info import pixobj_combine_info
from pyhorace.sqw.dnd_base import DnDBase


def _infer_nargout(default: int = 1, explicit_nargout: Optional[int] = None) -> int:
    """Infer expected number of return values from caller bytecode or explicit override."""
    if explicit_nargout is not None:
        return int(explicit_nargout)
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return default
    caller = frame.f_back.f_back
    try:
        import dis

        instructions = list(dis.get_instructions(caller.f_code))
        lasti = caller.f_lasti
        for idx, instr in enumerate(instructions):
            if instr.offset == lasti:
                for nxt in instructions[idx + 1 : idx + 5]:
                    if nxt.opname == "UNPACK_SEQUENCE":
                        return int(nxt.argval)
                    if nxt.opname in ("STORE_FAST", "STORE_NAME", "POP_TOP", "RETURN_VALUE"):
                        return 1
                break
    except Exception:
        pass
    return default


def _is_text(x: Any) -> bool:
    return isinstance(x, (str, os.PathLike))


def normalize_signal(
    s: np.ndarray, e: np.ndarray, npix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert accumulated signal and error into bin-average signal and error.

    Ported from ``horace_core/utilities/normalize_signal.m``.
    """
    npix_f = np.asarray(npix, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    e_arr = np.asarray(e, dtype=np.float64)

    no_pix = npix_f == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        s_out = np.where(no_pix, 0.0, s_arr / npix_f)
        if e_arr.size > 0:
            e_out = np.where(no_pix, 0.0, e_arr / (npix_f**2))
        else:
            e_out = e_arr.copy()
    return s_out, e_out


def parse_additional_input4_join_sqw_(
    *args: Any,
) -> Tuple[np.ndarray, Any, bool, List[Any]]:
    """Parse input to extract pixel range and initialized job dispatcher if provided.

    Ported from ``horace_core/algorithms/private/parse_additional_input4_join_sqw_.m``.
    """
    data_range = PixelDataBase.EMPTY_RANGE.copy()
    job_disp = None
    jd_initialized = False
    argi: List[Any] = []

    if len(args) == 0:
        return data_range, job_disp, jd_initialized, argi

    rem = list(args)
    is_jd = [
        x is not None and type(x).__name__ == "JobDispatcher" for x in rem
    ]
    if any(is_jd):
        jds = [x for x, m in zip(rem, is_jd) if m]
        if len(jds) > 1:
            raise HoraceError(
                "HORACE:write_nsqw_to_sqw:invalid_argument",
                "only one instance of JobDispatcher can be provided as input",
            )
        job_disp = jds[0]
        if not getattr(job_disp, "is_initialized", False):
            raise HoraceError(
                "HORACE:write_nsqw_to_sqw:invalid_argument",
                "Only initialized JobDispatcher is currently supported"
                " as input for write_nsqw_to_sqw."
                ' Use "parallel" option to combine files in parallel',
            )
        jd_initialized = True
        rem = [x for x, m in zip(rem, is_jd) if not m]

    if len(rem) == 0:
        return data_range, job_disp, jd_initialized, argi

    is_range = [
        isinstance(x, np.ndarray) and x.shape == (2, 9) for x in rem
    ]
    if not any(is_range):
        argi = [x for x in rem if x is not None and not (isinstance(x, list) and len(x) == 0)]
        return data_range, job_disp, jd_initialized, argi

    if sum(is_range) > 1:
        raise HoraceError(
            "HORACE:write_nsqw_to_sqw:invalid_argument",
            "More then one variable in input arguments is interpreted as range."
            " This is not currently supported",
        )

    for x, m in zip(rem, is_range):
        if m:
            data_range = np.asarray(x, dtype=np.float64).copy()
        elif x is not None and not (isinstance(x, list) and len(x) == 0):
            argi.append(x)

    return data_range, job_disp, jd_initialized, argi


def check_img_consistency_(
    img_metadata: Sequence[Any], filenames: Sequence[str]
) -> np.ndarray:
    """Check the consistency of image metadata across contributing objects/files.

    Ported from ``horace_core/algorithms/private/check_img_consistency_.m``.
    """
    n_inputs = len(img_metadata)
    img_range = np.asarray(img_metadata[0].img_range, dtype=np.float64).copy()
    proj = img_metadata[0].proj
    single_eps = float(np.finfo(np.float32).eps)

    for i in range(1, n_inputs):
        loc_range = np.asarray(img_metadata[i].img_range, dtype=np.float64)
        ok_proj = proj.equal_to_tol(img_metadata[i].proj, tol=4.0 * single_eps)
        if isinstance(ok_proj, tuple):
            ok_proj = ok_proj[0]
        if not ok_proj:
            raise HoraceError(
                "HORACE:algorithms:invalid_arguments",
                f"The image projection for contributing sqw/tmp objects have to be the same.\n "
                f"the projection for object N{i + 1}, file-name: {filenames[i]} different from "
                f"the projection for the first contributing obj, filename: {filenames[0]}\n",
            )
        if np.any(np.abs(img_range.ravel() - loc_range.ravel()) > 4.0 * single_eps):
            raise HoraceError(
                "HORACE:algorithms:invalid_arguments",
                f"The binning ranges for all contributing sqw/tmp objects have to be the same.\n "
                f"Range for obj N{i + 1}, file-name: {filenames[i]} different from the range "
                f"of the first contributing object: {filenames[0]}\n"
                f" *** min1: {img_range[0, :]} min{i + 1}: {loc_range[0, :]}\n"
                f" *** max1: {img_range[1, :]} max{i + 1}: {loc_range[1, :]}\n",
            )
        img_range = minmax_ranges(img_range, loc_range)

    return img_range


def _close_ldrs(ldrs: Sequence[Any]) -> List[Any]:
    out = list(ldrs)
    for i, ldr in enumerate(out):
        if ldr is not None:
            try:
                ldr.delete()
            except Exception:
                pass
            out[i] = None
    return out


def check_img_consistency(
    sqw_obj_input: Any, nargout: Optional[int] = None
) -> Any:
    """Validate SQW images (DnD objects) consistency for combining without rebinning.

    Ported from ``horace_core/algorithms/check_img_consistency.m``.
    """
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory
    from pyhorace.sqw.sqw import sqw

    nout = _infer_nargout(default=1, explicit_nargout=nargout)

    if isinstance(sqw_obj_input, (sqw, DnDBase)):
        inputs = [sqw_obj_input]
    elif isinstance(sqw_obj_input, np.ndarray):
        inputs = list(sqw_obj_input.ravel(order="F"))
    elif isinstance(sqw_obj_input, (list, tuple)):
        inputs = list(sqw_obj_input)
    else:
        inputs = [sqw_obj_input]

    img_metadata: List[Any] = []
    filenames: List[str] = []
    ldrs: List[Any] = []

    for in_obj in inputs:
        if isinstance(in_obj, sqw):
            img_metadata.append(in_obj.data.metadata)
            filenames.append(in_obj.full_filename)
            ldrs.append(None)
        elif isinstance(in_obj, DnDBase):
            img_metadata.append(in_obj.metadata)
            filenames.append(in_obj.full_filename)
            ldrs.append(None)
        elif _is_text(in_obj):
            fname = str(in_obj)
            ldr = sqw_formats_factory.instance().get_loader(fname)
            img_metadata.append(ldr.get_dnd_metadata())
            filenames.append(fname)
            ldrs.append(ldr)
        else:
            _close_ldrs(ldrs)
            raise HoraceError(
                "HORACE:algorithms:invalid_argument",
                f"class: {type(in_obj).__name__} of object {in_obj!r} does not supported",
            )

    try:
        img_range = check_img_consistency_(img_metadata, filenames)
    except Exception:
        _close_ldrs(ldrs)
        raise

    if nout < 2:
        _close_ldrs(ldrs)
        return img_range
    return img_range, ldrs


def build_combined_img_(
    axes_bl: Any,
    proj: Any,
    img_sources: Sequence[Any],
    job_disp: Any = None,
    hor_log_level: int = -1,
) -> DnDBase:
    """Collect images from multiple input sources, accumulate them, and return combined DnD object.

    Ported from ``horace_core/algorithms/private/build_combined_img_.m`` and
    ``horace_core/sqw/@accumulate_headers_job/accumulate_headers_job.m:accumulate_headers``.
    """
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory
    from pyhorace.sqw.sqw import sqw

    img = DnDBase.dnd(axes_bl.copy(), proj.copy())

    if hor_log_level > -1:
        print(" ")
        print("Reading and accumulating binning information of input file(s)...")

    s_accum: Optional[np.ndarray] = None
    e_accum: Optional[np.ndarray] = None
    npix_accum: Optional[np.ndarray] = None

    for i, src in enumerate(img_sources):
        if isinstance(src, sqw):
            s_i = np.asarray(src.data.s, dtype=np.float64)
            e_i = np.asarray(src.data.e, dtype=np.float64)
            npix_i = np.asarray(src.data.npix, dtype=np.float64)
        elif isinstance(src, DnDBase):
            s_i = np.asarray(src.s, dtype=np.float64)
            e_i = np.asarray(src.e, dtype=np.float64)
            npix_i = np.asarray(src.npix, dtype=np.float64)
        else:
            bindata = src.get_se_npix()
            if isinstance(bindata, dict):
                s_i = np.asarray(bindata["s"], dtype=np.float64)
                e_i = np.asarray(bindata["e"], dtype=np.float64)
                npix_i = np.asarray(bindata["npix"], dtype=np.float64)
            else:
                s_i = np.asarray(bindata[0], dtype=np.float64)
                e_i = np.asarray(bindata[1], dtype=np.float64)
                npix_i = np.asarray(bindata[2], dtype=np.float64)

        s_weighted = s_i * npix_i
        e_weighted = e_i * (npix_i**2)
        if i == 0 or s_accum is None:
            s_accum = s_weighted
            e_accum = e_weighted
            npix_accum = npix_i.copy()
        else:
            s_accum = s_accum + s_weighted
            e_accum = e_accum + e_weighted
            npix_accum = npix_accum + npix_i

    if s_accum is not None and e_accum is not None and npix_accum is not None:
        s_norm, e_norm = normalize_signal(s_accum, e_accum, npix_accum)
        target_shape = img.s.shape
        img.s = s_norm.reshape(target_shape, order="F")
        img.e = e_norm.reshape(target_shape, order="F")
        img.npix = npix_accum.reshape(target_shape, order="F")

    return img


def combine_exper_and_img_(
    experiments: Sequence[Any],
    img_metadata: Sequence[Any],
    inputs: Sequence[Any],
    allow_equal_headers: bool,
    keep_runid: bool,
    job_dp: Any = None,
    hor_log_level: int = -1,
) -> Tuple[DnDBase, Any, np.ndarray, main_header_cl]:
    """Check consistency of DnD image and combine input image data and experiment data together.

    Ported from ``horace_core/algorithms/private/combine_exper_and_img_.m``.
    """
    filenames = [getattr(x, "full_filename", "") for x in inputs]
    img_range = check_img_consistency_(img_metadata, filenames)

    exper_combined, nspe, run_id_array = experiments[0].combine_experiments(
        experiments[1:],
        allow_equal_headers=allow_equal_headers,
        keep_runid=keep_runid,
        nargout=3,
    )

    from pyhorace.io.main_header_cl import main_header_cl

    nfiles_tot = int(np.sum(nspe))
    mhc = main_header_cl(nfiles=nfiles_tot)

    ab = img_metadata[0].axes
    proj = img_metadata[0].proj

    dnd_data = build_combined_img_(ab, proj, inputs, job_dp, hor_log_level)
    dnd_data.axes.img_range = img_range

    return dnd_data, exper_combined, run_id_array, mhc


def read_input_headers_(
    infiles: Sequence[str],
) -> Tuple[
    List[Any],
    List[Any],
    List[Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Any],
]:
    """Read header information necessary for combining SQW/tmp files together.

    Ported from ``horace_core/sqw/@accumulate_headers_job/private/read_input_headers_.m``.
    """
    nfiles = len(infiles)
    main_header: List[Any] = [None] * nfiles
    exp_info: List[Any] = [None] * nfiles
    datahdr: List[Any] = [None] * nfiles
    pos_npixstart = np.zeros(nfiles, dtype=np.int64)
    pos_pixstart = np.zeros(nfiles, dtype=np.int64)
    npixtot = np.zeros(nfiles, dtype=np.int64)

    from pyhorace.io.sqw_formats_factory import sqw_formats_factory

    ldrs_raw = sqw_formats_factory.instance().get_loader(list(infiles))
    ldrs = ldrs_raw if isinstance(ldrs_raw, list) else [ldrs_raw]

    for i in range(nfiles):
        ldr = ldrs[i]
        data_type = getattr(ldr, "data_type", "")
        if str(data_type).lower() != "a":
            _close_ldrs(ldrs)
            raise HoraceError(
                "WRITE_NSQW_TO_SQW:invalid_argument",
                f"No pixel information in {infiles[i]}",
            )
        main_header[i] = ldr.get_main_header()
        exp_info[i] = ldr.get_exp_info("-all")
        datahdr[i] = ldr.get_dnd_metadata()
        det_tmp = ldr.get_detpar() if exp_info[i].detector_arrays.n_runs == 0 else None
        if det_tmp is not None:
            if isinstance(det_tmp, dict) and exp_info[i].detector_arrays.n_runs == 0:
                if len(det_tmp) > 0:
                    detector = IX_detector_array(det_tmp)
                    exp_info[i].detector_arrays = exp_info[i].detector_arrays.add(detector)
            elif isinstance(det_tmp, IX_detector_array) and exp_info[i].detector_arrays.n_runs == 0:
                exp_info[i].detector_arrays = exp_info[i].detector_arrays.add(det_tmp)
            elif isinstance(det_tmp, unique_objects_container) and exp_info[i].detector_arrays.n_runs == 0:
                if det_tmp.n_objects > 0:
                    exp_info[i].detector_arrays.unique_objects = det_tmp
            elif isinstance(det_tmp, unique_references_container):
                exp_info[i].detector_arrays = det_tmp

            if (
                exp_info[i].detector_arrays.n_runs == 1
                and exp_info[i].n_runs > 1
            ):
                exp_info[i].detector_arrays = exp_info[i].detector_arrays.replicate_runs(
                    exp_info[i].n_runs
                )

        pos_npixstart[i] = int(ldr.npix_position)
        pos_pixstart[i] = int(ldr.pix_position)
        npixtot[i] = int(ldr.npixels)

    return main_header, exp_info, datahdr, pos_npixstart, pos_pixstart, npixtot, ldrs


def get_pix_comb_info_(
    infiles: Sequence[str],
    data_range: np.ndarray,
    job_disp: Any,
    allow_equal_headers: bool,
    keep_runid: bool,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Any]:
    """Prepare pixel combining structure from a list of SQW files.

    Ported from ``horace_core/algorithms/private/get_pix_comb_info_.m``.
    """
    hor_log_level = config_store.instance().get_value("hor_config", "log_level")

    file_list = [str(f) for f in infiles]
    missing = [f for f in file_list if not os.path.isfile(f)]
    if missing:
        raise HoraceError(
            "HORACE:algorithms:invalid_argument",
            f"Can not find files: {missing[0]} ",
        )

    if hor_log_level > -1:
        print(" ")
        print("Reading header(s) of input file(s) and checking consistency...")

    _, experiments_from_files, img_hdrs, pos_npixstart, pos_pixstart, npixtot, ldrs = (
        read_input_headers_(file_list)
    )
    try:
        undef = data_range == PixelDataBase.EMPTY_RANGE
        if np.any(undef):
            data_range = pixfile_combine_info.recalc_data_range_from_loaders(
                ldrs, keep_runid
            )

        dnd_data, exper_combined, run_id_array, mhc = combine_exper_and_img_(
            experiments_from_files,
            img_hdrs,
            ldrs,
            allow_equal_headers,
            keep_runid,
            job_disp,
            hor_log_level,
        )
    finally:
        _close_ldrs(ldrs)

    pix = pixfile_combine_info(
        file_list,
        int(dnd_data.npix.size),
        npixtot,
        pos_npixstart,
        pos_pixstart,
        run_id_array,
    )
    pix.data_range = data_range

    sqw_sum_struc: Dict[str, Any] = {
        "main_header": mhc,
        "experiment_info": exper_combined,
        "detpar": None,
        "data": dnd_data,
        "pix": pix,
    }
    return sqw_sum_struc, data_range, run_id_array, job_disp


def _get_pix_comb_info_from_sqw(
    inputs: Sequence[Any],
    pix_data_range: np.ndarray,
    allow_equal_headers: bool,
    keep_runid: bool,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Construct information about pixel distribution from sequence of ``sqw`` objects in memory."""
    img_hdrs = [w.get_dnd_metadata() for w in inputs]
    experiments_from_sqw = [w.experiment_info for w in inputs]
    pix_list = [w.pix for w in inputs]
    npix_list = [w.data.npix.ravel(order="F") for w in inputs]

    ll = config_store.instance().get_value("hor_config", "log_level")

    dnd_data, exper_combined, run_id_array, mhc = combine_exper_and_img_(
        experiments_from_sqw,
        img_hdrs,
        inputs,
        allow_equal_headers,
        keep_runid,
        None,
        ll,
    )

    pix = pixobj_combine_info(pix_list, npix_list)
    pix.run_label = run_id_array
    if not np.any(pix_data_range == PixelDataBase.EMPTY_RANGE):
        pix.data_range = pix_data_range
    pix_data_range = pix.data_range

    det = inputs[0].detpar
    sqw_sum_struc: Dict[str, Any] = {
        "main_header": mhc,
        "experiment_info": exper_combined,
        "detpar": det,
        "data": dnd_data,
        "pix": pix,
    }
    return sqw_sum_struc, pix_data_range, run_id_array


def collect_sqw_metadata(
    inputs: Any, *args: Any, nargout: Optional[int] = None
) -> Any:
    """Collect metadata from multiple SQW objects or files on a common image grid.

    Ported from ``horace_core/algorithms/collect_sqw_metadata.m``.
    """
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory
    from pyhorace.sqw.sqw import sqw

    nout = _infer_nargout(default=1, explicit_nargout=nargout)

    options = ["-allow_equal_headers", "-keep_runid"]
    ok, mess, allow_equal_headers, keep_runid, argi = parse_char_options(
        args, options
    )
    if not ok:
        raise HoraceError("HORACE:algorithms:invalid_argument", mess)

    pix_data_range, job_disp, _, _ = parse_additional_input4_join_sqw_(*argi)

    if isinstance(inputs, sqw):
        seq = [inputs]
    elif isinstance(inputs, np.ndarray):
        seq = list(inputs.ravel(order="F"))
    elif isinstance(inputs, (list, tuple)):
        seq = list(inputs)
    else:
        raise HoraceError(
            "HORACE:algorithms:invalid_argument",
            f"This routine accepts only list of sqw files or array/list of sqw objects.\n"
            f" First non-sqw object class is: {type(inputs).__name__}",
        )

    if all(_is_text(x) for x in seq):
        sqw_sum_struc, pix_data_range, run_id_array, job_disp = get_pix_comb_info_(
            [str(x) for x in seq],
            pix_data_range,
            job_disp,
            allow_equal_headers,
            keep_runid,
        )
    elif all(isinstance(x, sqw) for x in seq):
        sqw_sum_struc, pix_data_range, run_id_array = _get_pix_comb_info_from_sqw(
            seq, pix_data_range, allow_equal_headers, keep_runid
        )
    else:
        non_sqw = [x for x in seq if not (isinstance(x, sqw) or _is_text(x))]
        bad_cls = type(non_sqw[0]).__name__ if non_sqw else type(seq[0]).__name__
        raise HoraceError(
            "HORACE:algorithms:invalid_argument",
            f"This routine accepts only list of sqw files or sqw objects.\n"
            f" First unknown object's class is: {bad_cls}",
        )

    sqw_out = sqw(sqw_sum_struc)
    if nout == 1:
        return sqw_out
    if nout == 2:
        return sqw_out, pix_data_range
    if nout == 3:
        return sqw_out, pix_data_range, run_id_array
    return sqw_out, pix_data_range, run_id_array, job_disp


def write_nsqw_to_sqw(
    infiles: Any, outfile: Any = None, *args: Any, nargout: Optional[int] = None
) -> Any:
    """Read a collection of SQW files with a common grid and write to a single SQW file.

    Ported from ``horace_core/algorithms/write_nsqw_to_sqw.m``.
    """
    from pyhorace.coord_transf.page_operations.page_op_join_sqw import (
        PageOp_join_sqw,
    )
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory
    from pyhorace.sqw.sqw import sqw

    nout = _infer_nargout(default=3, explicit_nargout=nargout)

    if outfile is None:
        raise HoraceError(
            "HORACE:write_nsqw_to_sqw:invalid_argument",
            "function should have at least 2 input arguments",
        )

    accepted_options = ["-parallel", "-keep_runid"]
    ok, mess, combine_in_parallel, keep_runid, argi = parse_char_options(
        args, accepted_options
    )
    if not ok:
        raise HoraceError("HORACE:write_nsqw_to_sqw:invalid_argument", mess)

    pix_data_range, job_disp, jd_initialized, argi = (
        parse_additional_input4_join_sqw_(*argi)
    )

    outfile_str = str(outfile)
    out_dir = os.path.dirname(os.path.abspath(outfile_str))
    if out_dir and not os.path.isdir(out_dir):
        raise HoraceError(
            "HORACE:write_nsqw_to_sqw:invalid_argument",
            f"Output directory does not exist: {out_dir}",
        )
    if os.path.isfile(outfile_str):
        try:
            os.remove(outfile_str)
        except OSError as exc:
            raise HoraceError(
                "HORACE:write_nsqw_to_sqw:invalid_argument",
                f"Cannot overwrite existing file {outfile_str}: {exc}",
            ) from exc

    job_disp_4head = job_disp if jd_initialized else None

    meta_args = list(argi)
    if keep_runid:
        meta_args = ["-keep_runid"] + meta_args

    sqw_mem_part, _, run_id, job_disp = collect_sqw_metadata(
        infiles, pix_data_range, job_disp_4head, *meta_args, nargout=4
    )
    if job_disp is not None and hasattr(job_disp, "finalize_all"):
        job_disp.finalize_all()

    sqw_mem_part.full_filename = outfile_str
    sqw_mem_part.creation_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    hc = hor_config()
    hor_log_level = hc.log_level
    use_mex = bool(hc.use_mex)
    if hor_log_level > -1:
        print(" ")
        print(f"Writing to output file {outfile_str} ...")

    page_op = PageOp_join_sqw()
    page_op.outfile = outfile_str

    hpc = hpc_config()
    use_mex = use_mex and str(hpc.combine_sqw_using).lower().startswith("mex")
    page_op, wout = page_op.init(sqw_mem_part, run_id, use_mex)
    wout = sqw.apply_op(wout, page_op)

    img_db_range = wout.data.img_range
    pix_data_range_out = wout.pix.data_range

    if nout == 1:
        return img_db_range
    if nout == 2:
        return img_db_range, pix_data_range_out
    return img_db_range, pix_data_range_out, wout


def join_sqw(w: Any, *args: Any) -> Any:
    """Join an array or sequence of ``sqw`` objects or ``sqw`` files into a single ``sqw`` object.

    Ported from ``horace_core/sqw/@sqw/join.m``.
    """
    from pyhorace.coord_transf.page_operations.page_op_join_sqw import (
        PageOp_join_sqw,
    )
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory
    from pyhorace.sqw.sqw import sqw

    if isinstance(w, sqw):
        w_seq = [w]
    elif _is_text(w):
        w_seq = [str(w)]
    elif isinstance(w, np.ndarray):
        w_seq = list(w.ravel(order="F"))
    elif isinstance(w, (list, tuple)):
        w_seq = list(w)
    else:
        w_seq = [w]

    nfiles = len(w_seq)
    if nfiles == 1:
        wout = w_seq[0]
        if _is_text(wout):
            wout = sqw(str(wout))
        return wout

    wi = None
    if len(args) > 0 and isinstance(args[0], sqw):
        wi = args[0]
        argi = list(args[1:])
    else:
        argi = list(args)

    initflag = wi is not None and wi.main_header.nfiles == nfiles
    opts = ["-allow_equal_headers", "-recalc_runid"]
    ok, mess, allow_equal_headers, recalc_runid, argi = parse_char_options(
        argi, opts
    )
    if not ok:
        raise HoraceError("HORACE:join:invalid_argument", mess)

    if len(argi) > 0 and _is_text(argi[0]):
        outfile = str(argi[0])
    else:
        outfile = ""

    if initflag and wi is not None:
        wout = wi.copy()
        _, ldrs = check_img_consistency([wout] + w_seq, nargout=2)
        if ldrs[0] is not None:
            ldrs[0].delete()
            ldrs[0] = None
        ldrs = ldrs[1:]

        membased = [ldr is None for ldr in ldrs]
        if all(membased):
            pix_list = [x.pix for x in w_seq]
            npix_list = [x.data.npix.ravel(order="F") for x in w_seq]
            wout.pix = pixobj_combine_info(pix_list, npix_list)
            if recalc_runid:
                wout.experiment_info.runid_map = np.arange(1, len(w_seq) + 1)
        else:
            if any(membased):
                _close_ldrs(ldrs)
                raise HoraceError(
                    "HORACE:sqw:not_implemented",
                    "join combines either input files or input objects. Possibility to mix them is not implemented",
                )
            run_label: Any = (
                np.arange(1, len(w_seq) + 1) if recalc_runid else "nochange"
            )
            npixtot = np.array([int(ldr.npixels) for ldr in ldrs], dtype=np.int64)
            pos_npixstart = np.array(
                [int(ldr.npix_position) for ldr in ldrs], dtype=np.int64
            )
            pos_pixstart = np.array(
                [int(ldr.pix_position) for ldr in ldrs], dtype=np.int64
            )
            pix = pixfile_combine_info(
                [str(x) for x in w_seq],
                int(wout.data.npix.size),
                npixtot,
                pos_npixstart,
                pos_pixstart,
                run_label,
            )
            pix.data_range = wout.pix.data_range
            wout.pix = pix
            _close_ldrs(ldrs)
    else:
        meta_args: List[str] = []
        if allow_equal_headers:
            meta_args.append("-allow_equal_headers")
        if not recalc_runid:
            meta_args.append("-keep_runid")
        wout = collect_sqw_metadata(w_seq, *meta_args, nargout=1)

    hpc = hpc_config()
    hc = hor_config()
    use_mex = bool(hc.use_mex) and str(hpc.combine_sqw_using).lower().startswith("mex")

    page_op = PageOp_join_sqw()
    page_op.outfile = outfile
    if outfile:
        page_op.init_filebacked_output = True

    if recalc_runid:
        run_id = np.asarray(wout.runid_map.keys, dtype=np.float64)
    else:
        run_id = np.array([], dtype=np.float64)

    page_op, wout = page_op.init(wout, run_id, use_mex)
    wout = sqw.apply_op(wout, page_op)
    if not wout.full_filename:
        wout.full_filename = page_op.outfile

    return wout


__all__ = [
    "build_combined_img_",
    "check_img_consistency",
    "check_img_consistency_",
    "collect_sqw_metadata",
    "combine_exper_and_img_",
    "get_pix_comb_info_",
    "join_sqw",
    "normalize_signal",
    "parse_additional_input4_join_sqw_",
    "read_input_headers_",
    "write_nsqw_to_sqw",
]
