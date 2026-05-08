#pragma once
#include "CommonBinCode.h"
// templates for various binning modes:
#include "processNpixOnly.h"
#include "processSigErr.h"

template<class SRC, class TRG>
struct processSigerrCell {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        std::vector<double*> accum_ptr(3);
        accum_ptr[0] = s.data();
        accum_ptr[1] = e.data();
        accum_ptr[2] = npix.data();
        auto n_cells_to_bin = ctx.bin_par_ptr->n_Cells_to_bin;
        bool npix_acc_separate = n_cells_to_bin < 3; // values for npix accumulators may be provided in separate array
        // if they are not, calculate this value anyway
        std::vector<const double*> cell_data_ptr(n_cells_to_bin, nullptr);

        // fill in cell_data_ptr with pointers to contents of cell data to bin
        for (auto i = 0; i < n_cells_to_bin; i++) {
            const mxArray* cell_array_ptr = mxGetCell(ctx.bin_par_ptr->all_pix_ptr, i);
            cell_data_ptr[i] = mxGetPr(cell_array_ptr);
        }
        std::vector<double> qu(ctx.COORD_STRIDE);
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i, qu))
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid
            size_t il = ctx.pix_position(qu);

            if (npix_acc_separate) {
                // calculate npix accumulators separately if their value is not provided as input
                npix[il]++;
            }

            // calculate signal and, if necessary error accumulators
            for (auto j = 0; j < n_cells_to_bin; j++) {
                auto acc_ptr = accum_ptr[j];
                auto data_ptr = cell_data_ptr[j];
                acc_ptr[il] += data_ptr[i];
            }
        }
    }
};

template<class SRC, class TRG>
struct processWithSorting {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        std::vector<mxInt64> pix_ok_bin_idx;
        pix_ok_bin_idx.swap(ctx.bin_par_ptr->pix_ok_bin_idx);
        std::vector<size_t> npix1;
        npix1.swap(ctx.bin_par_ptr->npix1);
        std::vector<double> qu(ctx.COORD_STRIDE);
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i, qu))
                continue;
            // drop out already selected pixels, if requested
            size_t ip0 = i * ctx.PIX_STRIDE;
            if (ctx.check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0)
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid and add values of this pixels to the accumulators
            // It is almost like add_pixels_to_accumulators but npix1 instead of npix and types of these arrays are different
            // calculate location of pixel within the image grid
            size_t il = ctx.pix_position(qu);
            // calculate npix accumulators for single page of pixels
            npix1[il]++;
            // calculate signal and error accumulators
            // calculate signal and error accumulators taken from current pixel
            s[il] += (double)ctx.pix_coord[ip0 + pix_flds::iSign];
            e[il] += (double)ctx.pix_coord[ip0 + pix_flds::iErr];
            // store indices of contributing pixels
            pix_ok_bin_idx[i] = il;
            // calculate pix ranges
            calc_pix_ranges<SRC>(ctx.pix_ranges, ctx.pix_coord, ctx.PIX_STRIDE, i);
        }
        // allocate memory for pixels to retain contributing pixels.
        TRG* sorted_pix_ptr(nullptr); // pointer to the actual data position.
        ctx.bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, ctx.nPixel_retained, sorted_pix_ptr);
        // calculate ranges of cells to place pixels
        std::vector<size_t> bin_start;
        bin_start.swap(ctx.bin_par_ptr->npix_bin_start);
        bin_start[0] = 0;
        npix[0] += npix1[0];
        if (ctx.distribution_size > 1) {
            for (size_t i = 1; i < ctx.distribution_size; i++) {
                bin_start[i] = bin_start[i - 1] + npix1[i - 1]; // range of cell to place pixels
                npix[i] += npix1[i]; // increase multi-call accumulators
            }
        }
        bool align_result = ctx.bin_par_ptr->alignment_matrix.size() == 9;
        size_t targ_pix_pos(0);
        bool keep_unique_id = ctx.bin_par_ptr->binMode == opModes::sort_and_uid;
        // actually sort pixels and copy selected pixels into proper locations within the target array
        for (size_t i = 0; i < ctx.data_size; i++) {
            if (pix_ok_bin_idx[i] < 0) // drop pixels with have not been included above
                continue;

            size_t il = (size_t)pix_ok_bin_idx[i]; // number of cell pixel should go to
            auto cell_pix_ind = bin_start[il]++; // pixel position within the array defined by cell
            if (align_result) {
                // align q-coordinates and copy all other pixel data into the location requested
                targ_pix_pos = align_and_copy_pixels<SRC, TRG>(ctx.bin_par_ptr->alignment_matrix, ctx.pix_coord, i, sorted_pix_ptr, cell_pix_ind);
            }
            else {
                targ_pix_pos = copy_pixels<SRC, TRG>(ctx.pix_coord, i, sorted_pix_ptr, cell_pix_ind); // copy all pixel data into the location requested
            }
            if (keep_unique_id) {
                ctx.bin_par_ptr->unique_runID.insert(uint32_t(sorted_pix_ptr[targ_pix_pos + pix_flds::irun]));
            }
        }
        // swap memory of working arrays back to binning_arguments to retain it for the next call
        ctx.bin_par_ptr->pix_ok_bin_idx.swap(pix_ok_bin_idx);
        ctx.bin_par_ptr->npix_bin_start.swap(bin_start);
        ctx.bin_par_ptr->npix1.swap(npix1);
    }
};

template<class SRC, class TRG>
struct processWithNoSorting {

    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        std::vector<mxInt64> pix_ok_bin_idx;
        pix_ok_bin_idx.swap(ctx.bin_par_ptr->pix_ok_bin_idx);
        std::vector<double> qu(ctx.COORD_STRIDE);
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i, qu))
                continue;

            // drop out already selected pixels, if requested
            size_t ip0 = i * ctx.PIX_STRIDE;
            if (ctx.check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0)
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid and add values of this pixels to the accumulators
            auto il = ctx.add_pix_to_accumulators(qu, ip0, npix, s, e);

            // store indices of contributing pixels
            pix_ok_bin_idx[i] = il;
            // calculate pix ranges
            calc_pix_ranges<SRC>(ctx.pix_ranges, ctx.pix_coord, ctx.PIX_STRIDE, i);
        }
        // allocate memory for pixels to retain.
        TRG* selected_pix_ptr(nullptr); // pointer to the actual data position.
        ctx.bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, ctx.nPixel_retained, selected_pix_ptr);
        // allocated memory for pixel indices
        mxInt64* pix_img_idx_ptr(nullptr);
        ctx.bin_par_ptr->pix_img_idx_ptr = allocate_pix_memory<mxInt64>(ctx.nPixel_retained, 1, pix_img_idx_ptr);
        span<mxInt64> pix_img_idx(pix_img_idx_ptr, ctx.nPixel_retained);
        copy_results_to_final_arrays<SRC, TRG>(ctx.bin_par_ptr, ctx.pix_coord,
            ctx.data_size, ctx.nPixel_retained, pix_ok_bin_idx);
        // swap memory of working arrays back to binning_arguments to retain it for the next call
        ctx.bin_par_ptr->pix_ok_bin_idx.swap(pix_ok_bin_idx);
    }
};

template<class SRC, class TRG>
struct processWithNoSortSel {

    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

        auto return_selected_only = ctx.bin_par_ptr->binMode == opModes::siger_selected;
        std::vector<mxInt64> pix_ok_bin_idx;
        if (!return_selected_only) {
            pix_ok_bin_idx.swap(ctx.bin_par_ptr->pix_ok_bin_idx);
        }

        // Allocate memory for logical array of selected pixels
        mxLogical* is_pix_selected_ptr(nullptr);
        span<mxLogical> is_pix_selected;
        ctx.bin_par_ptr->is_pix_selected_ptr = allocate_pix_memory<mxLogical>(1, ctx.data_size, is_pix_selected_ptr);
        is_pix_selected = span<mxLogical>(is_pix_selected_ptr, ctx.data_size);
        std::vector<double> qu(ctx.COORD_STRIDE);
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i, qu)) {
                is_pix_selected[i] = false;
                continue;
            }
            else {
                is_pix_selected[i] = true;
            }

            // drop out already selected pixels, if requested
            size_t ip0 = i * ctx.PIX_STRIDE;
            if (ctx.check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0) {
                is_pix_selected[i] = false;
                continue;
            }

            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid and add values of this pixels to the accumulators
            auto il = ctx.add_pix_to_accumulators(qu, ip0, npix, s, e);
            if (!return_selected_only) {
                pix_ok_bin_idx[i] = il;
                // calculate pix ranges
                calc_pix_ranges<SRC>(ctx.pix_ranges, ctx.pix_coord, ctx.PIX_STRIDE, i);
            }
        }
        if (return_selected_only) {
            return;
        }
        copy_results_to_final_arrays<SRC, TRG>(ctx.bin_par_ptr, ctx.pix_coord,
            ctx.data_size, ctx.nPixel_retained, pix_ok_bin_idx);
        // swap memory of working arrays back to binning_arguments to retain it for the next call
        ctx.bin_par_ptr->pix_ok_bin_idx.swap(pix_ok_bin_idx);

    }
};

template<typename Cmd, typename SRC, typename TRG>
void invoke(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) {
    Cmd{}(ctx, npix, s, e);
};
template<typename Cmd, typename SRC, typename TRG>
void invoke_and_transf(CommonBinCodeWithTransf<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) {
    Cmd{}(ctx, npix, s, e);
};


// Define table which contains various binning sub-algorithms
template<typename SRC, typename TRG>
auto makeBinTable() {
    using Fn = void(*)(CommonBinCode<SRC, TRG>&,
        span<double>&,
        span<double>&,
        span<double>&);

    const static size_t n_modes = static_cast<size_t>(opModes::N_OP_Modes);
    std::array<Fn, 2*n_modes> t{};

    t[static_cast<size_t>(opModes::npix_only)] = &invoke<processNpixOnly<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err)] = &invoke<processSigErr<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sigerr_cell)] = &invoke<processSigerrCell<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_pix)] = &invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid)] = &invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort)] = &invoke<processWithNoSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel)] = &invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected)] = &invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;

    t[static_cast<size_t>(opModes::npix_only)+ n_modes] = &invoke<processNpixOnlyWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err) + n_modes] = &invoke<processSigErrWithOMP<SRC, TRG>, SRC, TRG>;
    // stubs for the future
    t[static_cast<size_t>(opModes::sigerr_cell) + n_modes] = &invoke<processSigerrCell<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_pix) + n_modes] = &invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid) + n_modes] = &invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort) + n_modes] = &invoke<processWithNoSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel) + n_modes] = &invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected) + n_modes] = &invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;

    return t;
};

template<typename SRC, typename TRG>
auto makeTransfAndBinTable() {
    using Fn = void(*)(CommonBinCodeWithTransf<SRC, TRG>&,
        span<double>&,
        span<double>&,
        span<double>&);

    const static size_t n_modes = static_cast<size_t>(opModes::N_OP_Modes);
    std::array<Fn, 2*n_modes> t{};

    t[static_cast<size_t>(opModes::npix_only)] = &invoke_and_transf<processNpixOnly<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err)] = &invoke_and_transf<processSigErr<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_pix)] = &invoke_and_transf<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid)] = &invoke_and_transf<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort)] = &invoke_and_transf<processWithNoSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel)] = &invoke_and_transf<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected)] = &invoke_and_transf<processWithNoSortSel<SRC, TRG>, SRC, TRG>;

    t[static_cast<size_t>(opModes::npix_only)+ n_modes] = 
        &invoke_and_transf<processNpixOnlyWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err) + n_modes] = &invoke_and_transf<processSigErrWithOMP<SRC, TRG>, SRC, TRG>;
    // stubs for the future
    t[static_cast<size_t>(opModes::sort_pix) + n_modes] = &invoke_and_transf<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid) + n_modes] = &invoke_and_transf<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort) + n_modes] = &invoke_and_transf<processWithNoSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel) + n_modes] = &invoke_and_transf<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected) + n_modes] = &invoke_and_transf<processWithNoSortSel<SRC, TRG>, SRC, TRG>;

    return t;
};

template<typename SRC, typename TRG>
const auto& fBinTable() {
    static const auto table = makeBinTable<SRC, TRG>();
    return table;
};
template<typename SRC, typename TRG>
const auto& fTransfAndBinTable() {
    static const auto table = makeTransfAndBinTable<SRC, TRG>();
    return table;
};


/** Procedure calculates positions of the input pixels coordinates within specified
 *   image box and various other values related to distributions of pixels over the image
 *   bins, including signal per image box, error per image box and distribution of pixels
 *   according to the image. Template instantiated on the basis of SRC (numerical source type convertible to double)
 *   and TRG (numerical target type convertible to double)
 * Results:
 * npix        -- 1D representation of multidimensional array of pixel distributions over bins
 * s           -- 1D representation of multidimensional array of signal in bins
 * err         -- 1D representation of multidimensional array of error in bins
 * Input-Output parameter:
 * bin_par_ptr -- constant pointer to BinningArg class, containing input parameters which describe binning
 *                and output values calculated in some binning modes.
 */
template <class SRC, class TRG>
size_t bin_pixels(span<double>& npix, span<double>& s, span<double>& e, BinningArg* const bin_par_ptr)
{
    // initialize common code for pixel binning
    size_t bin_mode = static_cast<size_t>(bin_par_ptr->binMode);
    // omp modes
    if (bin_par_ptr->num_threads > 1) {
        bin_mode += static_cast<size_t>(opModes::N_OP_Modes);
    }
    if (bin_par_ptr->transform_pixels) {
        // initalize context for binning after transformation
        CommonBinCodeWithTransf<SRC, TRG> transfCtx(bin_par_ptr);
        //execute appropriate sub-algorithm
        fTransfAndBinTable<SRC, TRG>()[bin_mode](transfCtx, npix, s, e);
        return transfCtx.nPixel_retained;
    }
    else {
        // initalize context for direct binning
        CommonBinCode<SRC, TRG> binCtx(bin_par_ptr);
        //execute appropriate sub-algorithm
        fBinTable<SRC, TRG>()[bin_mode](binCtx, npix, s, e);
        return binCtx.nPixel_retained;
    }
}
