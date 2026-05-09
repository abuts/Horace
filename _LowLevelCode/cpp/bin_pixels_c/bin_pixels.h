#pragma once
#include "CommonBinCode.h"
// templates for various binning modes:
#include "processNpixOnly.h"
#include "processSigErr.h"
#include "processWithSorting.h"
#include "processWithNoSorting.h"
#include "processWithNoSortSel.h"

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
    std::array<Fn, 2 * n_modes> t{};

    t[static_cast<size_t>(opModes::npix_only)] = &invoke<processNpixOnly<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err)] = &invoke<processSigErr<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sigerr_cell)] = &invoke<processSigerrCell<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_pix)] = &invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid)] = &invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort)] = &invoke<processWithNoSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel)] = &invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected)] = &invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    // OMP calculations
    t[static_cast<size_t>(opModes::npix_only) + n_modes] = &invoke<processNpixOnlyWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err) + n_modes] = &invoke<processSigErrWithOMP<SRC, TRG>, SRC, TRG>;
    // no omp for this
    t[static_cast<size_t>(opModes::sigerr_cell) + n_modes] = &invoke<processSigerrCell<SRC, TRG>, SRC, TRG>;

    t[static_cast<size_t>(opModes::sort_pix) + n_modes] = &invoke<processWithSortingWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid) + n_modes] = &invoke<processWithSortingWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort) + n_modes] = &invoke<processWithNoSortingWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel) + n_modes] = &invoke<processWithNoSortSelWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected) + n_modes] = &invoke<processWithNoSortSelWithOMP<SRC, TRG>, SRC, TRG>;

    return t;
};

template<typename SRC, typename TRG>
auto makeTransfAndBinTable() {
    using Fn = void(*)(CommonBinCodeWithTransf<SRC, TRG>&,
        span<double>&,
        span<double>&,
        span<double>&);

    const static size_t n_modes = static_cast<size_t>(opModes::N_OP_Modes);
    std::array<Fn, 2 * n_modes> t{};

    t[static_cast<size_t>(opModes::npix_only)] = &invoke_and_transf<processNpixOnly<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err)] = &invoke_and_transf<processSigErr<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_pix)] = &invoke_and_transf<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid)] = &invoke_and_transf<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort)] = &invoke_and_transf<processWithNoSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel)] = &invoke_and_transf<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected)] = &invoke_and_transf<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    // OMP calculations
    t[static_cast<size_t>(opModes::npix_only) + n_modes] = \
        &invoke_and_transf<processNpixOnlyWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err) + n_modes] = &invoke_and_transf<processSigErrWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_pix) + n_modes] = &invoke_and_transf<processWithSortingWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid) + n_modes] = &invoke_and_transf<processWithSortingWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort) + n_modes] = &invoke_and_transf<processWithNoSortingWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel) + n_modes] = &invoke_and_transf<processWithNoSortSelWithOMP<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected) + n_modes] = &invoke_and_transf<processWithNoSortSelWithOMP<SRC, TRG>, SRC, TRG>;

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
    // omp modes; enable if requested and reasonable number of pixels provided 
    // TODO: make it externally configurable
    if (bin_par_ptr->num_threads > 1 && bin_par_ptr->n_data_points>100000) {
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
