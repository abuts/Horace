#pragma once
#include "CommonBinCode.h"
// templates for various binning modes:
#include "processNpixOnly.h"
#include "processSigErr.h"
#include "processWithSorting.h"
#include "processWithNoSorting.h"
#include "processWithNoSortSel.h"

// Macro to process unsupported binning modes failing gently
template<class SRC, class TRG>
struct processInvalidCall{
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        std::stringstream buf;
        buf << "Axess binning does not support operational mode" << ctx.bin_par_ptr->binMode;
        mexErrMsgIdAndTxt("HORACE:bin_pixels_c:invalid_argument",
            buf.str().c_str());
    }
};
// Macro to process unsupported binning and transformation modes failing gently.
template<class SRC, class TRG>
struct processInvalidCallWithTransf {
    void operator()(CommonBinCodeWithTransf<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        std::stringstream buf;
        buf << "binning with transformation (projection) does not support operational mode" << ctx.bin_par_ptr->binMode;
        mexErrMsgIdAndTxt("HORACE:bin_pixels_c:invalid_argument",
            buf.str().c_str());
    }
};

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
template<typename Cmd, typename CTX>
void invoke(CTX& ctx, span<double>& npix, span<double>& s, span<double>& e) {
    Cmd{}(ctx, npix, s, e);
}

// Define function table which contains various binning sub-algorithms
template<typename CTX, typename SRC, typename TRG>
auto makeBinTable() {
    using Fn = void(*)(CTX &,
        span<double>&,
        span<double>&,
        span<double>&);

    constexpr size_t N_MODES = static_cast<size_t>(opModes::N_OP_Modes);
    constexpr size_t ALL_MODES = 2 * N_MODES; // should satisfy cppcheck which does not recognize
    std::array<Fn, ALL_MODES > t{}; // this constant expression

    t[static_cast<size_t>(opModes::npix_only)] = &invoke<processNpixOnly<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::invalid_mode)] = &invoke<processInvalidCall<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::sig_err)] = &invoke<processSigErr<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::sigerr_cell)] = &invoke<processSigerrCell<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::sort_pix)] = &invoke<processWithSorting<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::sort_and_uid)] = &invoke<processWithSorting<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::nosort)] = &invoke<processWithNoSorting<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::nosort_sel)] = &invoke<processWithNoSortSel<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::siger_selected)] = &invoke<processWithNoSortSel<SRC, TRG>, CTX>;
    // OMP calculations
    t[static_cast<size_t>(opModes::npix_only) + N_MODES]  = &invoke<processNpixOnlyWithOMP<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::invalid_mode)+N_MODES] = &invoke<processInvalidCall<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::sig_err) + N_MODES]    = &invoke<processSigErrWithOMP<SRC, TRG>, CTX>;
    //TODO: no omp for this yet
    t[static_cast<size_t>(opModes::sigerr_cell) + N_MODES] = &invoke<processSigerrCell<SRC, TRG>, CTX>;

    t[static_cast<size_t>(opModes::sort_pix) + N_MODES] = &invoke<processWithSortingWithOMP<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::sort_and_uid) + N_MODES] = &invoke<processWithSortingWithOMP<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::nosort) + N_MODES] = &invoke<processWithNoSortingWithOMP<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::nosort_sel) + N_MODES] = &invoke<processWithNoSortSelWithOMP<SRC, TRG>, CTX>;
    t[static_cast<size_t>(opModes::siger_selected) + N_MODES] = &invoke<processWithNoSortSelWithOMP<SRC, TRG>, CTX>;

    return t;
};


template<typename CTX, typename SRC, typename TRG>
const auto& fBinTable() {
    static const auto table = makeBinTable<CTX, SRC, TRG>();
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
    if (bin_par_ptr->num_threads > 1) {
        bin_mode += static_cast<size_t>(opModes::N_OP_Modes);
    }
    if (bin_par_ptr->transform_pixels) {
        // initalize context for binning after transformation
        CommonBinCodeWithTransf<SRC, TRG> transfCtx(bin_par_ptr);
        //execute appropriate sub-algorithm
        fBinTable<CommonBinCodeWithTransf<SRC, TRG>,SRC, TRG>()[bin_mode](transfCtx, npix, s, e);
        return transfCtx.nPixel_retained;
    }
    else {
        // initalize context for direct binning
        CommonBinCode<SRC, TRG> binCtx(bin_par_ptr);
        //execute appropriate sub-algorithm
        fBinTable<CommonBinCode<SRC, TRG>,SRC, TRG>()[bin_mode](binCtx, npix, s, e);
        return binCtx.nPixel_retained;
    }
}
