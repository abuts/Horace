#pragma once
#include "BinningArg.h"
#include <array>
#include <algorithm>

/** identifies 1D index of the image cell where the particular pixel belongs to
* Inputs:
* qi       -- 1-dimensional vector of pixel coordinates to process
* pax      -- 1-to-4 elements vector of pixel indices accounted in binning. Indicates 
*             numbers of pixel coordinates from qi array to include in the binning.
*             I.e. if pax.size()==1 only one coordinates needs to be binned or if
*             pax.size()==4, all four qi coordinates have to be binned in 4-dimensional array
* cut_range -- 6 or 8-elements array defining ranges allowed for pixels. The same as cut_range
*              provided in out_of_range routine above.
* bin_step  -- 6 or 8-elements array defining bin step sizes e.g. (cut_range(2*n+1)-cut_range(2*n))/bin_cell_idx_range(n)
*              where n is the number of pixel coordinate to bin.
* bin_cell_idx_range
*           -- number of bins in each binned direction.
* stride    -- 1-to-4 element's vector which describes 1-D allocation of multidimensional array,
*              i.e. change of linear index per change of 1-4 dimensional index by 1 in each direction.
*             E.g.:
*              if one have 1D array, stride has 1 element and contains 1.
*              For 2-dimensional array of size 9x10, stride == [1,9]
*              For 3-dimensional array of size 9x10x11, stride == [1,9,9*10]
*              For 4-dimensional array of size 9x10x11x12, stride == [1,9,9*10,9*10*11]
* Returns:
* index of pixel in input multidimensional array.
*/
// copy selected pixels from original array to the target array, containing only selected pixels
// pixels are not sorted and array of indices which correspond to pixels positions according
// to image is returned instead
template <class SRC, class TRG>
void inline copy_results_to_final_arrays(BinningArg* const bin_par_ptr, span<const SRC> pix_coord,
    size_t data_size, size_t nPixel_retained, std::vector<mxInt64>& pix_ok_bin_idx)
{
    // allocate memory for pixels to retain.
    TRG* selected_pix_ptr(nullptr); // pointer to the actual data position.
    bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, nPixel_retained, selected_pix_ptr);
    // allocated memory for pixel indices
    mxInt64* pix_img_idx_ptr(nullptr);
    bin_par_ptr->pix_img_idx_ptr = allocate_pix_memory<mxInt64>(nPixel_retained, 1, pix_img_idx_ptr);
    span<mxInt64> pix_img_idx(pix_img_idx_ptr, nPixel_retained);

    bool align_result = bin_par_ptr->alignment_matrix.size() == 9;

    // actually move pixels and copy indices to the target array
    size_t targ_pix_pos(0);
    size_t targ_pix_array_pos(0);
    for (size_t i = 0; i < data_size; i++) {
        if (pix_ok_bin_idx[i] < 0) // drop pixels with have not been included above
            continue;

        size_t il = (size_t)pix_ok_bin_idx[i]; // number of image cell pixel should go to
        pix_img_idx[targ_pix_pos] = il + 1; // MATLB indices start from 1 and these -- from 0

        if (align_result) {
            // align q-coordinates and copy all other pixel data into the location requested
            targ_pix_array_pos = align_and_copy_pixels<SRC, TRG>(bin_par_ptr->alignment_matrix, pix_coord, i, selected_pix_ptr, targ_pix_pos);
        }
        else {
            // copy all pixel data into the location requested
            targ_pix_array_pos = copy_pixels<SRC, TRG>(pix_coord, i, selected_pix_ptr, targ_pix_pos);
        }
        // search for unique run_id;
        bin_par_ptr->unique_runID.insert(uint32_t(selected_pix_ptr[targ_pix_array_pos + pix_flds::irun]));

        targ_pix_pos++; // move to the next pixel position within the target array
    }
};

// define the structure which contains variable common for all binning sub-algorithms
template<class SRC, class TRG>
class common_bin_code {
public:
    BinningArg * const bin_par_ptr;
    const size_t distribution_size;

    const size_t COORD_STRIDE;  // size of coordinates dimension (4 or 3 accoring to input coordinates)
    const size_t PIX_STRIDE;    // size of pixel data dimension (9 according to input coordinates)

    span<SRC> coord;      // wrapper around input pixels coordinates (4xNpix or 3xNpix array of coordinates to bin)
    span<SRC> pix_coord;  // wrapper around whole input pixels array  (9xNpix array for Horace-3&4)

    // internal loop variables (firstprivate)
    size_t  nPixel_retained;    // counter for number of retained pixels
    size_t  nCellOccupied;      // counter for number of occupied cells

    std::vector<double> qi;     // holder for single pixel coordinates
    span<double> cut_range;     // 2x4 or 2x3 array, containing ranges to drom pixels off
    span<double> bin_step;      // 1x n-dimensions array defining inverse sizes of image cells 
    span<size_t> pax;           // numbers of axes (dimensions) to bin
    span<size_t> stride;        // 1D->4D array allocation information (how unary change in 4D index changes 1D index of underlying array)
    span<size_t> bin_cell_idx_range; // sizes of grid to bin

    // initialize space for calculating pixel data ranges if necessary
    span<double> pix_ranges;    // actual range of binned pixels 
    bool check_pix_selection;   //
    long data_size;

    // check if the coordinates of pixel number i belong within the pixel ranges provided.
    virtual bool out_of_ranges(long i)
    {
        size_t ic0 = i * COORD_STRIDE;
        for (size_t upix = 0; upix < COORD_STRIDE; upix++) {
            qi[upix] = double(coord[ic0 + upix]);
            if (qi[upix] < cut_range[2 * upix] || qi[upix] > cut_range[2 * upix + 1]) {
                return true;
            }
        }
        return false;
    };

    // identify the linear positon of pixel within the grid.
    size_t pix_position()
    {
        size_t il(0);
        for (size_t j = 0; j < pax.size(); j++) {
            auto bin_idx = pax[j];
            auto cell_idx = (size_t)std::floor((qi[bin_idx] - cut_range[2 * bin_idx]) * bin_step[j]);
            if (cell_idx > bin_cell_idx_range[j])
                cell_idx = bin_cell_idx_range[j];
            il += cell_idx * stride[j];
        }
        return il;
    };

    // calculate pixel contribution into image.
    size_t add_pix_to_accumulators(size_t pix_in_pix_pos,
        span<double>& npix, span<double>& s, span<double>& e)
    {
        // calculate location of pixel within the image grid
        auto il = this->pix_position();
        // calculate npix accumulators
        npix[il]++;
        // calculate signal and error accumulators
        s[il] += (double)pix_coord[pix_in_pix_pos + pix_flds::iSign];
        e[il] += (double)pix_coord[pix_in_pix_pos + pix_flds::iErr];

        return il;
    };
    // Constructor which defines all binning parameters
    common_bin_code(BinningArg* const bin_par_ptr):
        bin_par_ptr(bin_par_ptr),
        distribution_size(bin_par_ptr->n_grid_points()),
        COORD_STRIDE(bin_par_ptr->in_coord_width),
        PIX_STRIDE(bin_par_ptr->in_pix_width)
    {

        this->check_pix_selection = bin_par_ptr->check_pix_selection && (bin_par_ptr->all_pix_ptr != nullptr);

        this->data_size = bin_par_ptr->n_data_points;

        this->coord = span<SRC>(reinterpret_cast<SRC*>(mxGetPr(bin_par_ptr->coord_ptr)), data_size* COORD_STRIDE);
        if (check_pix_selection) {
            this->pix_coord = span<SRC>(reinterpret_cast<SRC*>(mxGetPr(bin_par_ptr->all_pix_ptr)), data_size*PIX_STRIDE);
        }

        // internal loop variables (firstprivate)
        this->nPixel_retained = 0;
        this->nCellOccupied = 0;

        this->qi.resize(COORD_STRIDE);
        this->cut_range = span<double>(bin_par_ptr->data_range);
        this->bin_step = span<double>(bin_par_ptr->bin_step);
        this->pax = span<size_t>(bin_par_ptr->pax); // projection axis
        this->stride = span<size_t>(bin_par_ptr->stride);
        this->bin_cell_idx_range = span<size_t>(bin_par_ptr->bin_cell_idx_range);

        // initialize space for calculating pixel data ranges if necessary
        auto pix_range_ids = (bin_par_ptr->pix_data_range_ptr == nullptr) ? 0 : 2 * pix_flds::PIX_WIDTH;
        if (bin_par_ptr->binMode > opModes::sigerr_cell && pix_range_ids > 0 && bin_par_ptr->binMode < opModes::siger_selected) { // higher modes process pixel ranges except
            pix_ranges = span<double>(mxGetPr(bin_par_ptr->pix_data_range_ptr), pix_range_ids);
            init_min_max_range_calc(pix_ranges, pix_flds::PIX_WIDTH);
        }
    };
    virtual ~common_bin_code() = default;
};

template<class SRC, class TRG>
class common_bin_code_with_transf : public common_bin_code<SRC,TRG> {
public:
    const bool diag_transf;     // if transformation matrix below is a diagonal matrix
    span<double> transf_matrix; // if defined, contains 3x3 matrix to use for pixel transformation the pixels
    const bool apply_offset;    // if offset has non-zero value and should be applied to data
    span<double> u_offset;      // if defined, 1D array of offsets to extract from pixel coordinates before transforming them
    const int  transf_matrix_width;  // 3 or 4 depending on transformation
    const bool ignore_nan;
    const bool ignore_inf;
    bool ignore_something;
    bool ignore_all;
    common_bin_code_with_transf(BinningArg* const bin_par_ptr) :
        common_bin_code<SRC,TRG>(bin_par_ptr),
        diag_transf(bin_par_ptr->diag_transf),
        apply_offset(bin_par_ptr->apply_offset),
        transf_matrix_width(bin_par_ptr->transf_matrix_width),
        ignore_nan(bin_par_ptr->ignore_nan),
        ignore_inf(bin_par_ptr->ignore_inf)
    {
        transf_matrix = span<double>(bin_par_ptr->transf_matrix);
        u_offset = span<double>(bin_par_ptr->u_offset);
        ignore_something = ignore_nan || ignore_inf;
        ignore_all       = ignore_nan && ignore_inf;
    }

    bool out_of_ranges(long i) override
    {
        size_t ic0 = i * this->PIX_STRIDE;
        if (ignore_something)
        {
            if (ignore_all)
            {
                if (std::isinf(this->pix_coord[ic0 + pix_flds::iSign]) || std::isnan(this->pix_coord[ic0 + pix_flds::iSign]) ||
                    std::isinf(this->pix_coord[ic0 + pix_flds::iErr]) || std::isnan(this->pix_coord[ic0 + pix_flds::iErr]))
                    return true;
            }
            else if (ignore_nan)
            {
                if (std::isnan(this->pix_coord[ic0 + pix_flds::iSign]) || std::isnan(this->pix_coord[ic0 + pix_flds::iErr]))
                    return true;
            }
            else if (ignore_inf)
            {
                if (std::isinf(this->pix_coord[ic0 + pix_flds::iSign]) || std::isinf(this->pix_coord[ic0 + pix_flds::iErr]))
                    return true;
            }
        }

        for (size_t upix = 0; upix < this->COORD_STRIDE; upix++) {
            this->qi[upix] = double(this->pix_coord[ic0 + upix]);
            if (this->qi[upix] < this->cut_range[2 * upix] || this->qi[upix] > this->cut_range[2 * upix + 1]) {
                return true;
            }
        }
        return false;
    };


};


template<class SRC, class TRG>
struct processNpixOnly{
    void operator()(common_bin_code<SRC,TRG> & ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i))
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid
            size_t il = ctx.pix_position();
            npix[il]++;
        }

    }
};

template<class SRC, class TRG>
struct processSigErr {
    void operator()(common_bin_code<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i))
                continue;
            // drop out already selected pixels, if requested
            size_t ip0 = i * ctx.PIX_STRIDE;
            if (ctx.check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0)
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid and add values of this pixels to the accumulators
            ctx.add_pix_to_accumulators(ip0,npix, s, e);
        }
    }
};

template<class SRC, class TRG>
struct processSigerrCell {
    void operator()(common_bin_code<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
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

        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i))
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid
            size_t il = ctx.pix_position();

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

template<class SRC,class TRG>
struct processWithSorting {
    void operator()(common_bin_code<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        std::vector<mxInt64> pix_ok_bin_idx;
        pix_ok_bin_idx.swap(ctx.bin_par_ptr->pix_ok_bin_idx);
        std::vector<size_t> npix1;
        npix1.swap(ctx.bin_par_ptr->npix1);
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i))
                continue;
            // drop out already selected pixels, if requested
            size_t ip0 = i * ctx.PIX_STRIDE;
            if (ctx.check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0)
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid and add values of this pixels to the accumulators
            // It is almost like add_pixels_to_accumulators but npix1 instead of npix and types of these arrays are different
            // calculate location of pixel within the image grid
            size_t il = ctx.pix_position();
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
        // allocate memory for pixels to retain.
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

    void operator()(common_bin_code<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        std::vector<mxInt64> pix_ok_bin_idx;
        pix_ok_bin_idx.swap(ctx.bin_par_ptr->pix_ok_bin_idx);

        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i))
                continue;

            // drop out already selected pixels, if requested
            size_t ip0 = i * ctx.PIX_STRIDE;
            if (ctx.check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0)
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid and add values of this pixels to the accumulators
            auto il = ctx.add_pix_to_accumulators(ip0,npix, s, e);

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
struct processWithNoSortSel{

    void operator()(common_bin_code<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

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

        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i)) {
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
            auto il = ctx.add_pix_to_accumulators(ip0,npix, s, e);
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
void invoke(common_bin_code<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) {
    Cmd{}(ctx, npix, s, e);
};
template<typename Cmd, typename SRC, typename TRG>
void transf_and_invoke(common_bin_code_with_transf<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) {
    Cmd{}(ctx, npix, s, e);
};


// Define table which contains various binning sub-algorithms
template<typename SRC, typename TRG>
auto makeBinTable() {
    using Fn = void(*)(common_bin_code<SRC, TRG>&,
        span<double>&,
        span<double>&,
        span<double>&);

    std::array<Fn, static_cast<size_t>(opModes::N_OP_Modes)> t{};

    t[static_cast<size_t>(opModes::npix_only)]      = &invoke<processNpixOnly<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err)]        = &invoke<processSigErr<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sigerr_cell)]    = &invoke<processSigerrCell<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_pix)]       = &invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid)]   = &invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort)]         = &invoke<processWithNoSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel)]     = &invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected)] = &invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;

    return t;
};

template<typename SRC, typename TRG>
auto makeTransfAndBinTable() {
    using Fn = void(*)(common_bin_code_with_transf<SRC, TRG>&,
        span<double>&,
        span<double>&,
        span<double>&);

    std::array<Fn, static_cast<size_t>(opModes::N_OP_Modes)> t{};

    t[static_cast<size_t>(opModes::npix_only)] = &transf_and_invoke<processNpixOnly<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sig_err)] = &transf_and_invoke<processSigErr<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_pix)] = &transf_and_invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::sort_and_uid)] = &transf_and_invoke<processWithSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort)] = &transf_and_invoke<processWithNoSorting<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::nosort_sel)] = &transf_and_invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
    t[static_cast<size_t>(opModes::siger_selected)] = &transf_and_invoke<processWithNoSortSel<SRC, TRG>, SRC, TRG>;
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
    if (bin_par_ptr->transform_pixels) {
        common_bin_code<SRC, TRG> binCtx(bin_par_ptr);
        //execute appropriate sub-algorithm
        fBinTable<SRC, TRG>()[bin_mode](binCtx, npix, s, e);
        return binCtx.nPixel_retained;
    }
    else {
        common_bin_code_with_transf<SRC, TRG> transfCtx(bin_par_ptr);

        //execute appropriate sub-algorithm
        fTransfAndBinTable<SRC, TRG>()[bin_mode](transfCtx, npix, s, e);
        return transfCtx.nPixel_retained;

    }






}
