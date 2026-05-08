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
class CommonBinCode {
public:
    BinningArg* const bin_par_ptr;
    const size_t distribution_size;

    const size_t COORD_STRIDE;  // size of coordinates dimension (4 or 3 accoring to input coordinates)
    const size_t PIX_STRIDE;    // size of pixel data dimension (9 according to input coordinates)

    span<const SRC> coord;      // wrapper around input pixels coordinates (4xNpix or 3xNpix array of coordinates to bin)
    span<const SRC> pix_coord;  // wrapper around whole input pixels array  (9xNpix array for Horace-3&4)

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
        for (size_t upixn = 0; upixn < COORD_STRIDE; upixn++) {
            qi[upixn] = double(coord[ic0 + upixn]);
            if (qi[upixn] < cut_range[2 * upixn] || qi[upixn] > cut_range[2 * upixn + 1]) {
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
    CommonBinCode(BinningArg* const bin_par_ptr) :
        bin_par_ptr(bin_par_ptr),
        distribution_size(bin_par_ptr->n_grid_points()),
        COORD_STRIDE(bin_par_ptr->in_coord_width),
        PIX_STRIDE(bin_par_ptr->in_pix_width)
    {

        this->check_pix_selection = bin_par_ptr->check_pix_selection && (bin_par_ptr->all_pix_ptr != nullptr);

        this->data_size = bin_par_ptr->n_data_points;

        this->coord = span<const SRC>(reinterpret_cast<SRC*>(mxGetPr(bin_par_ptr->coord_ptr)), data_size * COORD_STRIDE);
        if (check_pix_selection) {
            this->pix_coord = span<const SRC>(reinterpret_cast<SRC*>(mxGetPr(bin_par_ptr->all_pix_ptr)), data_size * PIX_STRIDE);
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
    virtual ~CommonBinCode() = default;
};

template<class SRC, class TRG>
class CommonBinCodeWithTransf : public CommonBinCode<SRC, TRG> {
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
    CommonBinCodeWithTransf(BinningArg* const bin_par_ptr) :
        CommonBinCode<SRC, TRG>(bin_par_ptr),
        diag_transf(bin_par_ptr->diag_transf),
        apply_offset(bin_par_ptr->apply_offset),
        transf_matrix_width(bin_par_ptr->transf_matrix_width),
        ignore_nan(bin_par_ptr->ignore_nan),
        ignore_inf(bin_par_ptr->ignore_inf)
    {
        transf_matrix = span<double>(bin_par_ptr->transf_matrix);
        u_offset = span<double>(bin_par_ptr->u_offset);
        ignore_something = ignore_nan || ignore_inf;
        ignore_all = ignore_nan && ignore_inf;
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
        std::vector<double> q_shifted(this->COORD_STRIDE);
        if (this->apply_offset) {
            for (size_t upixn = 0; upixn < this->COORD_STRIDE; upixn++) {
                q_shifted[upixn] = (double)this->pix_coord[ic0 + upixn] - this->u_offset[upixn];
            }
        }
        else {
            for (size_t upixn = 0; upixn < this->COORD_STRIDE; upixn++) {
                q_shifted[upixn] = (double)this->pix_coord[ic0 + upixn];
            }
        }
        double accum(0);
        auto trmw = this->transf_matrix_width;
        for (size_t upixn = 0; upixn < this->COORD_STRIDE; upixn++) {
            if (upixn < trmw) {
                if (this->diag_transf) {
                    accum = this->transf_matrix[upixn] * q_shifted[upixn];
                }
                else {
                    accum = 0;
                    for (size_t j = 0; j < trmw; j++) {
                        accum += this->transf_matrix[j * trmw + upixn] * q_shifted[j];
                    }
                }
            }
            else {
                accum = q_shifted[upixn];
            }
            // drop transformed out-of range pixels.
            if (accum < this->cut_range[2 * upixn] || accum > this->cut_range[2 * upixn + 1]) {
                return true;
            }
            this->qi[upixn] = accum;
        }
        return false;
    };

};