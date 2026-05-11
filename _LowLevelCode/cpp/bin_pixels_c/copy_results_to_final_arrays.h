#pragma once
/* Copy selected pixels from original array to the target array, containing only selected pixels.
** Pixels are not sorted and array of indices which correspond to pixels positions according
** to image is returned instead
* Inputs:
* bin_par_ptr -- pointer to binning parameters of the routine, containing
*                routine input parameters (alignment matrix, if any) and holder
*                for output parameters (contributing pixels array and  array of
*                pixel indices in image.
* pix_data    -- pointer to array of input pixels previously evaluated and checked
*                on what pixel contributes and what does not.
* data_size   -- number of pixels in pixel_coord array.
* nPixel_retained
*             -- number of pixels contributed into final image
* pix_ok_bin_idx
*             -- pointer to array containg info about pix_data position in image
*                if pixel contributes to image or negative if it does not.
* Returns:
* bin_par_ptr  with modified "pix_ok_ptr", "pix_img_idx_ptr" and "unique_runID" fields.
*              Two first pointers are holding MATLAB arrays allocated by routine,
*              and contain contributed pixels data array and their indices in target image.
* unique_runID is unordered map containing unique indeces of runs, contributed to indices.
*              The map is modified by adding the indices, unique to this particular block of
*              pixel data.
*/
template <class SRC, class TRG>
void inline copy_results_to_final_arrays(BinningArg* const bin_par_ptr, span<const SRC> pix_data,
    size_t data_size, size_t nPixel_retained, span<mxInt64> pix_ok_bin_idx)
{
    // allocate memory for pixels to retain.
    span<TRG> selected_pix; // pointer to the actual data position.
    bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, nPixel_retained, selected_pix);

    // allocated memory for actual pixel indices
    span<mxInt64> pix_img_idx;
    bin_par_ptr->pix_img_idx_ptr = allocate_pix_memory<mxInt64>(nPixel_retained, 1, pix_img_idx);


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
            targ_pix_array_pos = align_and_copy_pixels<SRC, TRG>(bin_par_ptr->alignment_matrix, pix_data, i, selected_pix, targ_pix_pos);
        }
        else {
            // copy all pixel data into the location requested
            targ_pix_array_pos = copy_pixels<SRC, TRG>(pix_data, i, selected_pix, targ_pix_pos);
        }
        // search for unique run_id;
        bin_par_ptr->unique_runID.insert(uint32_t(selected_pix[targ_pix_array_pos + pix_flds::irun]));

        targ_pix_pos++; // move to the next pixel position within the target array
    }
};

/* Copy selected pixels from original array to the target array, containing only
** selected pixels using OMP.
** Pixels are not sorted and array of indices which correspond to pixels positions
** according to image is returned instead.
* Inputs:
* bin_par_ptr -- pointer to binning parameters of the routine, containing
*                routine input parameters (alignment matrix, if any) and holder
*                for output parameters (contributing pixels array and  array of
*                pixel indices in image.
* pix_data    -- pointer to array of input pixels previously evaluated and checked
*                on what pixel contributes and what does not.
* data_size   -- number of pixels in pixel_coord array.
* nPixel_retained
*             -- number of pixels contributed into final image
* pix_ok_bin_idx
*             -- pointer to array containg info about pix_data position in image
*                if pixel contributes to image or negative if it does not.
* Returns:
* bin_par_ptr  with modified "pix_ok_ptr", "pix_img_idx_ptr" and "unique_runID" fields.
*              Two first pointers are holding MATLAB arrays allocated by routine,
*              and contain contributed pixels data array and their indices in target image.
* unique_runID is unordered map containing unique indeces of runs, contributed to indices.
*              The map is modified by adding the indices, unique to this particular block of
*              pixel data.
*/
template <class SRC, class TRG>
void inline copy_results_to_final_arraysWithOMP(int n_thread,
    span<TRG> &selected_pix, span<mxInt64> &pix_img_idx, std::vector<std::unordered_set<uint32_t> >& unique_ID_tls,
    bool align_result, const std::vector<double> & alignment_matrix, span<const SRC> pix_coord, size_t data_size,
    size_t nPixel_retained, span<mxInt64> pix_ok_bin_idx,
    const std::vector<size_t> &npix_thread_contibution_start, const std::vector<thread_range> &tls_thread_range)
{

    // actually move pixels and copy indices to the target array
    size_t targ_pix_pos(npix_thread_contibution_start[n_thread]);
    size_t targ_pix_array_pos(0);
    auto iStart = tls_thread_range[n_thread].min_idx;
    auto iEnd   = tls_thread_range[n_thread].max_idx;
    for (auto i = iStart; i <= iEnd; i++) {
        if (pix_ok_bin_idx[i] < 0) // drop pixels with have not been included above
            continue;

        // number of image cell pixel should go to
        pix_img_idx[targ_pix_pos] = static_cast<size_t>(pix_ok_bin_idx[i] + 1); // MATLB indices start from 1 and these -- from 0

        if (align_result) {
            // align q-coordinates and copy all other pixel data into the location requested
            targ_pix_array_pos = align_and_copy_pixels<SRC, TRG>(alignment_matrix, pix_coord, i, selected_pix, targ_pix_pos);
        }
        else {
            // copy all pixel data into the location requested
            targ_pix_array_pos = copy_pixels<SRC, TRG>(pix_coord, i, selected_pix, targ_pix_pos);
        }
        // search for unique run_id;
        unique_ID_tls[n_thread].insert(uint32_t(selected_pix[targ_pix_array_pos + pix_flds::irun]));

        targ_pix_pos++; // move to the next pixel position within the target array
    }
};
