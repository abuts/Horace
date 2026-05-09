#pragma once
// copy selected pixels from original array to the target array, containing only selected pixels
// pixels are not sorted and array of indices which correspond to pixels positions according
// to image is returned instead
template <class SRC, class TRG>
void inline copy_results_to_final_arrays(BinningArg* const bin_par_ptr, span<const SRC> pix_coord,
    size_t data_size, size_t nPixel_retained, span<mxInt64> pix_ok_bin_idx)
{
    // allocate memory for pixels to retain.
    span<TRG> selected_pix; // pointer to the actual data position.
    bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, nPixel_retained, selected_pix);
    // allocated memory for pixel indices

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
            targ_pix_array_pos = align_and_copy_pixels<SRC, TRG>(bin_par_ptr->alignment_matrix, pix_coord, i, selected_pix, targ_pix_pos);
        }
        else {
            // copy all pixel data into the location requested
            targ_pix_array_pos = copy_pixels<SRC, TRG>(pix_coord, i, selected_pix, targ_pix_pos);
        }
        // search for unique run_id;
        bin_par_ptr->unique_runID.insert(uint32_t(selected_pix[targ_pix_array_pos + pix_flds::irun]));

        targ_pix_pos++; // move to the next pixel position within the target array
    }
};


