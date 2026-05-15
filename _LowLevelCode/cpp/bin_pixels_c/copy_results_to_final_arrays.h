#pragma once
#include <include/CommonCode.h>
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
template <class SRC, class TRG>
void inline copy_results_to_final_arrays_dyn(BinningArg* const bin_par_ptr, span<const SRC> pix_data,
    size_t data_size, size_t nPixel_retained, std::vector<idx_accum> &pix_contribuion)
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
    for (idx_accum & sel_idx : pix_contribuion) {

        pix_img_idx[targ_pix_pos] = sel_idx.img_idx + 1; // MATLB indices start from 1 and these -- from 0

        if (align_result) {
            // align q-coordinates and copy all other pixel data into the location requested
            targ_pix_array_pos = align_and_copy_pixels<SRC, TRG>(bin_par_ptr->alignment_matrix, pix_data, sel_idx.pix_idx, selected_pix, targ_pix_pos);
        }
        else {
            // copy all pixel data into the location requested
            targ_pix_array_pos = copy_pixels<SRC, TRG>(pix_data, sel_idx.pix_idx, selected_pix, targ_pix_pos);
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
* Parameters:
* n_thread     -- number of thread to copy array for
* selected_pix -- target array of pixel data
* pix_img_idx  -- target array of contributing pixel's indices in the image
* unique_ID_tls-- target array of thread-specific unique indices contribution
* align_resut  -- if true, resulting pixels have be aligned.
* alignment_matrix
*              -- if aligh_result is true, 9-element (3x3) rotation matix describing pixel alignment
*                 may be empty if align_resut == false
* pix_coord    -- pointer to source array of pixel data
* 
* pix_data    -- pointer to array of input pixels previously evaluated and checked
*                on what pixel contributes and what does not.
* pix_ok_bin_idx
*             -- pointer to array containg info about pix_data position in image
*                if pixel contributes to image or negative if it does not.
* npix_thread_contibution_start
*             -- array of num_opm_thead size, identifying target position of
*                each thread contribution in the selected_pix and pix_img_idx
*                arrays.
* tls_indices -- load-balanced array of size num_omp_thread containing arrays
                 of pix_coord indices of pixels contributing to the image.
*/
template <class SRC, class TRG>
void inline copy_results_to_final_arraysWithOMP(int n_thread,
    span<TRG> &selected_pix, span<mxInt64> &pix_img_idx, std::vector<std::unordered_set<uint32_t> >& unique_ID_tls,
    bool align_result, const std::vector<double> & alignment_matrix, span<const SRC> pix_coord,
    const std::vector<size_t> &npix_thread_contibution_start, const std::vector<std::vector<idx_accum> > &tls_indices)
{

    // actually move pixels and copy indices to the target array
    size_t targ_pix_pos(npix_thread_contibution_start[n_thread]);
    size_t targ_pix_array_pos(0);
    for (size_t i = 0; i < tls_indices[n_thread].size();++i) {
        const idx_accum &info = tls_indices[n_thread][i];
        // number of image cell pixel should go to
        pix_img_idx[targ_pix_pos] = info.img_idx+1; // MATLB indices start from 1 and these -- from 0

        if (align_result) {
            // align q-coordinates and copy all other pixel data into the location requested
            targ_pix_array_pos = align_and_copy_pixels<SRC, TRG>(alignment_matrix, pix_coord, info.pix_idx, selected_pix, targ_pix_pos);
        }
        else {
            // copy all pixel data into the location requested
            targ_pix_array_pos = copy_pixels<SRC, TRG>(pix_coord, info.pix_idx, selected_pix, targ_pix_pos);
        }
        // search for unique run_id;
        unique_ID_tls[n_thread].insert(uint32_t(selected_pix[targ_pix_array_pos + pix_flds::irun]));

        targ_pix_pos++; // move to the next pixel position within the target array
    }
};


template<class SRC, class TRG>
void inline copy_pixels_to_final_arrays(BinningArg* const bin_par_ptr, span<const SRC> pix_data,
    bool keep_unique_id, const std::vector<idx_accum>& pix_idx_included, span<size_t> bin_start){

    bool align_result = bin_par_ptr->alignment_matrix.size() == 9;

    size_t nPixel_retained = pix_idx_included.size();
    // allocate memory for sorted pixels to retain.
    span<TRG> sorted_pix; // pointer to the actual data position.
    bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, nPixel_retained, sorted_pix);

    // actually sort pixels and copy selected pixels into proper locations within the target array
    size_t targ_pix_pos(0);
    for (const idx_accum& contr_idx : pix_idx_included) {

        size_t il = contr_idx.img_idx;       // number of cell pixel should go to
        auto cell_pix_ind = bin_start[il]++; // pixel position within the array defined by cell
        if (align_result) {
            // align q-coordinates and copy all other pixel data into the location requested
            targ_pix_pos = align_and_copy_pixels<SRC, TRG>(bin_par_ptr->alignment_matrix, pix_data, contr_idx.pix_idx, sorted_pix, cell_pix_ind);
        }
        else {
            targ_pix_pos = copy_pixels<SRC, TRG>(pix_data, contr_idx.pix_idx, sorted_pix, cell_pix_ind); // copy all pixel data into the location requested
        }
        if (keep_unique_id) {
            bin_par_ptr->unique_runID.insert(uint32_t(sorted_pix[targ_pix_pos + pix_flds::irun]));
        }
    }
};
