#pragma once
#include <include/CommonCode.h>

template<class SRC, class TRG>
struct processWithSorting {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

        span<mxInt64> pix_ok_bin_idx(ctx.bin_par_ptr->pix_ok_bin_idx);
        span<size_t> npix1(ctx.bin_par_ptr->npix1.data(), ctx.bin_par_ptr->npix1.size());

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
            auto il = ctx.pix_position(qu);
            // calculate npix accumulators
            npix1[il]++;
            // calculate signal and error accumulators
            s[il] += (double)ctx.pix_coord[ip0 + pix_flds::iSign];
            e[il] += (double)ctx.pix_coord[ip0 + pix_flds::iErr];

            // store indices of contributing pixels
            pix_ok_bin_idx[i] = il;
            // calculate pix ranges
            calc_pix_ranges<SRC>(ctx.pix_ranges, ctx.pix_coord, ip0, ctx.PIX_STRIDE);
        }
        // allocate memory for pixels to retain contributing pixels.
        span<TRG> sorted_pix; // pointer to the actual data position.
        ctx.bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, ctx.nPixel_retained, sorted_pix);
        // calculate ranges of cells to place pixels
        span<size_t> bin_start(ctx.bin_par_ptr->npix_bin_start);
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
                targ_pix_pos = align_and_copy_pixels<SRC, TRG>(ctx.bin_par_ptr->alignment_matrix, ctx.pix_coord, i, sorted_pix, cell_pix_ind);
            }
            else {
                targ_pix_pos = copy_pixels<SRC, TRG>(ctx.pix_coord, i, sorted_pix, cell_pix_ind); // copy all pixel data into the location requested
            }
            if (keep_unique_id) {
                ctx.bin_par_ptr->unique_runID.insert(uint32_t(sorted_pix[targ_pix_pos + pix_flds::irun]));
            }
        }
    }
};
