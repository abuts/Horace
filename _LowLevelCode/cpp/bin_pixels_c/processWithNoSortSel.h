#pragma once
#include "CommonBinCode.h"

template<class SRC, class TRG>
struct processWithNoSortSel {

    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

        auto return_selected_only = ctx.bin_par_ptr->binMode == opModes::siger_selected;
        span<mxInt64> pix_ok_bin_idx;
        if (!return_selected_only) {
            pix_ok_bin_idx = span<mxInt64>(ctx.bin_par_ptr->pix_ok_bin_idx);
        }

        // Allocate memory for logical array of selected pixels
        span<mxLogical> is_pix_selected;
        ctx.bin_par_ptr->is_pix_selected_ptr = allocate_pix_memory<mxLogical>(1, ctx.data_size, is_pix_selected);

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
                calc_pix_ranges<SRC>(ctx.pix_ranges, ctx.pix_coord, ip0, ctx.PIX_STRIDE);
            }
        }
        if (return_selected_only) {
            return;
        }
        copy_results_to_final_arrays<SRC, TRG>(ctx.bin_par_ptr, ctx.pix_coord,
            ctx.data_size, ctx.nPixel_retained, pix_ok_bin_idx);
    }
};
template<class SRC, class TRG>
struct processWithNoSortSelWithOMP {

    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

        auto return_selected_only = ctx.bin_par_ptr->binMode == opModes::siger_selected;
        span<mxInt64> pix_ok_bin_idx;
        if (!return_selected_only) {
            pix_ok_bin_idx = span<mxInt64>(ctx.bin_par_ptr->pix_ok_bin_idx);
        }

        // Allocate memory for logical array of selected pixels
        span<mxLogical> is_pix_selected;
        ctx.bin_par_ptr->is_pix_selected_ptr = allocate_pix_memory<mxLogical>(1, ctx.data_size, is_pix_selected);

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
                calc_pix_ranges<SRC>(ctx.pix_ranges, ctx.pix_coord, ip0, ctx.PIX_STRIDE);
            }
        }
        if (return_selected_only) {
            return;
        }
        copy_results_to_final_arrays<SRC, TRG>(ctx.bin_par_ptr, ctx.pix_coord,
            ctx.data_size, ctx.nPixel_retained, pix_ok_bin_idx);
    }
};

