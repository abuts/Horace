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

    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const
    {

        auto return_selected_only = ctx.bin_par_ptr->binMode == opModes::siger_selected;
        span<mxInt64> pix_ok_bin_idx;
        if (!return_selected_only) {
            pix_ok_bin_idx = span<mxInt64>(ctx.bin_par_ptr->pix_ok_bin_idx);
        }

        // Allocate memory for logical array of selected pixels
        span<mxLogical> is_pix_selected;
        ctx.bin_par_ptr->is_pix_selected_ptr = allocate_pix_memory<mxLogical>(1, ctx.data_size, is_pix_selected);
        // reseve place for the pointer for selected pixels and pixels indices if they found necessary
        span<TRG> selected_pix; // pointer to the selected pixels.
        span<mxInt64> pix_img_idx; // pointer to the selected pixels indices

        //------------------------------------------------------------------
        // Parallel stuff
       // copy to local variables
        auto num_pix = ctx.nPixel_retained;
        auto distribution_size = ctx.distribution_size;
        auto num_OMP_threads = ctx.bin_par_ptr->num_threads;
        auto check_pix_selection = ctx.check_pix_selection;
        auto PIX_STRIDE = ctx.PIX_STRIDE;
        bool align_result = ctx.bin_par_ptr->alignment_matrix.size() == 9;

        // and where these pixel contribution starts in final pixel array 
        std::vector<size_t> thread_contribution_res_start;
        std::vector<std::unordered_set<uint32_t> > tls_unique_ID;
        // for npix, signal and arror
        std::vector<std::vector<bin_accum>> img_tls;
        init_tls_storage<bin_accum>(num_OMP_threads, distribution_size, img_tls);
        // for pixel ranges
        using tlsMem = std::vector<double>;
        std::vector<tlsMem> range_tls_stor(num_OMP_threads, tlsMem(2 * PIX_STRIDE));
        // use span as original min/max range calculation routine expects span
        std::vector<span<double>>p_range_tls;
        std::vector<std::vector<int>> tls_contribution;
        std::vector<std::vector<int>> balanced_idx;
        if (!return_selected_only) {
            p_range_tls.resize(num_OMP_threads);
            for (int i = 0; i < num_OMP_threads; ++i) {
                p_range_tls[i] = span<double>(range_tls_stor[i].data(), 2 * PIX_STRIDE);
                init_min_max_range_calc(p_range_tls[i], PIX_STRIDE);
            }

            tls_unique_ID.resize(num_OMP_threads);
            tls_contribution.resize(num_OMP_threads);
        }
        omp_set_num_threads(num_OMP_threads);

#pragma omp parallel firstprivate(check_pix_selection,PIX_STRIDE,return_selected_only)
        {
            std::vector<double> qu(ctx.COORD_STRIDE);
#pragma omp for schedule(dynamic,1000) reduction(+:num_pix)
            for (int i = 0; i < ctx.data_size; i++) {
                // store actual thread ranges for using it in a pixel copying
                auto n_thread = omp_get_thread_num();
                // drop out coordinates outside of the binning range
                if (ctx.out_of_ranges(i, qu)) {
                    is_pix_selected[i] = false;
                    continue;
                }
                else {
                    is_pix_selected[i] = true;
                }

                // drop out already selected pixels, if requested
                size_t ip0 = i * PIX_STRIDE;
                if (check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0) {
                    is_pix_selected[i] = false;
                    continue;
                }
                num_pix++;

                // calculate location of pixel within the image grid and add values of this pixels to the accumulators
                auto il = ctx.add_pix_to_tls_accum(qu, ip0, img_tls[n_thread]);
                if (!return_selected_only) {
                    //npix_thread_contibution[n_thread]++;
                    tls_contribution[n_thread].push_back(i);
                    // store indices to keep
                    pix_ok_bin_idx[i] = il;
                    // calculate pix ranges
                    calc_pix_ranges<SRC>(p_range_tls[n_thread], ctx.pix_coord, ip0, PIX_STRIDE);
                }
            }
#pragma omp barrier // should be implicit? will do no harm.
#pragma omp single
            {
                ctx.nPixel_retained = num_pix;
            }//single
#pragma omp for schedule(static)
            for (long iimg = 0; iimg < distribution_size; ++iimg) {
                for (int n_thread = 0; n_thread < num_OMP_threads; n_thread++) {
                    npix[iimg] += (double)img_tls[n_thread][iimg].npix;
                    s[iimg] += img_tls[n_thread][iimg].s;
                    e[iimg] += img_tls[n_thread][iimg].e;
                    /* I do not understand why it does not equivalent to row 144 above!!!!
                    * It may be, of course slower for images larger then pixels
                    if (!return_selected_only) {
                        npix_thread_contibution[n_thread] += img_tls[n_thread][iimg].npix;
                    }
                    */
                }
            }
#pragma omp single
            {
                if (!return_selected_only) {
                    merge_tls_ranges(range_tls_stor, ctx.pix_ranges, num_OMP_threads, PIX_STRIDE);
                    balance_copying_load(ctx.nPixel_retained,tls_contribution, balanced_idx, thread_contribution_res_start);
                }
            }//single

            if (!return_selected_only) {
#pragma omp single
                {
                    // allocate memory for pixels to retain.
                    ctx.bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, ctx.nPixel_retained, selected_pix);
                    // allocated memory for pixel indices
                    ctx.bin_par_ptr->pix_img_idx_ptr = allocate_pix_memory<mxInt64>(ctx.nPixel_retained, 1, pix_img_idx);
                }
#pragma omp barrier
                if (ctx.nPixel_retained > 0) {
                    int n_thread = omp_get_thread_num();
                    copy_results_to_final_arraysWithOMP<SRC, TRG>(n_thread,
                        selected_pix, pix_img_idx, tls_unique_ID,
                        align_result, ctx.bin_par_ptr->alignment_matrix, ctx.pix_coord,
                        pix_ok_bin_idx, thread_contribution_res_start, balanced_idx);
                }
#pragma omp barrier
#pragma omp single
                {   // collect all unique ID-s from threads into final unique run_id set
                    for (size_t n_thr = 0; n_thr < num_OMP_threads; ++n_thr) {
                        for (auto runid : tls_unique_ID[n_thr]) {
                            ctx.bin_par_ptr->unique_runID.insert(runid);
                        }
                    }
                }
            }
        } // end of parallel region
    } // end of operator
};
