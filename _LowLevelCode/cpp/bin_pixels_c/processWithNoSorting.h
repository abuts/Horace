#pragma once
#include <include/CommonCode.h>
#include "copy_results_to_final_arrays.h"

template<class SRC, class TRG>
struct processWithNoSorting {

    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        // access working bufer persistent between calls to this function:
        const auto bin_par_ptr = ctx.bin_par_ptr;
        // pixel rejection cache
        span<mxInt64> pix_ok_bin_idx;
        bin_par_ptr->init_pix_ok_cache(pix_ok_bin_idx);

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
            auto il = ctx.add_pix_to_accumulators(qu, ip0, npix, s, e);

            // store indices of contributing pixels
            pix_ok_bin_idx[i] = il;
            // calculate pix ranges
            calc_pix_ranges<SRC>(ctx.pix_ranges, ctx.pix_coord, ip0, ctx.PIX_STRIDE);
        }

        copy_results_to_final_arrays<SRC, TRG>(ctx.bin_par_ptr, ctx.pix_coord,
            ctx.data_size, ctx.nPixel_retained, pix_ok_bin_idx);

    }
};

template<class SRC, class TRG>
struct processWithNoSortingWithOMP {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

        // reseve placeholder for the pointer for selected pixels and pixels indices
        span<TRG> selected_pix; // pointer to the selected pixels.
        span<mxInt64> pix_img_idx; // pointer to the selected pixels indices
        // things which calculated at the end but need to be placed here to be visible
        // in parallel region
        std::vector<std::vector<idx_accum> > balanced_idx;
        // and where these pixel contribution starts in final pixel array 
        std::vector<size_t> thread_contribution_res_start;
        std::vector<std::unordered_set<uint32_t> > tls_unique_ID;

        // Parallel stuff
       // copy to local variables
        auto num_pix = ctx.nPixel_retained;
        auto distribution_size = ctx.distribution_size;
        auto num_OMP_threads = ctx.bin_par_ptr->num_threads;
        auto check_pix_selection = ctx.check_pix_selection;
        auto PIX_STRIDE = ctx.PIX_STRIDE;
        bool align_result = ctx.bin_par_ptr->alignment_matrix.size() == 9;

        // for npix, signal and arror
        std::vector<std::vector<bin_accum>> img_tls;
        init_tls_storage<bin_accum>(num_OMP_threads, distribution_size, img_tls);
        // for pixel ranges
        using tlsMem = std::vector<double>;
        std::vector<tlsMem> range_tls_stor(num_OMP_threads, tlsMem(2 * PIX_STRIDE));
        // use span as original min/max range calculation routine expects span
        std::vector<span<double>>p_range_tls;
        std::vector<std::vector<idx_accum> > tls_contribution;

        p_range_tls.resize(num_OMP_threads);
        for (int i = 0; i < num_OMP_threads; ++i) {
            p_range_tls[i] = span<double>(range_tls_stor[i].data(), 2 * PIX_STRIDE);
            init_min_max_range_calc(p_range_tls[i], PIX_STRIDE);
        }
        tls_contribution.resize(num_OMP_threads);

        auto chunk_size = set_omp_scheduling(ctx.bin_par_ptr);
        omp_set_num_threads(num_OMP_threads);

#pragma omp parallel firstprivate(check_pix_selection,PIX_STRIDE)
        {
            // identify id of parallel worker
            auto n_thread = omp_get_thread_num();
            std::vector<double> qu(ctx.COORD_STRIDE);
#ifdef omp3_available
#pragma omp for schedule(runtime) reduction(+:num_pix)
#else
#ifdef DISABLE_DYNAMIC_SHEDULER
#pragma omp for schedule(static,chunk_size) reduction(+:num_pix)
#else
    #pragma omp for schedule(dynamic,chunk_size) reduction(+:num_pix)
#endif
#endif

            for (int i = 0; i < ctx.data_size; i++) {
                // drop out coordinates outside of the binning range
                if (ctx.out_of_ranges(i, qu))
                    continue;

                // drop out already selected pixels, if requested
                size_t ip0 = i * PIX_STRIDE;
                if (check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0)
                    continue;
                num_pix++;

                // calculate location of pixel within the image grid and add values of this pixels to the accumulators
                // store indices of contributing pixels
                auto il = ctx.add_pix_to_tls_accum(qu, ip0, img_tls[n_thread]);
                tls_contribution[n_thread].push_back(idx_accum(i, il));
                // calculate pix ranges
                calc_pix_ranges<SRC>(p_range_tls[n_thread], ctx.pix_coord, ip0, PIX_STRIDE);
            } // end of for loop.
#pragma omp barrier // should be implicit? will do no harm.
#pragma omp for schedule(static)  // Combine thread-calculated images to final image
            for (long iimg = 0; iimg < distribution_size; ++iimg) {
                for (int n_thread = 0; n_thread < num_OMP_threads; n_thread++) {
                    npix[iimg] += (double)img_tls[n_thread][iimg].npix;
                    s[iimg] += img_tls[n_thread][iimg].s;
                    e[iimg] += img_tls[n_thread][iimg].e;
                }
            }
#pragma omp single
            {
                // retrieve thread-combined number of pixels
                ctx.nPixel_retained = num_pix;
                tls_unique_ID.resize(num_OMP_threads);
                // merge pixel ranges obtained from each thrad
                merge_tls_ranges(range_tls_stor, ctx.pix_ranges, num_OMP_threads, PIX_STRIDE);
                balance_copying_load(ctx.nPixel_retained, tls_contribution, balanced_idx, thread_contribution_res_start);

                // allocate memory for pixels to retain.
                ctx.bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, ctx.nPixel_retained, selected_pix);
                // allocated memory for contributing pixel indices
                ctx.bin_par_ptr->pix_img_idx_ptr = allocate_pix_memory<mxInt64>(ctx.nPixel_retained, 1, pix_img_idx);

            }//single
#pragma omp barrier
            if (ctx.nPixel_retained > 0) {  // per-thread, copy data to target
                copy_results_to_final_arraysWithOMP<SRC, TRG>(n_thread,
                    selected_pix, pix_img_idx, tls_unique_ID,
                    align_result, ctx.bin_par_ptr->alignment_matrix, ctx.pix_coord,
                    thread_contribution_res_start, balanced_idx);
            }
        } // end of parallel region
               // collect all unique ID-s from threads into final unique run_id set
        for (size_t n_thr = 0; n_thr < num_OMP_threads; ++n_thr) {
            for (auto runid : tls_unique_ID[n_thr]) {
                ctx.bin_par_ptr->unique_runID.insert(runid);
            }
        }


    }// end of () operator.
};

