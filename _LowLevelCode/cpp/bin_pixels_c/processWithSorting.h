#pragma once
#include <include/CommonCode.h>

template<class SRC, class TRG>
struct processWithSorting {

    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        // access working bufer persistent between calls to this function:
        const auto bin_par_ptr = ctx.bin_par_ptr;
        // pixel rejection cache
        span<mxInt64> pix_ok_bin_idx;
        bin_par_ptr->init_pix_ok_cache(pix_ok_bin_idx);
        // accumulators needed for sorting pixels according to bins
        span<size_t> npix1;
        span<size_t> bin_start;
        bin_par_ptr->init_npix1_step_cache(npix1, bin_start);

        size_t num_pix(0);
        std::vector<double> qu(ctx.COORD_STRIDE);
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i, qu))
                continue;
            // drop out already selected pixels, if requested
            size_t ip0 = i * ctx.PIX_STRIDE;
            if (ctx.check_pix_selection && ctx.pix_coord[ip0 + pix_flds::idet] < 0)
                continue;
            num_pix++;

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
        bin_start[0] = 0;
        npix[0] += npix1[0];
        if (ctx.distribution_size > 1) {
            for (size_t i = 1; i < ctx.distribution_size; i++) {
                bin_start[i] = bin_start[i - 1] + npix1[i - 1]; // range of cell to place pixels
                npix[i] += npix1[i]; // increase multi-call accumulators
            }
        }

        ctx.nPixel_retained = num_pix;
        bool keep_unique_id = ctx.bin_par_ptr->binMode == opModes::sort_and_uid;
        copy_pixels_to_final_arrays<SRC, TRG>(ctx.bin_par_ptr, ctx.pix_coord, num_pix,
            keep_unique_id, pix_ok_bin_idx, bin_start);
    }
};

template<class SRC, class TRG>
struct processWithSortingWithOMP {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        // access working bufer persistent between calls to this function:
        const auto bin_par_ptr = ctx.bin_par_ptr;


        span<mxInt64> pix_ok_bin_idx;
        bin_par_ptr->init_pix_ok_cache(pix_ok_bin_idx);
        // accumulators needed for sorting pixels according to bins
        span<size_t> npix1;
        span<size_t> bin_start;
        ctx.bin_par_ptr->init_npix1_step_cache(npix1, bin_start);

        // copy to local variables
        auto num_pix = ctx.nPixel_retained;
        auto distribution_size = ctx.distribution_size;
        auto num_OMP_threads = ctx.bin_par_ptr->num_threads;
        auto check_pix_selection = ctx.check_pix_selection;
        auto PIX_STRIDE = ctx.PIX_STRIDE;
        // alocate TLS storage:
        // for npix, signal and arror
        std::vector<std::vector<bin_accum>> img_tls;
        init_tls_storage<bin_accum>(num_OMP_threads, distribution_size, img_tls);
        // for pixel ranges
        using tlsMem = std::vector<double>;
        std::vector<tlsMem> range_tls_stor(num_OMP_threads, tlsMem(2 * PIX_STRIDE));
        // use span as original min/max range calculation routine expects span
        std::vector<span<double>>p_range_tls(num_OMP_threads);
        for (int i = 0; i < num_OMP_threads; ++i) {
            p_range_tls[i] = span<double>(range_tls_stor[i].data(), 2 * PIX_STRIDE);
            init_min_max_range_calc(p_range_tls[i], PIX_STRIDE);
        }
        omp_set_num_threads(num_OMP_threads);
        auto chunk_size = set_omp_scheduling(ctx.bin_par_ptr);

#pragma omp parallel     \
        firstprivate(check_pix_selection,PIX_STRIDE )
        {
            std::vector<double> qu(ctx.COORD_STRIDE);
            auto n_thread = omp_get_thread_num();
#ifdef DISABLE_DYNAMIC_SCHEDULER
#pragma omp for schedule(static) reduction(+:num_pix)
#else
#ifdef omp3_available
#pragma omp for schedule(runtime) reduction(+:num_pix)
#else
#pragma omp for schedule(dynamic,chunk_size) reduction(+:num_pix)
#endif
#endif
            for (long i = 0; i < ctx.data_size; i++) {
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
                pix_ok_bin_idx[i] = ctx.add_pix_to_tls_accum(qu, ip0, img_tls[n_thread]);
                // calculate pix ranges
                calc_pix_ranges<SRC>(p_range_tls[n_thread], ctx.pix_coord, ip0, PIX_STRIDE);
            }
#pragma omp barrier // should be implicit? will do no harm.
#pragma omp for schedule(static)
            for (long i = 0; i < distribution_size; i++) {
                for (int n_thread = 0; n_thread < num_OMP_threads; n_thread++) {
                    npix1[i] += img_tls[n_thread][i].npix;
                    s[i] += img_tls[n_thread][i].s;
                    e[i] += img_tls[n_thread][i].e;
                }
            }
        } // end of parallel region
        ctx.nPixel_retained = num_pix;
        merge_tls_ranges(range_tls_stor, ctx.pix_ranges, num_OMP_threads, PIX_STRIDE);

        // calculate ranges of cells to place pixels
        bin_start[0] = 0;
        npix[0] += npix1[0];
        if (distribution_size > 1) {
            for (size_t i = 1; i < distribution_size; i++) {
                bin_start[i] = bin_start[i - 1] + npix1[i - 1]; // range of cell to place pixels
                npix[i] += npix1[i]; // increase multi-call accumulators
            }
        }

        bool keep_unique_id = ctx.bin_par_ptr->binMode == opModes::sort_and_uid;
        copy_pixels_to_final_arrays<SRC, TRG>(ctx.bin_par_ptr, ctx.pix_coord, num_pix,
            keep_unique_id, pix_ok_bin_idx, bin_start);
    }
};
