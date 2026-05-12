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

template<class SRC, class TRG>
struct processWithSortingWithOMP {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

        span<mxInt64> pix_ok_bin_idx(ctx.bin_par_ptr->pix_ok_bin_idx);
        span<size_t> npix1(ctx.bin_par_ptr->npix1.data(), ctx.bin_par_ptr->npix1.size());

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

#pragma omp parallel     \
        firstprivate(check_pix_selection,PIX_STRIDE )
        {
            std::vector<double> qu(ctx.COORD_STRIDE);
#pragma omp for schedule(dynamic,1000) reduction(+:num_pix)
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
                    // calculate location of pixel within the image grid and add values of this pixels to the accumulators
                auto n_thread = omp_get_thread_num();
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

        // allocate memory for pixels to retain contributing pixels.
        span<TRG> sorted_pix; // pointer to the actual data position.
        ctx.bin_par_ptr->pix_ok_ptr = allocate_pix_memory<TRG>(pix_flds::PIX_WIDTH, ctx.nPixel_retained, sorted_pix);
        // calculate ranges of cells to place pixels
        span<size_t> bin_start(ctx.bin_par_ptr->npix_bin_start);
        bin_start[0] = 0;
        npix[0] += npix1[0];
        if (distribution_size > 1) {
            for (size_t i = 1; i < distribution_size; i++) {
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
