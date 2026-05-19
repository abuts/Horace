#pragma once
#include <include/CommonCode.h>
// calculate npix, signal and error image-contribution from various types of pixel data
template<class SRC, class TRG>
struct processSigErr {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
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
            ctx.add_pix_to_accumulators(qu, ip0, npix, s, e);
        }
    }
};

template<class SRC, class TRG>
struct processSigErrWithOMP {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        auto num_pix = ctx.nPixel_retained;
        auto distribution_size = ctx.distribution_size;
        auto num_OMP_threads = ctx.bin_par_ptr->num_threads;
        auto check_pix_selection = ctx.check_pix_selection;
        auto PIX_STRIDE = ctx.PIX_STRIDE;
        std::vector<std::vector<bin_accum>> img_tls;
        init_tls_storage<bin_accum>(num_OMP_threads, distribution_size, img_tls);
        auto chunk_size = set_omp_scheduling(ctx.bin_par_ptr);
        omp_set_num_threads(num_OMP_threads);

        std::vector<double> qu(ctx.COORD_STRIDE);
#pragma omp parallel     \
        firstprivate(qu,check_pix_selection,PIX_STRIDE )
        {
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
                auto n_thread = omp_get_thread_num();
                ctx.add_pix_to_tls_accum(qu, ip0, img_tls[n_thread]);
            }
#pragma omp barrier
#pragma omp for schedule(static)
            for (long i = 0; i < distribution_size; i++) {
                for (int n_thread = 0; n_thread < num_OMP_threads; n_thread++) {
                    npix[i] += (double)img_tls[n_thread][i].npix;
                    s[i] += img_tls[n_thread][i].s;
                    e[i] += img_tls[n_thread][i].e;
                }
            }
            ctx.nPixel_retained = num_pix;
        }
    }
};
