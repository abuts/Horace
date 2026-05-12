#pragma once
#include <include/CommonCode.h>
template<class SRC, class TRG>
struct processNpixOnly {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {
        std::vector<double> qu(ctx.COORD_STRIDE);
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i, qu))
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid
            size_t il = ctx.pix_position(qu);
            npix[il]++;
        }

    }
};

template<class SRC, class TRG>
struct processNpixOnlyWithOMP {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

        auto num_pix = ctx.nPixel_retained;
        auto distribution_size = ctx.distribution_size;
        auto num_OMP_threads = ctx.bin_par_ptr->num_threads;
        std::vector<std::vector<size_t>> npix1_tls;
        init_tls_storage<size_t>(num_OMP_threads, distribution_size, npix1_tls);
        omp_set_num_threads(num_OMP_threads);

        std::vector<double> qu(ctx.COORD_STRIDE);
#pragma omp parallel     \
        firstprivate(qu)
        {
#pragma omp for schedule(dynamic,1000) reduction(+:num_pix)
            for (long i = 0; i < ctx.data_size; i++) {
                // drop out coordinates outside of the binning range
                if (ctx.out_of_ranges(i, qu))
                    continue;
                num_pix++;

                // calculate location of pixel within the image grid
                size_t il = ctx.pix_position(qu);
                auto n_thread = omp_get_thread_num();
                npix1_tls[n_thread][il]++;
            }
#pragma omp barrier
#pragma omp for schedule(static)
            for (long i = 0; i < distribution_size; i++) {
                for (int n_thread = 0; n_thread < num_OMP_threads; n_thread++) {
                    npix[i] += npix1_tls[n_thread][i];
                }
            }
            ctx.nPixel_retained = num_pix;
        } // end parallel
    };
};

