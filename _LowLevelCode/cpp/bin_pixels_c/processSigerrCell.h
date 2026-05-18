#pragma once
#include <include/CommonCode.h>
#include "copy_results_to_final_arrays.h"
// calculate npix, signal and error cell-contribution from data represented as cellarray of pixel data blocks
template<class SRC, class TRG>
struct processSigerrCell {
    void operator()(CommonBinCode<SRC, TRG>& ctx, span<double>& npix, span<double>& s, span<double>& e) const {

        std::vector<double*> accum_ptr(3);
        accum_ptr[0] = s.data();
        accum_ptr[1] = e.data();
        accum_ptr[2] = npix.data();
        auto n_cells_to_bin = ctx.bin_par_ptr->n_Cells_to_bin;
        bool npix_acc_separate = n_cells_to_bin < 3; // values for npix accumulators may be provided in separate array
        // if they are not, calculate this value anyway
        std::vector<const double*> cell_data_ptr(n_cells_to_bin, nullptr);

        // fill in cell_data_ptr with pointers to contents of cell data to bin
        for (auto i = 0; i < n_cells_to_bin; i++) {
            const mxArray* cell_array_ptr = mxGetCell(ctx.bin_par_ptr->all_pix_ptr, i);
            cell_data_ptr[i] = mxGetPr(cell_array_ptr);
        }
        std::vector<double> qu(ctx.COORD_STRIDE);
        for (long i = 0; i < ctx.data_size; i++) {
            // drop out coordinates outside of the binning range
            if (ctx.out_of_ranges(i, qu))
                continue;
            ctx.nPixel_retained++;

            // calculate location of pixel within the image grid
            size_t il = ctx.pix_position(qu);

            if (npix_acc_separate) {
                // calculate npix accumulators separately if their value is not provided as input
                npix[il]++;
            }

            // calculate signal and, if necessary error accumulators
            for (auto j = 0; j < n_cells_to_bin; j++) {
                auto acc_ptr = accum_ptr[j];
                auto data_ptr = cell_data_ptr[j];
                acc_ptr[il] += data_ptr[i];
            }
        }
    }
};

