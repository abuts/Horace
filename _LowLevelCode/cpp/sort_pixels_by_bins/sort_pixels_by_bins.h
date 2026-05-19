#pragma once


#include <include/CommonCode.h>
#include <algorithm>
enum Input_Arguments {
    Pixel_data,
    Pixel_Indexes,
    Pixel_Distribution,
    keep_type,
    N_INPUT_Arguments
};
enum Out_Arguments {
    Pixels_Sorted,
    Pixels_range,
    N_OUTPUT_Arguments
};
/* What kind of input/output types the routine supports*/
enum InputOutputTypes {
    Pix8IndIOut8, // Double pixels, Int64 indexes, double output
    Pix8IndDOut8, // Double pixels, Double indexes, double output
    Pix4IndIOut8,
    Pix4IndDOut8, // Float pixels Int64 indexes double output
    Pix4IndIOut4, // Float pixels Int64 indexes float output
    Pix4IndDOut4,
    Pix4Ind4Out4, // float pixels float indexes, fload output
    Pix4Ind4Out8, // float pixels float indexes, double output
    Pix8Ind4Out8, // double pixels float indexes, double output
    ERROR,
    N_InputCases
};
enum InputIndexesType {
    IndI64,  // input indexes are unit64 type
    IndD64,  // input indexes are double64 type
    IndF32,  // input indexes are float32 type.
    N_InputIndexes
};

#define iRound(x)  (int)floor((x)+0.5)

InputOutputTypes process_types(bool float_pix, InputIndexesType index_type, bool double_out);

//
template<class SRC, class N, class TRG>
void sort_pixels_by_bins( TRG * const pPixelSorted, size_t nPixelsSorted, double *const pPixRange,
    std::vector<const void *> &PixelData, std::vector<size_t> &NPixels,
    std::vector<const void *> &PixelIndexes, const std::vector<size_t> &NIndexes,
    double const *const pCellDens, size_t distribution_size,
    size_t *const ppInd) {

    const size_t PIX_STRIDE = static_cast<size_t>(pix_flds::PIX_WIDTH);
    span<TRG> pixel_sorted(pPixelSorted, nPixelsSorted* PIX_STRIDE);

    ppInd[0] = 0;
    for (size_t i = 1; i < distribution_size; i++) {   // calculate the ranges of the cell arrays
        ppInd[i] = ppInd[i - 1] + (size_t)pCellDens[i - 1]; // the next cell starts from the the previous one
    };                                      // plus the number of pixels in the previous cell
    if (ppInd[distribution_size - 1] + (size_t)pCellDens[distribution_size - 1] != nPixelsSorted) {
        mexErrMsgIdAndTxt("HORACE:sort_pixels_by_bins:invalid_argument",\
            "pixels data and their cell distributions are inconsistent");
    }
    bool calc_pix_range(false);
    span<double> pix_range;
    if (pPixRange) {
        calc_pix_range = true;
        pix_range = span<double>(pPixRange, 2 * pix_flds::PIX_WIDTH);
        init_min_max_range_calc(pix_range, (size_t)pix_flds::PIX_WIDTH);
    }


    //#pragma omp parallel
    for(size_t nblock=0; nblock < PixelIndexes.size();nblock++)
    {
        size_t nBlockInd = NIndexes[nblock];
        const N* pCellInd = reinterpret_cast<const N*>(PixelIndexes[nblock]);
        if (pCellInd == nullptr)continue;

        auto pPixData= reinterpret_cast<const SRC *>(PixelData[nblock]);
        if (pPixData == nullptr)continue;
        auto pix_data = span<const SRC>(pPixData, NPixels[nblock]);

        // sort pixels according to cells
        for (size_t j = 0; j < nBlockInd ; j++) {
            size_t ind = (size_t)(pCellInd[j] - 1); // -1 as Matlab arrays start from one;
            auto cell_pix_ind = ppInd[ind]++;       // pixel position within a cell

            if (calc_pix_range) {
                size_t pixpos = PIX_STRIDE*j;
                calc_pix_ranges<SRC>(pix_range, pix_data,pixpos,PIX_STRIDE);
            }
            copy_pixels<SRC, TRG>(pix_data, j, pixel_sorted, cell_pix_ind); // copy all pixel data into the location requested
        }
    }
}
