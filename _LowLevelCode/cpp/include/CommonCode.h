#pragma once
//
//
#include <limits>
//
#include <mex.h>
#include <matrix.h>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <memory>
#include <mutex>
#include <type_traits>
//#include <omp_guard.hpp>

#ifndef _OPENMP
inline void omp_set_num_threads(int nThreads) {};
#define omp_get_num_threads() 1
#define omp_get_max_threads() 1
#define omp_get_thread_num()  0
#else
#include <omp.h>
#endif

#if defined(_OPENMP) && _OPENMP >= 200805
#define omp3_available
#endif

// check span supported
//#undef __cpp_lib_span
#if defined(__cpp_lib_span)
    #include <span>
    template<class T, std::size_t Extent = std::dynamic_extent>
    using span = std::span<T, Extent>;
#else
    #include "./span.hpp"
#endif


#ifdef __cplusplus
extern "C" bool utIsInterruptPending();
extern "C" bool ioFlush(void);
#else
extern bool utIsInterruptPending();
extern bool ioFlush(void);
#endif


enum pix_flds
{
    u1 = 0, //      -|
    u2 = 1, //       |  Coordinates of pixel in the pixel projection axes. A constants within code rely on this set-up
    u3 = 2, //       |
    u4 = 3, //      -|
    irun = 4, //        Run index in the header block from which pixel came
    idet = 5, //        Detector group number in the detector listing for the pixel
    ien = 6, //         Energy bin number for the pixel in the array in the (irun)th header
    iSign = 7, //      Signal array
    iErr = 8, //         Error array (variance i.e. error bar squared)
    PIX_WIDTH = 9  // Number of pixel fields
};

// Copy pixels from source to target array
template<class SRC,class TRG> 
inline size_t copy_pixels(span<const SRC> pixel_data, long source_pos, span<TRG> pix_sorted, size_t targ_pos)
{
    //
    targ_pos *= pix_flds::PIX_WIDTH; // each position in a grid cell corresponds to a pixel of the size PIX_WIDTH;
    // in the pixels array
    source_pos *= pix_flds::PIX_WIDTH;

    for (size_t i = 0; i < pix_flds::PIX_WIDTH; i++) {
        pix_sorted[targ_pos + i] = static_cast<TRG>(pixel_data[source_pos + i]);
    }
    return targ_pos;
};
// Align and copy pixels from source to target array template <class SRC, class TRG>
template <class SRC, class TRG>
inline size_t align_and_copy_pixels(const std::vector<double> &al_matr,span<const SRC> pixel_data, long source_pos, span<TRG> pix_sorted, size_t targ_pos)
{
    //
    targ_pos *= pix_flds::PIX_WIDTH; // each position in a grid cell corresponds to a pixel of the size PIX_WIDTH;
    // in the pixels array
    source_pos *= pix_flds::PIX_WIDTH;
    for (size_t i = 0; i < 3; i++) {
        double accum(0);
        for (size_t j = 0; j < 3; j++) {
            accum += double((pixel_data[source_pos + j])) * al_matr[j*3+i];
        }
        pix_sorted[targ_pos + i] = static_cast<TRG>(accum);
    }

    for (size_t i = 3; i < pix_flds::PIX_WIDTH; i++) {
        pix_sorted[targ_pos + i] = static_cast<TRG>(pixel_data[source_pos + i]);
    }
    return targ_pos;
};

/* Initialize pixel ranges for calculating correct range.
 *  This means assigning to min/max holders values which are completely invalid, namely
 *  minima equal to maximal double value and maxima equal to minimal double value */
inline void init_min_max_range_calc(span<double>& pix_ranges, size_t PIX_STRIDE)
{
    auto max_range = std::numeric_limits<double>::max();
    auto min_range = -max_range;
    for (size_t i = 0; i < PIX_STRIDE; i++) {
        pix_ranges[2 * i] = max_range;
        pix_ranges[2 * i + 1] = min_range;
    }
};

// identify range of all pixel coordinates for given initial pixels position
template <class SRC>
void inline calc_pix_ranges(span<double>& pix_ranges, span<const SRC> &pix_data,size_t ip0,size_t PIX_STRIDE)
{
    for (size_t j = 0; j < PIX_STRIDE; j++) {
        pix_ranges[2 * j] = std::min(pix_ranges[2 * j], (double)pix_data[ip0 + j]);
        pix_ranges[2 * j + 1] = std::max(pix_ranges[2 * j + 1], (double)pix_data[ip0 + j]);
    }
};
// merge together all partial pixel ranges, calculated by threads
void inline merge_tls_ranges(std::vector<std::vector<double>> &tls_ranges,span<double>& pix_ranges,size_t n_threads, size_t PIX_STRIDE)
{
    for (int i = 0; i < n_threads; i++) {
        for (size_t j = 0; j < PIX_STRIDE; j++) {
            pix_ranges[2 * j] = std::min(pix_ranges[2 * j], tls_ranges[i][2 * j]);
            pix_ranges[2 * j + 1] = std::max(pix_ranges[2 * j + 1], tls_ranges[i][2 * j + 1]);
        }
    }
};

/* allocate pixels memory to keep with MATLAB and use in C++ code
* Parameters:
* PIX_WIDTH    -- number of elements in a pixel. Usually constant
* n_pixels     -- number of pixels to allocate memory for.
* mem_wrap     -- span container, wrapping raw pointer to MATLAB array
*                 with pixels memory
* Returns:
* Pointer to MATLAB array with appropriate type of memory.
*/
template<class TRG> 
mxArray* allocate_pix_memory(size_t PIX_WIDTH, size_t n_pixels, span<TRG>& mem_wrap)
{ 
    mxArray* mxData_ptr(nullptr);
    TRG* data_ptr(nullptr);
    std::string mem_name;
    if constexpr (std::is_same_v<TRG, double>) {
        mxData_ptr = mxCreateDoubleMatrix(PIX_WIDTH, n_pixels, mxREAL);
        if (mxData_ptr)
            data_ptr = mxGetPr(mxData_ptr);
        mem_name = "resulting binned pixels of type 'double'";
    } else if constexpr (std::is_same_v<TRG, float>) {
        mxData_ptr = mxCreateNumericMatrix(PIX_WIDTH, n_pixels, mxSINGLE_CLASS, mxREAL);
        if (mxData_ptr)
            data_ptr = reinterpret_cast<float*>(mxGetPr(mxData_ptr));
        mem_name = "resulting binned pixels of type 'signle'";
    } else if constexpr (std::is_same_v<TRG, mxInt64>) {
        mxData_ptr = mxCreateNumericMatrix(PIX_WIDTH, n_pixels, mxINT64_CLASS, mxREAL);
        if (mxData_ptr)
            data_ptr = reinterpret_cast<mxInt64 *>(mxGetPr(mxData_ptr));
        mem_name = "resulting array of indices of type 'UINT64'";
    } else if constexpr(std::is_same_v<TRG, mxLogical>) {
        mxData_ptr = mxCreateLogicalMatrix(PIX_WIDTH, n_pixels);
        if (mxData_ptr)
            data_ptr = mxGetLogicals(mxData_ptr);
        mem_name = "resulting array of logical indices";
    } else {
        mexErrMsgIdAndTxt("HORACE:bin_pixels_c:runtime_error",
            "Attempt to allocate memory for unsupported type of variable. Only float and double are supported");
    }
    if (mxData_ptr == nullptr) {
        std::stringstream buf;
        buf << "Can not allocate memory for: " << n_pixels << mem_name;
        mexErrMsgIdAndTxt("HORACE:bin_pixels_c:runtime_error",
            buf.str().c_str());
    }
    mem_wrap = span<TRG>(data_ptr, PIX_WIDTH * n_pixels);
    return mxData_ptr;
};

// nullify input mxArray (used as accumulator)
inline void nullify_array(const mxArray* mxData_ptr) {

    size_t n_elements = mxGetNumberOfElements(mxData_ptr);
    auto pData = mxGetPr(mxData_ptr);
    for (size_t i = 0; i < n_elements;i++) {
        pData[i] = 0;
    }
}

//* Possible prototype for a generic function
template<class T>
T getMatlabScalar(const mxArray* pPar, const char* const fieldName) {
    if (pPar == NULL) {
        std::stringstream buf;
        buf << " The input variable: " << *fieldName << " has to be defined\n";

        mexErrMsgIdAndTxt("HORACE:getMatlabScalar_mex:invalid_argument",
            buf.str().c_str());
    }
    if (mxGetM(pPar) != 1 || mxGetN(pPar) != 1) {
        std::stringstream buf;
        buf << " The input variable: " << *fieldName << " has to be a scalar\n";
        mexErrMsgIdAndTxt("HORACE:getMatlabScalar_mex:invalid_argument",
            buf.str().c_str());
    }
    return static_cast<T>(*mxGetPr(pPar));
};
/* Accumulators used in tls storage to keep part of image, calculated by every
** OMP thread */
struct bin_accum{
    size_t npix;
    double s;
    double e;
    bin_accum(size_t val) :
        npix(val),s(double(val)),e(double(val))
    {}
};
// structure used in storing contributing pixel indices within TLS storage
struct idx_accum {
    int pix_idx;    // position of contributed index in original pixel array 
    size_t img_idx; // index of pixel position in image array calculated through binning
    idx_accum():
        pix_idx(-1), img_idx(0)
    {}
    idx_accum(int pix_idx, size_t img_idx) {
        this->pix_idx = pix_idx;
        this->img_idx = img_idx;
    }
};

template<class T>
void init_tls_storage(size_t num_OMP_threads, size_t distribution_size,std::vector<std::vector<T> > &tls_storage) {
    try {
        tls_storage.resize(num_OMP_threads,
            std::vector<T>(distribution_size, 0)
        );
    }
    catch (...) {
        std::stringstream buf;
        buf << "Can not allocate TLS for size: " << distribution_size << " distribution.\nDecrease number of threads or perform serial calculations";
        mexErrMsgIdAndTxt("HORACE:bin_pixels_c:runtime_error",
            buf.str().c_str());
    }
};
