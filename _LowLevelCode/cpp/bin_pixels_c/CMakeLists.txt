set(
    SRC_FILES
    "bin_pixels_c.cpp"
    "BinningArg.cpp"
)

set(
    HDR_FILES
    "bin_pixels.h"
    "BinningArg.h"   
    "copy_results_to_final_arrays.h"
    "CommonBinCode.h"
    "processNpixOnly.h"
    "processSigErr.h"
    "processWithSorting.h"
    "processWithNoSorting.h"
    "processWithNoSortSel.h"
    "${CXX_SOURCE_DIR}/include/CommonCode.h"
    "${CXX_SOURCE_DIR}/include/span.hpp"
    "${CXX_SOURCE_DIR}/include/MatlabCppClassHolder.hpp"
)

set(MEX_NAME "bin_pixels_c")
pace_add_mex(
    NAME "${MEX_NAME}"
    SRC "${SRC_FILES}" "${HDR_FILES}"
)
if(OPENMP_FOUND)
  message(" ADDITIONAL CPP PARAMETERS: ${OMP_TYPES}")
  target_compile_definitions("${MEX_NAME}" PRIVATE ${OMP_TYPES})
  target_compile_options("${MEX_NAME}" PRIVATE ${OpenMP_CXX_FLAGS})
  if (DISABLE_DYNAMIC_SCHEDULER)
        target_link_options(${MEX_NAME} PRIVATE -static-libgomp)
  else()
    target_link_options("${MEX_NAME}" PRIVATE ${OpenMP_EXE_LINKER_FLAGS})  
  endif()
endif()
