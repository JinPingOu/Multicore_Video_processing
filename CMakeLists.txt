
cmake_minimum_required(VERSION 3.0.0)
project(multicore_final_project VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(contour_with_cv contour_with_cv.cpp)

#target_link_libraries( multicore_final_project ${OpenCV_LIBS} )
target_link_libraries(contour_with_cv ${OpenCV_LIBS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75)

#message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
