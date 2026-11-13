cmake_minimum_required(VERSION 3.18)
project(rt_video_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Orin = SM 8.7, Thor = SM 11.0
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

# TensorRT (JetPack installs into the system multiarch paths)
find_path(TRT_INCLUDE NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include)
find_library(NVINFER nvinfer PATHS /usr/lib/aarch64-linux-gnu)
# NvBufSurface (zero-copy NVMM path)
find_path(NVBUF_INCLUDE nvbufsurface.h PATHS /usr/src/jetson_multimedia_api/include)
find_library(NVBUFSURFACE nvbufsurface PATHS /usr/lib/aarch64-linux-gnu/tegra)
find_library(EGL_LIB EGL)

add_executable(rtinfer
  src/main.cpp
  src/gst_source.cpp
  src/trt_engine.cpp
  src/profiler.cpp
  src/preprocess.cu)

target_include_directories(rtinfer PRIVATE src ${TRT_INCLUDE} ${NVBUF_INCLUDE})
target_link_libraries(rtinfer PRIVATE
  PkgConfig::GST ${NVINFER} ${NVBUFSURFACE} ${EGL_LIB}
  CUDA::cudart CUDA::cuda_driver)
target_compile_options(rtinfer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-deprecated-declarations>)
