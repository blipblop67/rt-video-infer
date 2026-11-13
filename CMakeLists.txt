cmake_minimum_required(VERSION 3.18)
project(rt_video_infer LANGUAGES CXX)

# The app needs a Jetson (CUDA, TensorRT, GStreamer, NvBufSurface). The pure-logic core and its
# unit tests build anywhere, which is what CI runs: -DRTINFER_BUILD_APP=OFF
option(RTINFER_BUILD_APP "Build the rtinfer app (CUDA, TensorRT, GStreamer required)" ON)
option(RTINFER_BUILD_TESTS "Build unit tests for the pure-logic core" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Dependency-free logic shared by the app and the tests.
add_library(rtcore STATIC src/source_spec.cpp)
target_include_directories(rtcore PUBLIC src)
target_compile_options(rtcore PRIVATE -Wall -Wextra)

if(RTINFER_BUILD_APP)
  # Orin = SM 8.7, Thor = SM 11.0
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  enable_language(CUDA)

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

  target_include_directories(rtinfer PRIVATE ${TRT_INCLUDE} ${NVBUF_INCLUDE})
  target_link_libraries(rtinfer PRIVATE
    rtcore PkgConfig::GST ${NVINFER} ${NVBUFSURFACE} ${EGL_LIB}
    CUDA::cudart CUDA::cuda_driver)
  target_compile_options(rtinfer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-deprecated-declarations>)
endif()

if(RTINFER_BUILD_TESTS)
  enable_testing()
  add_executable(unit_tests
    tests/test_main.cpp
    tests/test_source_spec.cpp
    tests/test_stats.cpp
    tests/test_topk.cpp
    tests/test_batch_plan.cpp)
  target_link_libraries(unit_tests PRIVATE rtcore)
  target_compile_options(unit_tests PRIVATE -Wall -Wextra)
  add_test(NAME unit_tests COMMAND unit_tests)
endif()
