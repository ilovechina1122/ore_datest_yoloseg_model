cmake_minimum_required(VERSION 3.10)
project(trt_seg_infer)

set(CMAKE_CXX_STANDARD 17)

# ---------- 按你的机器改 ----------
# TensorRT 解压目录 (已确认正确)
set(TRT_ROOT "/home/sunweining/下载/TensorRT-8.6.1.6")
# CUDA: Debian 系安装, 头文件在 /usr/include, 库在 /usr/lib/x86_64-linux-gnu
set(CUDA_ROOT "/usr")
# OpenCV 的 cmake 目录 (已查到 4.5.4)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")

find_package(OpenCV REQUIRED)

# 不使用 find_package(TensorRT), 因为 TRT 解压包里没有 TensorRTConfig.cmake。
# 直接手动指定头文件和链接库即可。

add_executable(trt_seg_infer main.cpp)

target_include_directories(trt_seg_infer PRIVATE
    ${TRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(trt_seg_infer PRIVATE
    ${TRT_ROOT}/lib
    /usr/lib/x86_64-linux-gnu
)

target_link_libraries(trt_seg_infer PRIVATE
    nvinfer
    cudart
    ${OpenCV_LIBS}
)
