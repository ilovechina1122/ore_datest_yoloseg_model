cmake_minimum_required(VERSION 3.14)
project(ort_seg_demo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "/home/sunweining/ore_deploy/ort_seg_demo/onnxruntime-linux-x64-1.16.3")

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(ort_seg_demo main.cpp)
target_link_libraries(ort_seg_demo
    ${OpenCV_LIBS}
    onnxruntime
)
# linux filesystem库
if(UNIX)
    target_link_libraries(ort_seg_demo stdc++fs)
endif()

