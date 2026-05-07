# 设置最低版本号
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# 设置项目名称
project(rk3588-demo VERSION 0.0.1 LANGUAGES CXX)

# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")


# 设置编译器
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

link_directories(
    mpp_api
    mk_api
    /usr/local/lib
)

#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)

message(STATUS "RKNN API include path: ${RKNN_API_INCLUDE_PATH}")

# rknn_api lib 路径 - 优先使用项目中的库，而不是系统库
if(EXISTS ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
    set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
    message(STATUS "Using project RKNN lib: ${RKNN_API_LIB_PATH}")
elseif(EXISTS "/usr/lib/librknnrt.so")
    set(RKNN_API_LIB_PATH "/usr/lib/librknnrt.so")
    message(WARNING "Using system RKNN lib (may have debug logs): ${RKNN_API_LIB_PATH}")
else()
    set(RKNN_API_LIB_PATH "rknnrt")
    message(WARNING "RKNN lib not found, using system default: rknnrt")
endif()
message(STATUS "RKNN lib: ${RKNN_API_LIB_PATH}")


# 寻找OpenCV库，使用自定义的OpenCV_DIR
set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RGA_DIR ${3RDPARTY_PATH}/rga/${DEVICE_NAME})

# RGA lib 路径 - 优先使用系统库
if(EXISTS "/usr/lib/aarch64-linux-gnu/librga.so")
    set(RGA_LIB "/usr/lib/aarch64-linux-gnu/librga.so")
elseif(EXISTS ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)
    set(RGA_LIB ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)
else()
    set(RGA_LIB "rga")
endif()
message(STATUS "RGA lib: ${RGA_LIB}")

# set(OpenCV_DIR ${3RDPARTY_PATH}/opencv/opencv-linux-${LIB_ARCH}/share/OpenCV)
find_package(OpenCV REQUIRED)
# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")


# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RGA_DIR}/include
    /usr/include/rga
    ${CMAKE_CURRENT_LIST_DIR}/mpp_api/include
    ${CMAKE_CURRENT_LIST_DIR}/mk_api/include
    /usr/local/include
    /usr/include/rockchip
    ${CMAKE_CURRENT_SOURCE_DIR}/rkmedia/utils
    /usr/include/rknn
)

# 构建预处理和后处理库
add_library(nn_process SHARED
            process/preprocess.cpp
            process/postprocess.cpp
)
# 链接库
target_link_libraries(nn_process
    ${OpenCV_LIBS}
    ${RGA_LIB}
)

# 构建自定义封装API库
add_library(rknn_engine SHARED engine/rknn_engine.cpp) 
# 链接库
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)
# yolov8_lib
add_library(yolov8_lib SHARED task/yolov8_custom.cpp)
# 链接库
target_link_libraries(yolov8_lib
    rknn_engine
    nn_process
)

# draw_lib
add_library(draw_lib SHARED draw/cv_draw.cpp)
# 链接库
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)

# =====================================================
# 注意：utils 的功能已通过直接编译 mpp_decoder.cpp 和 
# mpp_encoder.cpp 实现，不需要单独的 libutils.so
# =====================================================

# 测试rknn_usb_to_rtsp
add_executable(rknn_usb_to_rtsp 
    rknn_usb_to_rtsp.cpp
    rkmedia/utils/mpp_decoder.cpp
    rkmedia/utils/mpp_encoder.cpp
)

# 链接库（移除 utils）
target_link_libraries(rknn_usb_to_rtsp 
    draw_lib
    yolov8_lib
    rockchip_mpp 
    mk_api 
    ${OpenCV_LIBS}
    ${RGA_LIB}
    pthread
)


# 测试自rknn_usb2_to_rtsp
add_executable(rknn_usb2_to_rtsp
    rknn_usb2_to_rtsp.cpp
    task/yolov8_thread_pool.cpp
    rkmedia/utils/mpp_decoder.cpp
    rkmedia/utils/mpp_encoder.cpp
)


# 链接库（移除 utils）
target_link_libraries(rknn_usb2_to_rtsp 
    draw_lib
    yolov8_lib
    rockchip_mpp 
    mk_api 
    ${OpenCV_LIBS}
    ${RGA_LIB}
    pthread
)

# 测试usb2_to_rtsp
add_executable(usb2_to_rtsp 
    usb2_to_rtsp.cpp
    rkmedia/utils/mpp_decoder.cpp
    rkmedia/utils/mpp_encoder.cpp
)

# 链接库（移除 utils）
target_link_libraries(usb2_to_rtsp
    rockchip_mpp 
    mk_api 
    ${OpenCV_LIBS}
    ${RGA_LIB}
    pthread
)

# 测试usb_to_rtsp
add_executable(usb_to_rtsp 
    usb_to_rtsp.cpp
    rkmedia/utils/mpp_decoder.cpp
    rkmedia/utils/mpp_encoder.cpp
)

# 链接库（移除 utils）
target_link_libraries(usb_to_rtsp
    rockchip_mpp 
    mk_api 
    ${OpenCV_LIBS}
    ${RGA_LIB}
    pthread
)

# 测试usb2_to_rknn
add_executable(usb2_to_rknn
    usb2_to_rknn.cpp
    task/yolov8_thread_pool.cpp
    rkmedia/utils/mpp_decoder.cpp
    rkmedia/utils/mpp_encoder.cpp
)

# 链接库（移除 utils）
target_link_libraries(usb2_to_rknn
    draw_lib
    yolov8_lib
    rockchip_mpp 
    mk_api 
    ${OpenCV_LIBS}
    ${RGA_LIB}
    pthread
)

#测试usb3_to_rknn
add_executable(usb3_to_rknn
    usb3_to_rknn.cpp
    task/yolov8_thread_pool.cpp
    rkmedia/utils/mpp_decoder.cpp
    rkmedia/utils/mpp_encoder.cpp
)

# 链接库（移除 utils）
target_link_libraries(usb3_to_rknn
    draw_lib
    yolov8_lib
    rockchip_mpp 
    mk_api 
    ${OpenCV_LIBS}
    ${RGA_LIB}
    pthread
)

# 测试usb_to_rknn
add_executable(usb_to_rknn 
    usb_to_rknn.cpp
    task/yolov8_thread_pool.cpp
    rkmedia/utils/mpp_decoder.cpp
    rkmedia/utils/mpp_encoder.cpp
)

# 链接库（移除 utils）
target_link_libraries(usb_to_rknn
    draw_lib
    yolov8_lib
    rockchip_mpp 
    mk_api 
    ${OpenCV_LIBS}
    ${RGA_LIB}
    pthread
)

