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

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 输出OpenCV信息
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# RGA lib 路径 - 优先使用系统库
if(EXISTS "/usr/lib/aarch64-linux-gnu/librga.so")
    set(RGA_LIB "/usr/lib/aarch64-linux-gnu/librga.so")
else()
    set(RGA_LIB "rga")
endif()
message(STATUS "RGA lib: ${RGA_LIB}")

# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/rga
)

add_executable(three_hdmi_video 
    three_hdmi_video.cpp
)

# 链接库        
target_link_libraries(three_hdmi_video 
        ${OpenCV_LIBS}
        ${RGA_LIB}
)


add_executable(two_opencv_video
    two_opencv_video.cpp
)

# 链接库
target_link_libraries(two_opencv_video 
        ${OpenCV_LIBS}
        ${RGA_LIB}
)