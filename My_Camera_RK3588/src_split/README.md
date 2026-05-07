# 多目摄像头拼接系统

这是一个专门用于多摄像头图像拼接的系统，支持实时全景画面生成，针对香橙派5Pro的RK3588平台优化。

## 功能特色

### 核心功能
- **三摄像头实时拼接**：支持同时处理3个USB摄像头
- **自动畸变校正**：内置摄像头标定和畸变矫正算法
- **透视变换**：支持多种投影模式的图像拼接
- **图像融合**：平滑的图像边界融合，消除拼接缝隙
- **HDMI实时输出**：拼接结果实时显示在HDMI屏幕
- **性能优化**：针对RK3588的GPU和多核CPU优化

### 技术特点
- 使用OpenCV stitching模块
- 支持特征点检测和匹配
- 实现图像配准和变换
- 多线程并行处理
- 内存优化管理

## 项目结构

```
hdmi_test/
├── src/
│   ├── three_hdmi_video.cpp         # 三摄像头拼接主程序
│   ├── hdmi_video.cpp               # 单摄像头HDMI输出
│   └── new2.cpp                     # 图像处理测试程序
├── build/                           # 编译输出目录
│   ├── three_hdmi_video             # 三摄像头拼接可执行文件
│   ├── hdmi_video                   # 单摄像头可执行文件
│   └── new2                         # 测试程序
├── CMakeLists.txt                   # CMake构建配置
└── test.jpg                         # 测试图片
```

## 系统要求

### 硬件要求
- **香橙派5Pro**（RK3588芯片）
- **3个USB摄像头**（推荐相同型号）
- **HDMI显示器**（用于显示拼接结果）
- **充足的USB供电**（建议使用带电源的USB Hub）

### 软件要求
- Ubuntu 20.04/22.04 ARM64
- OpenCV 4.x
- DRM显示支持
- V4L2摄像头驱动

## 编译安装

### 1. 环境准备

```bash
# 安装编译依赖
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libopencv-contrib-dev \
    libdrm-dev \
    libv4l-dev \
    v4l-utils

# 安装其他依赖
sudo apt install -y \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgbm-dev
```

### 2. 编译项目

```bash
cd hdmi_test

# 创建编译目录
mkdir -p build && cd build

# 配置CMake
cmake ..

# 编译（使用所有CPU核心）
make -j$(nproc)

# 验证编译结果
ls -la
```

## 使用指南

### 1. 摄像头配置

#### 连接摄像头
```bash
# 检查摄像头设备
ls /dev/video*
# 应该看到：/dev/video0, /dev/video2, /dev/video4 等

# 检查摄像头详细信息
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video2 --all
v4l2-ctl --device=/dev/video4 --all
```

#### 设置摄像头参数
```bash
# 设置统一的分辨率和帧率
for device in /dev/video0 /dev/video2 /dev/video4; do
    if [ -e "$device" ]; then
        echo "配置摄像头: $device"
        v4l2-ctl --device=$device --set-fmt-video=width=1280,height=720,pixelformat=MJPG
        v4l2-ctl --device=$device --set-parm=30
    fi
done
```

#### 测试摄像头
```bash
# 测试单个摄像头
ffplay /dev/video0
ffplay /dev/video2
ffplay /dev/video4
```

### 2. 运行拼接程序

#### 基本运行
```bash
cd build

# 启动三摄像头拼接（需要连接HDMI显示器）
./three_hdmi_video

# 程序会自动：
# 1. 检测并打开三个摄像头设备
# 2. 进行摄像头标定和畸变校正
# 3. 实时拼接生成全景画面
# 4. 通过HDMI输出显示结果
```

#### 单摄像头测试
```bash
# 测试单个摄像头的HDMI输出
./hdmi_video

# 图像处理测试
./new2
```

### 3. 操作说明

运行程序后：
- **ESC键**：退出程序
- **空格键**：暂停/继续拼接
- **S键**：保存当前拼接画面
- **R键**：重新初始化拼接器
- **C键**：重新标定摄像头

## 配置选项

### 摄像头布局配置

在`three_hdmi_video.cpp`中可以配置摄像头的物理布局：

```cpp
// 摄像头设备ID配置
const int CAMERA_LEFT = 0;    // 左摄像头 /dev/video0
const int CAMERA_CENTER = 2;  // 中摄像头 /dev/video2  
const int CAMERA_RIGHT = 4;   // 右摄像头 /dev/video4

// 摄像头分辨率配置
const int FRAME_WIDTH = 1280;
const int FRAME_HEIGHT = 720;
const int FRAME_FPS = 30;
```

### 拼接参数调优

```cpp
// 特征检测参数
cv::Ptr<cv::Feature2D> finder = cv::SIFT::create(
    0,      // nfeatures: 特征点数量（0为自动）
    3,      // nOctaveLayers: 层数
    0.04,   // contrastThreshold: 对比度阈值
    10,     // edgeThreshold: 边缘阈值
    1.6     // sigma: 高斯核标准差
);

// 匹配参数
cv::Ptr<cv::DescriptorMatcher> matcher = 
    cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

// 拼接器配置
cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
stitcher->setRegistrationResol(0.6);    // 配准分辨率
stitcher->setSeamEstimationResol(0.1);  // 接缝估计分辨率
stitcher->setCompositingResol(1.0);     // 合成分辨率
stitcher->setPanoConfidenceThresh(1.0); // 全景置信度阈值
```

### 相机内参配置

如果需要手动设置相机内参：

```cpp
// 相机内参矩阵 (fx, fy, cx, cy)
cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << 
    800.0, 0.0, 320.0,      // fx, 0, cx
    0.0, 800.0, 240.0,      // 0, fy, cy
    0.0, 0.0, 1.0           // 0, 0, 1
);

// 畸变系数 (k1, k2, p1, p2, k3)
cv::Mat dist_coeffs = (cv::Mat_<double>(1,5) << 
    -0.2, 0.1, 0.0, 0.0, 0.0
);
```

## 性能优化

### 1. 硬件加速

```cpp
// 使用GPU加速（如果支持）
stitcher->setWarper(cv::makePtr<cv::SphericalWarperGpu>());

// 使用OpenCL加速
cv::ocl::setUseOpenCL(true);
```

### 2. 多线程优化

```cpp
// 设置OpenCV线程数
cv::setNumThreads(cv::getNumberOfCPUs());

// 并行处理摄像头读取
std::vector<std::thread> capture_threads;
for (int i = 0; i < num_cameras; ++i) {
    capture_threads.emplace_back(capture_worker, i);
}
```

### 3. 内存优化

```cpp
// 预分配图像缓冲区
std::vector<cv::Mat> image_buffer(buffer_size);
for (auto& img : image_buffer) {
    img.create(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
}

// 使用对象池避免频繁分配
class ImagePool {
    std::queue<cv::Mat> available;
    std::mutex mutex;
public:
    cv::Mat acquire() { /* 获取图像 */ }
    void release(cv::Mat img) { /* 归还图像 */ }
};
```

## 故障排除

### 1. 摄像头问题

```bash
# 摄像头无法打开
# 检查设备权限
ls -la /dev/video*
sudo chmod 666 /dev/video*

# 检查USB带宽
lsusb -t

# 重置USB摄像头
sudo rmmod uvcvideo
sudo modprobe uvcvideo
```

### 2. 拼接失败

```bash
# 特征点太少
# 解决方案：
# 1. 确保场景有足够的纹理特征
# 2. 调整摄像头角度，增加重叠区域
# 3. 改善光照条件
# 4. 降低特征检测阈值
```

### 3. 性能问题

```bash
# 帧率过低
# 检查CPU使用率
htop

# 检查内存使用
free -h

# 检查温度
cat /sys/class/thermal/thermal_zone*/temp

# 设置性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 4. 显示问题

```bash
# HDMI输出异常
# 检查显示设备
drm_info

# 检查DRM权限
ls -la /dev/dri/
sudo usermod -a -G video $USER
```

## 高级功能

### 1. 实时标定

程序支持运行时重新标定摄像头：

```cpp
// 标定功能
void calibrate_cameras() {
    // 收集标定图像
    // 计算内参和畸变系数
    // 更新拼接器参数
}
```

### 2. 自适应拼接

```cpp
// 根据场景动态调整拼接参数
void adaptive_stitching(const std::vector<cv::Mat>& images) {
    // 分析图像质量
    // 调整特征检测参数
    // 优化拼接策略
}
```

### 3. 质量评估

```cpp
// 拼接质量评估
double evaluate_stitching_quality(const cv::Mat& panorama) {
    // 计算接缝质量
    // 分析重叠区域
    // 返回质量分数
}
```

## 应用场景

### 1. 安防监控
- 360度全景监控
- 无盲区监控覆盖
- 实时录像和回放

### 2. 直播应用
- 全景直播
- 虚拟现实内容制作
- 活动现场直播

### 3. 工业检测
- 产品360度检测
- 质量控制
- 自动化生产线监控

### 4. 智能驾驶
- 全景行车记录
- 盲区监测
- 自动泊车辅助

## 技术原理

### 拼接流程

1. **图像采集**：同时从多个摄像头获取图像
2. **预处理**：畸变校正、亮度均衡
3. **特征检测**：SIFT/SURF特征点检测
4. **特征匹配**：计算图像间的对应关系
5. **几何变换**：计算单应性矩阵
6. **图像配准**：将图像投影到同一坐标系
7. **接缝检测**：寻找最佳拼接边界
8. **图像融合**：平滑融合重叠区域
9. **结果输出**：生成最终全景图像

### 数学模型

```cpp
// 单应性变换
cv::Mat H = cv::findHomography(src_points, dst_points, 
                               cv::RANSAC, 3.0);

// 透视变换
cv::Mat warped;
cv::warpPerspective(src_image, warped, H, dst_size);

// 图像融合
cv::Mat blended;
cv::detail::MultiBandBlender blender;
blender.blend(blended, corners, images, masks);
```

这个多目摄像头拼接系统提供了完整的全景视频处理解决方案，适用于各种需要全景视野的应用场景。
