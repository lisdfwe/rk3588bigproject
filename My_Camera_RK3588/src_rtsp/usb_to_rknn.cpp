
#include <opencv2/opencv.hpp>

#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

#include "task/yolov8_thread_pool.h"

static int g_frame_start_id = 0; // 读取视频帧的索引
static int g_frame_end_id = 0;   // 模型处理完的索引

// 创建线程池
static Yolov8ThreadPool *g_pool = nullptr;
bool end = false;

// // 畸变矫正函数，输入为图像、相机矩阵和畸变系数
// cv::Mat undistortImage(const cv::Mat& img, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
//     cv::Mat newCameraMatrix, undistortedImg;
//     cv::Size imageSize = img.size();
//     newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize);
//     undistort(img, undistortedImg, cameraMatrix, distCoeffs, newCameraMatrix);
//     return undistortedImg;
// }

// // 透视变换函数，输入为图像和四点坐标
// cv::Mat perspectiveTransformation(const cv::Mat& img, const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {
//     cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
//     cv::Mat transformedImg;
//     warpPerspective(img, transformedImg, M, img.size());
//     return transformedImg;
// }

// // 拼接逻辑（包含透视变换）
// cv::Mat smartStitch(const cv::Mat& imgLeft, const cv::Mat& imgRight, int &blendWidth) {
//     int height = imgLeft.rows;
//     int widthLeft = imgLeft.cols;
//     int widthRight = imgRight.cols;

//     cv::Mat result = cv::Mat::zeros(height, widthLeft + widthRight - blendWidth, imgLeft.type());

//     imgLeft(cv::Rect(0, 0, widthLeft - blendWidth, height)).copyTo(result(cv::Rect(0, 0, widthLeft - blendWidth, height)));
//     imgRight(cv::Rect(blendWidth, 0, widthRight - blendWidth, height)).copyTo(result(cv::Rect(widthLeft, 0, widthRight - blendWidth, height)));

//     for (int i = 0; i < blendWidth; ++i) {
//         float alpha = (blendWidth - i) / float(blendWidth);
//         addWeighted(imgLeft.col(widthLeft - blendWidth + i), alpha, imgRight.col(i), 1 - alpha, 0, result.col(widthLeft - blendWidth + i));
//     }

//     return result;
// }

void get_results()
{

    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true)
    {

        // 结果
        cv::Mat img;
        auto ret = g_pool->getTargetImgResult(img, g_frame_end_id++);
        // 如果读取完毕，且模型处理完毕，结束
        if (end && ret != NN_SUCCESS)
        {
            g_pool->stopAll();
            break;
        }

        //显示结果
        cv::imshow("result", img);
        
        // 算法2：计算超过 1s 一共处理了多少张图片
        frame_count++;
        // all end
        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        // 每隔1秒打印一次
        if (elapsed_all_2 > 1000)
        {
            NN_LOG_INFO("Method2 Time:%fms, FPS:%f, Frame Count:%d", elapsed_all_2, frame_count / (elapsed_all_2 / 1000.0f), frame_count);
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    // 结束所有线程
    g_pool->stopAll();
    NN_LOG_INFO("Get results end.");
}
// 读取视频帧，提交任务
void read_stream(const char *video_path1)
{
    // 读取视频
    cv::VideoCapture cap1(video_path1);
    if (!cap1.isOpened())
    {
        printf("Failed to open video file: %s", video_path1);
        //return -1;
    }
    // 使用前需要使用v4l2-ctl --device=/dev/video0 --list-formats-ext检查一下设备支持范围
    cap1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // set width
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    // set height
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    // set fps
    cap1.set(cv::CAP_PROP_FPS,30);

    // 畸变矫正参数
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1000, 0, 640, 0, 1000, 360, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << 0.1, -0.25, 0, 0);
    int d = 200; // 渐入渐出融合宽度
    std::vector<cv::Point2f> srcPoints = { cv::Point2f(100, 100), cv::Point2f(500, 100), cv::Point2f(100, 400), cv::Point2f(500, 400) };
    std::vector<cv::Point2f> dstPoints = { cv::Point2f(0, 0), cv::Point2f(640, 0), cv::Point2f(0, 480), cv::Point2f(640, 480) };

    // 画面
    cv::Mat img;

    while (true)
    {
        // 读取视频帧
        cap1 >> img;
        if (img.empty())
        {
            printf("Video end.");
            end = true;
            break;
        }
        // 按 'q' 键退出
        if (cv::waitKey(30) >= 0) {
            printf("Close...");
            end = true;
            break;
        }
        flip(img, img, -1); // 旋转图像180度
        // flip(img2, img2, -1); // 旋转图像180度

        // img1 = undistortImage(img1, cameraMatrix, distCoeffs);// 畸变矫正
        // img2 = undistortImage(img2, cameraMatrix, distCoeffs);

        // img1 = perspectiveTransformation(img1, srcPoints, dstPoints);// 透视变换
        // img2 = perspectiveTransformation(img2, srcPoints, dstPoints);

        // img = smartStitch(img1, img2, d);

        // 提交任务，这里使用clone，因为不这样数据在内存中可能不连续，导致绘制错误
        g_pool->submitTask(img.clone(), g_frame_start_id++);
    }
    // 释放资源
    cap1.release();
    // 释放摄像头资源
    cv::destroyAllWindows();
}

int main(int argc, char **argv)
{
    // 检查参数
    if (argc != 4)
    {
        printf("Usage: %s<model_path> <video_path1> <pool_num>\n", argv[0]);
        return -1;
    }
    
    std::string model_path = (char *) argv[1]; // 模型路径
    char *stream_url1 = argv[2];   // 视频流地址 /dev/video0
    const int num_threads = (argc > 4) ? atoi(argv[3]) : 12;// 参数：线程池数量

    // 线程1：读取视频帧，提交任务
    // 线程池：模型运行
    // 线程2：拿到结果，绘制结果

    // 创建线程池
    g_pool = new Yolov8ThreadPool();
    g_pool->setUp(model_path, num_threads);

    // 读取视频
    std::thread read_stream_thread(read_stream,  stream_url1);
    // 启动结果线程
    std::thread result_thread(get_results);

    // 等待线程结束
    read_stream_thread.join();
    result_thread.join();

    

    return 0;
}