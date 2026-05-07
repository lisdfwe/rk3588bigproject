/*-------------------------------------------
                Includes
-------------------------------------------*/
// 包含所需的头文件
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "task/yolov8_thread_pool.h"

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"

#include "rkmedia/utils/mpp_decoder.h"
#include "rkmedia/utils/mpp_encoder.h"

#include "mk_mediakit.h"

static int g_frame_start_id = 0; // 读取视频帧的索引
static int g_frame_end_id = 0;   // 模型处理完的索引

// 创建线程池
static Yolov8ThreadPool *g_pool = nullptr; // 线程池
bool end = false; // 结束标志

// 定义应用程序上下文结构体
typedef struct
{
    MppEncoder *encoder;
    mk_media media;
    mk_pusher pusher;
    const char *push_url;

    int video_type=264;

    int push_rtsp_port;
    std::string push_path_first;
    std::string push_path_second;

} rknn_app_context_t;

// 释放媒体资源
void release_media(mk_media *ptr)
{
    if (ptr && *ptr)
    {
        mk_media_release(*ptr);
        *ptr = NULL;
    }
}

// 释放推流资源
void release_pusher(mk_pusher *ptr)
{
    if (ptr && *ptr)
    {
        mk_pusher_release(*ptr);
        *ptr = NULL;
    }
}

// 释放轨道资源
void release_track(mk_track *ptr)
{
    if (ptr && *ptr)
    {
        mk_track_unref(*ptr);
        *ptr = NULL;
    }
}

// 将数字填充为16的倍数
int padToMultipleOf16(int number) {  
    // 如果number已经是16的倍数，则直接返回  
    if (number % 16 == 0) {  
        return number;  
    }  
    // 否则，计算需要添加的额外量（即16 - (number % 16)）  
    // 这等价于找到比number大的最小16的倍数，并减去number  
    int extra = 16 - (number % 16);  
    // 返回扩充后的数  
    return number + extra;  
}

// 推流事件处理函数
void API_CALL on_mk_push_event_func(void *user_data, int err_code, const char *err_msg)
{
    rknn_app_context_t *ctx = (rknn_app_context_t *)user_data;
    if (err_code == 0)
    {
        // push success
        log_info("push %s success!", ctx->push_url);
        printf("push %s success!\n", ctx->push_url);
    }
    else
    {
        log_warn("push %s failed:%d %s", ctx->push_url, err_code, err_msg);
        printf("push %s failed:%d %s\n", ctx->push_url, err_code, err_msg);
        release_pusher(&(ctx->pusher));
    }
}

// 媒体源注册事件处理函数
void API_CALL on_mk_media_source_regist_func(void *user_data, mk_media_source sender, int regist)
{
    rknn_app_context_t *ctx = (rknn_app_context_t *)user_data;
    const char *schema = mk_media_source_get_schema(sender);
    if (strncmp(schema, ctx->push_url, strlen(schema)) == 0)
    {
        release_pusher(&(ctx->pusher));
        if (regist)
        {
            ctx->pusher = mk_pusher_create_src(sender);
            mk_pusher_set_on_result(ctx->pusher, on_mk_push_event_func, ctx);
            mk_pusher_set_on_shutdown(ctx->pusher, on_mk_push_event_func, ctx);
            log_info("push started!");
            printf("push started!\n");
        }
        else
        {
            log_info("push stoped!");
            printf("push stoped!\n");
        }
        printf("push_url:%s\n", ctx->push_url);
    }
    else
    {
        printf("unknown schema:%s\n", schema);
    }
}

// 关闭事件处理函数
void API_CALL on_mk_shutdown_func(void *user_data, int err_code, const char *err_msg, mk_track tracks[], int track_count)
{
    printf("play interrupted: %d %s", err_code, err_msg);
}

// 畸变矫正函数，输入为图像、相机矩阵和畸变系数
cv::Mat undistortImage(const cv::Mat& img, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    cv::Mat newCameraMatrix, undistortedImg;
    cv::Size imageSize = img.size();
    newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize);
    undistort(img, undistortedImg, cameraMatrix, distCoeffs, newCameraMatrix);
    return undistortedImg;
}

// 透视变换函数，输入为图像和四点坐标
cv::Mat perspectiveTransformation(const cv::Mat& img, const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {
    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
    cv::Mat transformedImg;
    warpPerspective(img, transformedImg, M, img.size());
    return transformedImg;
}

// 拼接逻辑（包含透视变换）
cv::Mat smartStitch(const cv::Mat& imgLeft, const cv::Mat& imgRight, int blendWidth) {
    int height = imgLeft.rows;
    int widthLeft = imgLeft.cols;
    int widthRight = imgRight.cols;

    cv::Mat result = cv::Mat::zeros(height, widthLeft + widthRight - blendWidth, imgLeft.type());

    imgLeft(cv::Rect(0, 0, widthLeft - blendWidth, height)).copyTo(result(cv::Rect(0, 0, widthLeft - blendWidth, height)));
    imgRight(cv::Rect(blendWidth, 0, widthRight - blendWidth, height)).copyTo(result(cv::Rect(widthLeft, 0, widthRight - blendWidth, height)));

    for (int i = 0; i < blendWidth; ++i) {
        float alpha = (blendWidth - i) / float(blendWidth);
        addWeighted(imgLeft.col(widthLeft - blendWidth + i), alpha, imgRight.col(i), 1 - alpha, 0, result.col(widthLeft - blendWidth + i));
    }

    return result;
}

// 读取视频帧，提交任务
void read_stream(const char *video_path1,const char *video_path2)
{
    // 使用V4L2后端打开摄像头，避免GStreamer格式协商问题
    cv::VideoCapture cap1, cap2;
    
    // 判断是摄像头设备还是视频文件
    if (strncmp(video_path1, "/dev/video", 10) == 0)
    {
        int device_id1 = atoi(video_path1 + 10);
        cap1.open(device_id1, cv::CAP_V4L2);
        if (!cap1.isOpened())
        {
            printf("Failed to open camera with V4L2: %s\n", video_path1);
            return;
        }
        cap1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap1.set(cv::CAP_PROP_FPS, 120);
    }
    else
    {
        cap1.open(video_path1);
        if (!cap1.isOpened())
        {
            printf("Failed to open video file: %s\n", video_path1);
            return;
        }
    }

    if (strncmp(video_path2, "/dev/video", 10) == 0)
    {
        int device_id2 = atoi(video_path2 + 10);
        cap2.open(device_id2, cv::CAP_V4L2);
        if (!cap2.isOpened())
        {
            printf("Failed to open camera with V4L2: %s\n", video_path2);
            cap1.release();
            return;
        }
        cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap2.set(cv::CAP_PROP_FPS, 120);
    }
    else
    {
        cap2.open(video_path2);
        if (!cap2.isOpened())
        {
            printf("Failed to open video file: %s\n", video_path2);
            cap1.release();
            return;
        }
    }

    // 畸变矫正参数
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1000, 0, 640, 0, 1000, 360, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << 0.1, -0.25, 0, 0);
    int d = 200; // 渐入渐出融合宽度
    std::vector<cv::Point2f> srcPoints = { cv::Point2f(100, 100), cv::Point2f(500, 100), cv::Point2f(100, 400), cv::Point2f(500, 400) };
    std::vector<cv::Point2f> dstPoints = { cv::Point2f(0, 0), cv::Point2f(640, 0), cv::Point2f(0, 480), cv::Point2f(640, 480) };

    // 画面
    cv::Mat img,img1, img2;

    while (true)
    {
        // 读取视频帧
        cap1 >> img1;
        cap2 >> img2;
        if (img1.empty() || img2.empty())
        {
            printf("Video end.\n");
            end = true;
            break;
        }

        img1 = undistortImage(img1, cameraMatrix, distCoeffs);// 畸变矫正
        img2 = undistortImage(img2, cameraMatrix, distCoeffs);

        img1 = perspectiveTransformation(img1, srcPoints, dstPoints);// 透视变换
        img2 = perspectiveTransformation(img2, srcPoints, dstPoints);

        img = smartStitch(img1, img2, d);

        // 提交任务，这里使用clone，因为不这样数据在内存中可能不连续，导致绘制错误
        g_pool->submitTask(img.clone(), g_frame_start_id++);
    }
    // 释放资源
    cap1.release();
    cap2.release();
}

void get_results(rknn_app_context_t *ctx  )
{
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    int cap_width = 1280;
    int cap_height = 480;
    int fps = 30;

    ctx->push_url = "rtsp://localhost/live/stream";

    ctx->media = mk_media_create("__defaultVhost__", ctx->push_path_first.c_str(), ctx->push_path_second.c_str(), 0, 0, 0);

    codec_args v_args = {0};
    mk_track v_track = mk_track_create(MKCodecH264, &v_args);
    mk_media_init_track(ctx->media, v_track);

    mk_media_init_complete(ctx->media);
    mk_media_set_on_regist(ctx->media, on_mk_media_source_regist_func, ctx);

    // 初始化编码器
    MppEncoder *mpp_encoder = new MppEncoder();
    MppEncoderParams enc_params;
    memset(&enc_params, 0, sizeof(MppEncoderParams));
    enc_params.width = cap_width;
    enc_params.height = cap_height;
    enc_params.fmt = MPP_FMT_YUV420SP;
    enc_params.type = MPP_VIDEO_CodingAVC;
    mpp_encoder->Init(enc_params, ctx);
    ctx->encoder = mpp_encoder;

    // mpp编码配置
    void *mpp_frame = NULL;
    int mpp_frame_fd = 0;
    void *mpp_frame_addr = NULL;
    int enc_data_size;
    
    int frame_index = 0;
    int ret1 = 0;

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

        // 计算需要添加的黑边宽度
        //int targetWidth = 1280;
        //int imgWidth = img.cols; // 原图宽度
        //int borderWidth = (targetWidth - imgWidth) / 2; // 每边添加的宽度
        int borderWidth =100; // 每边添加的宽度

        // 使用copyMakeBorder添加黑边
        cv::copyMakeBorder(img, img, 0, 0, borderWidth, borderWidth, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        // imwrite("test.jpg", img); // 保存图片
        // // 获取图片的大小
        // cv::Size size = img.size();
        // std::cout << "Width: " << size.width << ", Height: " << size.height << std::endl;

        frame_index++;
        // 结束计时
        auto end_time = std::chrono::high_resolution_clock::now();
        // 将当前时间点转换为毫秒级别的时间戳
        auto millis = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();

        // 获取解码后的帧
        mpp_frame = ctx->encoder->GetInputFrameBuffer();
        // 获取解码后的帧fd
        mpp_frame_fd = ctx->encoder->GetInputFrameBufferFd(mpp_frame);
        // 获取解码后的帧地址
        mpp_frame_addr = ctx->encoder->GetInputFrameBufferAddr(mpp_frame);

        rga_buffer_t src = wrapbuffer_fd(mpp_frame_fd, cap_width, cap_height, RK_FORMAT_YCbCr_420_SP,padToMultipleOf16(cap_width),padToMultipleOf16(cap_height));

        int enc_buf_size = ctx->encoder->GetFrameSize();

        char *enc_data = (char *)malloc(enc_buf_size);

        // 将图像数据包装成RGA缓冲区格式
        rga_buffer_t rgb_img = wrapbuffer_virtualaddr((void *)img.data, cap_width, cap_height, RK_FORMAT_BGR_888);
        // 将RGB图像复制到src中
        imcopy(rgb_img, src);

        if (frame_index == 1)
        {
            enc_data_size = ctx->encoder->GetHeader(enc_data, enc_buf_size);
        }
        // 内存初始化
        memset(enc_data, 0, enc_buf_size);
        
        // 对输入数据进行编码，并获取编码后的数据大小
        enc_data_size = ctx->encoder->Encode(mpp_frame, enc_data, enc_buf_size);

        // 将编码后的数据发送到媒体输入
        ret1 = mk_media_input_h264(ctx->media, enc_data, enc_data_size, millis, millis);
        if (ret1 != 1)
        {
            printf("mk_media_input_frame failed\n");
        }
        if (enc_data != nullptr)
        {
            free(enc_data);
        }
    }
    // 结束所有线程
    release_track(&v_track);
    release_media(&ctx->media);
    g_pool->stopAll();
    NN_LOG_INFO("Get results end.");
}

//参数列表的解释是：argc表示参数的个数，argv表示参数的字符数组，argv[0]表示程序名，argv[1]表示第一个参数，以此类推。
int main(int argc, char **argv)
{
    // 检查参数
    if (argc != 5)
    {
        printf("Usage: %s <model_path> <video_path1> <video_path2> <pool_num>\n", argv[0]);
        return -1;
    }
    
    std::string model_path = argv[1];  // 模型路径
    char *stream_url1 = argv[2];       // 视频流地址 /dev/video0
    char *stream_url2 = argv[3];       // 视频流地址 /dev/video2
    const int num_threads = atoi(argv[4]); // 线程池数量（修复：argc==5时argv[4]存在）
    int video_type = 264; // 视频编码格式，这里是h264

    printf("Starting with %d threads\n", num_threads);

    // 初始化流媒体
    mk_config config;
    memset(&config, 0, sizeof(mk_config));
    config.log_mask = LOG_CONSOLE;
    config.thread_num = 4;
    mk_env_init(&config);
    mk_rtsp_server_start(3554, 0);

    rknn_app_context_t app_ctx;                      // 创建上下文
    memset(&app_ctx, 0, sizeof(rknn_app_context_t)); // 初始化上下文
    app_ctx.video_type = video_type;
    app_ctx.push_path_first = "airport-live";
    app_ctx.push_path_second = "single";

    // 线程1：读取视频帧，提交任务
    // 线程池：模型运行
    // 线程2：拿到结果，绘制结果
    
    // 创建线程池
    g_pool = new Yolov8ThreadPool();
    g_pool->setUp(model_path, num_threads);

    // 读取视频
    std::thread read_stream_thread(read_stream,  stream_url1, stream_url2);
    // 启动结果线程
    std::thread result_thread(get_results, &app_ctx);

   // 等待线程结束
    read_stream_thread.join();
    result_thread.join();

    printf("waiting finish\n");
    usleep(3 * 1000 * 1000);

    if (app_ctx.encoder != nullptr)
    {
        delete (app_ctx.encoder);
        app_ctx.encoder = nullptr;
    }

    return 0;
}