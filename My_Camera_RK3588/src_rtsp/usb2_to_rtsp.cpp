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

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"

#include "rkmedia/utils/mpp_decoder.h"
#include "rkmedia/utils/mpp_encoder.h"

#include "mk_mediakit.h"

using namespace cv;
using namespace std;

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
Mat undistortImage(const Mat& img, const Mat& cameraMatrix, const Mat& distCoeffs) {
    Mat newCameraMatrix, undistortedImg;
    Size imageSize = img.size();
    newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize);
    undistort(img, undistortedImg, cameraMatrix, distCoeffs, newCameraMatrix);
    return undistortedImg;
}

// 透视变换函数，输入为图像和四点坐标
Mat perspectiveTransformation(const Mat& img, const vector<Point2f>& srcPoints, const vector<Point2f>& dstPoints) {
    Mat M = getPerspectiveTransform(srcPoints, dstPoints);
    Mat transformedImg;
    warpPerspective(img, transformedImg, M, img.size());
    return transformedImg;
}

// 拼接逻辑（包含透视变换）
Mat smartStitch(const Mat& imgLeft, const Mat& imgRight, int blendWidth) {
    int height = imgLeft.rows;
    int widthLeft = imgLeft.cols;
    int widthRight = imgRight.cols;

    Mat result = Mat::zeros(height, widthLeft + widthRight - blendWidth, imgLeft.type());

    imgLeft(Rect(0, 0, widthLeft - blendWidth, height)).copyTo(result(Rect(0, 0, widthLeft - blendWidth, height)));
    imgRight(Rect(blendWidth, 0, widthRight - blendWidth, height)).copyTo(result(Rect(widthLeft, 0, widthRight - blendWidth, height)));

    for (int i = 0; i < blendWidth; ++i) {
        float alpha = (blendWidth - i) / float(blendWidth);
        addWeighted(imgLeft.col(widthLeft - blendWidth + i), alpha, imgRight.col(i), 1 - alpha, 0, result.col(widthLeft - blendWidth + i));
    }

    return result;
}

// 处理视频文件的函数
int process_video_file(rknn_app_context_t *ctx, const char *video_path1, const char *video_path2)
{
    // 读取视频
    cv::VideoCapture cap1(video_path1);
    cv::VideoCapture cap2(video_path2);
    if (!cap1.isOpened() ||!cap2.isOpened())
    {
        printf("Failed to open video file: %s, %s", video_path1, video_path2);
        //return -1;
    }
    // 使用前需要使用v4l2-ctl --device=/dev/video0 --list-formats-ext检查一下设备支持范围
    cap1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // set width
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    // set height
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // set fps
    cap1.set(cv::CAP_PROP_FPS,30);
    cap2.set(cv::CAP_PROP_FPS,30);
    // 获取视频尺寸、帧率

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
    int ret = 0;

    // 畸变矫正参数
    cv::Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 640, 0, 1000, 360, 0, 0, 1);
    cv::Mat distCoeffs = (Mat_<double>(1, 4) << 0.1, -0.25, 0, 0);
    int d = 200; // 渐入渐出融合宽度
    vector<Point2f> srcPoints = { Point2f(100, 100), Point2f(500, 100), Point2f(100, 400), Point2f(500, 400) };
    vector<Point2f> dstPoints = { Point2f(0, 0), Point2f(640, 0), Point2f(0, 480), Point2f(640, 480) };

    // 画面
    cv::Mat img,img1,img2;

    while (true)
    {
        // 读取视频帧
        cap1 >> img1;
        cap2 >> img2;
        if (img1.empty() || img2.empty())
        {
            printf("Video end.");
            break;
        }



        img1 = undistortImage(img1, cameraMatrix, distCoeffs);// 畸变矫正
        img2 = undistortImage(img2, cameraMatrix, distCoeffs);

        img1 = perspectiveTransformation(img1, srcPoints, dstPoints);// 透视变换
        img2 = perspectiveTransformation(img2, srcPoints, dstPoints);

        img = smartStitch(img1, img2, d);// 拼接

        // 拉伸方式定义新的大小为1280x480
        // cv::Size newSize(1280, 480);
        // cv::resize(img, img, newSize);

        // 计算需要添加的黑边宽度
        //int targetWidth = 1280;
        //int imgWidth = img.cols; // 原图宽度
        //int borderWidth = (targetWidth - imgWidth) / 2; // 每边添加的宽度
        int borderWidth =100; // 每边添加的宽度

        // 使用copyMakeBorder添加黑边
        cv::copyMakeBorder(img, img, 0, 0, borderWidth, borderWidth, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  
        //hconcat(img1, img2, img); // 在水平方向拼接图像

        imwrite("test.jpg", img); // 保存图片
        // 获取图片的大小
        cv::Size size = img.size();
        std::cout << "Width: " << size.width << ", Height: " << size.height << std::endl;

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
        
        enc_data_size = ctx->encoder->Encode(mpp_frame, enc_data, enc_buf_size);

        ret = mk_media_input_h264(ctx->media, enc_data, enc_data_size, millis, millis);
        if (ret != 1)
        {
            printf("mk_media_input_frame failed\n");
        }
        if (enc_data != nullptr)
        {
            free(enc_data);
        }
    }
    // 释放资源
    cap1.release();
    cap2.release();
    release_track(&v_track);
    release_media(&ctx->media);

}

// 主函数
int main(int argc, char **argv)
{
    int status = 0;
    int ret;

    if (argc != 3)
    {
        printf("Usage: %s<video_path1> <video_path2>\n", argv[0]);
        return -1;
    }
    char *stream_url1 = argv[1];   // 视频流地址/dev/video0
    char *stream_url2 = argv[2];   // 视频流地址/dev/video2
    int video_type = 264;

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

    process_video_file(&app_ctx, stream_url1, stream_url2);

    printf("waiting finish\n");
    usleep(3 * 1000 * 1000);

    if (app_ctx.encoder != nullptr)
    {
        delete (app_ctx.encoder);
        app_ctx.encoder = nullptr;
    }

    return 0;
}