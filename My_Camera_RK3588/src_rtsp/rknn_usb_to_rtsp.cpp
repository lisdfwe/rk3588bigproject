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
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"

#include "rkmedia/utils/mpp_decoder.h"
#include "rkmedia/utils/mpp_encoder.h"

#include "mk_mediakit.h"

// 帧数据结构
struct FrameData {
    cv::Mat img;
    int frame_id;
    std::vector<Detection> objects;
    bool processed;
};

// 线程安全的帧队列
class FrameQueue {
private:
    std::queue<FrameData> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    int max_size_;
    bool stopped_;

public:
    FrameQueue(int max_size = 5) : max_size_(max_size), stopped_(false) {}

    void push(const FrameData& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return queue_.size() < max_size_ || stopped_; });
        if (!stopped_) {
            queue_.push(frame);
            cv_.notify_all();
        }
    }

    bool pop(FrameData& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return false;
        frame = queue_.front();
        queue_.pop();
        cv_.notify_all();
        return true;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

static FrameQueue g_input_queue(10);   // 待推理队列
static FrameQueue g_output_queue(10);  // 已推理队列
static bool g_running = true;

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

void release_media(mk_media *ptr)
{
    if (ptr && *ptr) { mk_media_release(*ptr); *ptr = NULL; }
}

void release_pusher(mk_pusher *ptr)
{
    if (ptr && *ptr) { mk_pusher_release(*ptr); *ptr = NULL; }
}

void release_track(mk_track *ptr)
{
    if (ptr && *ptr) { mk_track_unref(*ptr); *ptr = NULL; }
}

int padToMultipleOf16(int number) {  
    if (number % 16 == 0) return number;  
    return number + (16 - (number % 16));  
}

// 推流事件处理函数
void API_CALL on_mk_push_event_func(void *user_data, int err_code, const char *err_msg)
{
    rknn_app_context_t *ctx = (rknn_app_context_t *)user_data;
    if (err_code == 0) {
        log_info("push %s success!", ctx->push_url);
        printf("push %s success!\n", ctx->push_url);
    } else {
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
    if (strncmp(schema, ctx->push_url, strlen(schema)) == 0) {
        release_pusher(&(ctx->pusher));
        if (regist) {
            ctx->pusher = mk_pusher_create_src(sender);
            mk_pusher_set_on_result(ctx->pusher, on_mk_push_event_func, ctx);
            mk_pusher_set_on_shutdown(ctx->pusher, on_mk_push_event_func, ctx);
            log_info("push started!");
            printf("push started!\n");
        } else {
            log_info("push stoped!");
            printf("push stoped!\n");
        }
        printf("push_url:%s\n", ctx->push_url);
    } else {
        printf("unknown schema:%s\n", schema);
    }
}

// 关闭事件处理函数
void API_CALL on_mk_shutdown_func(void *user_data, int err_code, const char *err_msg, mk_track tracks[], int track_count)
{
    printf("play interrupted: %d %s", err_code, err_msg);
}

// 推理线程函数（多个线程并发推理）
void inference_thread(const std::string& model_path, int thread_id)
{
    printf("Inference thread %d started\n", thread_id);
    
    // ========================================
    // 关键修改：每个线程创建独立的 YOLOv8 实例和 rknn_context
    // 这样 RKNN 会自动将不同的 context 分配到不同的 NPU 核心
    // ========================================
    Yolov8Custom yolo;
    auto ret = yolo.LoadModel(model_path.c_str());
    if (ret != NN_SUCCESS) {
        printf("Thread %d: Failed to load model, ret=%d\n", thread_id, ret);
        return;
    }
    printf("Thread %d: Model loaded, independent rknn_context created\n", thread_id);

    int frame_count = 0;
    auto thread_start = std::chrono::high_resolution_clock::now();

    while (g_running) {
        FrameData frame;
        if (!g_input_queue.pop(frame)) break;

        // 推理
        yolo.Run(frame.img, frame.objects);
        
        // 设置颜色
        for (auto &object : frame.objects) {
            if (object.class_id >= 0 && object.class_id <= 20) {
                object.color = cv::Scalar(0, 0, 255);
            } else if (object.class_id >= 21 && object.class_id <= 40) {
                object.color = cv::Scalar(0, 255, 0);
            } else if (object.class_id >= 41 && object.class_id <= 60) {
                object.color = cv::Scalar(0, 255, 255);
            } else {
                object.color = cv::Scalar(255, 0, 0);
            }
        }
        
        // 绘制检测框
        DrawDetections(frame.img, frame.objects);
        frame.processed = true;

        // 放入输出队列
        g_output_queue.push(frame);

        frame_count++;
    }

    auto thread_end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(thread_end - thread_start).count();
    printf("Inference thread %d stopped, processed %d frames in %ld ms (%.2f FPS)\n", 
           thread_id, frame_count, elapsed, frame_count * 1000.0f / elapsed);
}

// 读取帧线程
void read_frames_thread(const char *video_path)
{
    cv::VideoCapture cap;
    
    if (strncmp(video_path, "/dev/video", 10) == 0) {
        int device_id = atoi(video_path + 10);
        cap.open(device_id, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            printf("Failed to open camera with V4L2: %s\n", video_path);
            g_running = false;
            return;
        }
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_FPS, 30);
    } else {
        cap.open(video_path);
        if (!cap.isOpened()) {
            printf("Failed to open video file: %s\n", video_path);
            g_running = false;
            return;
        }
    }

    int cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    printf("Video opened: %dx%d\n", cap_width, cap_height);

    int frame_id = 0;
    while (g_running) {
        FrameData frame;
        cap >> frame.img;
        if (frame.img.empty()) {
            printf("Video end.\n");
            g_running = false;
            break;
        }

        frame.frame_id = frame_id++;
        frame.processed = false;
        g_input_queue.push(frame);
    }

    cap.release();
    g_input_queue.stop();
    printf("Read frames thread stopped\n");
}

// 编码推流线程
int encode_and_stream(rknn_app_context_t *ctx, int cap_width, int cap_height)
{
    int fps = 30;
    ctx->push_url = "rtsp://localhost/live/stream";

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

    int enc_buf_size = ctx->encoder->GetFrameSize();
    char *enc_data = (char *)malloc(enc_buf_size);

    // 读取第一帧并编码以获取SPS/PPS
    FrameData first_frame;
    if (!g_output_queue.pop(first_frame)) {
        free(enc_data);
        return -1;
    }

    void *mpp_frame = ctx->encoder->GetInputFrameBuffer();
    int mpp_frame_fd = ctx->encoder->GetInputFrameBufferFd(mpp_frame);
    
    rga_buffer_t src = wrapbuffer_fd(mpp_frame_fd, cap_width, cap_height, RK_FORMAT_YCbCr_420_SP, 
                                     padToMultipleOf16(cap_width), padToMultipleOf16(cap_height));
    rga_buffer_t rgb_img = wrapbuffer_virtualaddr((void *)first_frame.img.data, cap_width, cap_height, RK_FORMAT_BGR_888);
    imcopy(rgb_img, src);
    
    memset(enc_data, 0, enc_buf_size);
    ctx->encoder->Encode(mpp_frame, enc_data, enc_buf_size);

    // 获取头信息
    memset(enc_data, 0, enc_buf_size);
    int header_size = ctx->encoder->GetHeader(enc_data, enc_buf_size);
    printf("H264 header size: %d\n", header_size);

    // 创建媒体源
    ctx->media = mk_media_create("__defaultVhost__", ctx->push_path_first.c_str(), ctx->push_path_second.c_str(), 0, 0, 0);
    codec_args v_args = {0};
    v_args.video.width = cap_width;
    v_args.video.height = cap_height;
    v_args.video.fps = fps;
    mk_track v_track = mk_track_create(MKCodecH264, &v_args);
    mk_media_init_track(ctx->media, v_track);
    mk_media_init_complete(ctx->media);
    mk_media_set_on_regist(ctx->media, on_mk_media_source_regist_func, ctx);

    if (header_size > 0) {
        auto millis = std::chrono::time_point_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now()).time_since_epoch().count();
        mk_media_input_h264(ctx->media, enc_data, header_size, millis, millis);
    }

    usleep(100 * 1000);

    // FPS统计
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    // 处理剩余帧
    while (g_running) {
        FrameData frame;
        if (!g_output_queue.pop(frame)) break;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto millis = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();

        mpp_frame = ctx->encoder->GetInputFrameBuffer();
        mpp_frame_fd = ctx->encoder->GetInputFrameBufferFd(mpp_frame);

        src = wrapbuffer_fd(mpp_frame_fd, cap_width, cap_height, RK_FORMAT_YCbCr_420_SP,
                           padToMultipleOf16(cap_width), padToMultipleOf16(cap_height));
        rgb_img = wrapbuffer_virtualaddr((void *)frame.img.data, cap_width, cap_height, RK_FORMAT_BGR_888);
        imcopy(rgb_img, src);

        memset(enc_data, 0, enc_buf_size);
        int enc_data_size = ctx->encoder->Encode(mpp_frame, enc_data, enc_buf_size);

        if (enc_data_size > 0) {
            mk_media_input_h264(ctx->media, enc_data, enc_data_size, millis, millis);
        }

        // FPS统计
        frame_count++;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_all).count();
        if (elapsed > 5000) {
            printf("Processed %d frames in %ld ms, FPS: %.2f\n", frame_count, elapsed, 
                   frame_count * 1000.0f / elapsed);
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }

    free(enc_data);
    release_track(&v_track);
    release_media(&ctx->media);
    return 0;
}

//参数列表的解释是：argc表示参数的个数，argv表示参数的字符数组
int main(int argc, char **argv)
{
    // 检查参数：增加线程数参数
    if (argc != 3 && argc != 4)
    {
        printf("Usage: %s <model_path> <video_path> [num_threads]\n", argv[0]);
        printf("  num_threads: number of inference threads (default: 3)\n");
        return -1;
    }
    
    const char *model_path = argv[1];
    char *stream_url = argv[2];
    int num_threads = (argc == 4) ? atoi(argv[3]) : 3;  // 默认3个推理线程
    int video_type = 264;

    printf("Starting with %d inference threads\n", num_threads);

    // 初始化流媒体
    mk_config config;
    memset(&config, 0, sizeof(mk_config));
    config.log_mask = LOG_CONSOLE;
    config.thread_num = 4;
    mk_env_init(&config);
    mk_rtsp_server_start(3554, 0);

    rknn_app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(rknn_app_context_t));
    app_ctx.video_type = video_type;
    app_ctx.push_path_first = "airport-live";
    app_ctx.push_path_second = "single";

    // 启动读帧线程
    std::thread reader_thread(read_frames_thread, stream_url);

    // 启动多个推理线程
    std::vector<std::thread> inference_threads;
    for (int i = 0; i < num_threads; i++) {
        inference_threads.emplace_back(inference_thread, std::string(model_path), i);
    }

    // 等待第一帧确定分辨率
    usleep(100 * 1000);

    // 启动编码推流线程
    std::thread encoder_thread(encode_and_stream, &app_ctx, 1920, 1080);

    // 等待所有线程结束
    reader_thread.join();
    for (auto& t : inference_threads) {
        t.join();
    }
    
    g_output_queue.stop();
    encoder_thread.join();

    printf("waiting finish\n");
    usleep(3 * 1000 * 1000);

    if (app_ctx.encoder != nullptr) {
        delete (app_ctx.encoder);
        app_ctx.encoder = nullptr;
    }

    return 0;
}