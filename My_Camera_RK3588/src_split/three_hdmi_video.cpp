#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

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

int main() {
    VideoCapture cap1(0);
    VideoCapture cap2(2);
    VideoCapture cap3(4);

    // 使用前需要使用v4l2-ctl --device=/dev/video0 --list-formats-ext检查一下设备支持范围
    cap1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap3.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // set width
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap3.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    // set height
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap3.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // set fps
    cap1.set(cv::CAP_PROP_FPS,30);
    cap2.set(cv::CAP_PROP_FPS,30);
    cap3.set(cv::CAP_PROP_FPS,30);

    Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 640, 0, 1000, 360, 0, 0, 1);//相机矩阵,参数分别摄像头的焦距、中心点坐标、图像中心点坐标
    Mat distCoeffs = (Mat_<double>(1, 4) << 0.1, -0.25, 0, 0);//畸变系数

    int rate = 60; 
    int delay = 1000 / rate;//延迟时间
    bool stop = false;//退出标志位
    int d = 200; // 渐入渐出融合宽度

    // 获取并打印摄像头的宽度和高度
    double width = cap1.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap1.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Width: " << width << std::endl;
    std::cout << "Height: " << height << std::endl;

    // 获取并打印摄像头的帧率
    double fps = cap1.get(cv::CAP_PROP_FPS);
    std::cout << "FPS: " << fps << std::endl;

    vector<Point2f> srcPoints = { Point2f(100, 100), Point2f(500, 100), Point2f(100, 400), Point2f(500, 400) };// 四点坐标
    vector<Point2f> dstPoints = { Point2f(0, 0), Point2f(640, 0), Point2f(0, 480), Point2f(640, 480) };// 四点坐标

    //namedWindow("cam1", WINDOW_AUTOSIZE);//窗口大小自适应
    //namedWindow("cam2", WINDOW_AUTOSIZE);//窗口大小自适应
    //namedWindow("cam3", WINDOW_AUTOSIZE);//窗口大小自适应
    //namedWindow("stitch", WINDOW_AUTOSIZE);//窗口大小自适应


    if (!cap1.isOpened() || !cap2.isOpened() || !cap3.isOpened()) {
        cout << "*** ***" << endl;
        cout << "警告：请检查所有摄像头是否安装好!" << endl;
        cout << "程序结束！" << endl;
        return -1;
    }
    else {
        cout << "*** ***" << endl;
        cout << "摄像头已启动！" << endl;
    }
    
    
    Mat img1, img2, img3;

    auto start_all = std::chrono::high_resolution_clock::now();// 开始计时
    int frame_count = 0; // 记录处理的帧数

    while (!stop) {
        // 开始计时
        auto start_1 = std::chrono::high_resolution_clock::now();

        bool ret1 = cap1.read(img1);    //读取摄像头画面
        bool ret2 = cap2.read(img2);    //读取摄像头画面
        bool ret3 = cap3.read(img3);    //读取摄像头画面

        // 记录读取视频帧的时间：读取视频帧的时间
        auto end_1 = std::chrono::high_resolution_clock::now();
        // microseconds 微秒，milliseconds 毫秒，seconds 秒，1微妙=0.001毫秒 = 0.000001秒
        auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count() / 1000.0;

        // 开始计时
        auto start_2 = std::chrono::high_resolution_clock::now();

        if (ret1 && ret2 && ret3) {
            flip(img1, img1, -1); //旋转摄像头画面180°
            flip(img2, img2, -1); //旋转摄像头画面180°
            flip(img3, img3, -1); //旋转摄像头画面180°

            img1 = undistortImage(img1, cameraMatrix, distCoeffs);//畸变矫正
            img2 = undistortImage(img2, cameraMatrix, distCoeffs);//畸变矫正
            img3 = undistortImage(img3, cameraMatrix, distCoeffs);//畸变矫正

            img1 = perspectiveTransformation(img1, srcPoints, dstPoints);//透视变换
            img2 = perspectiveTransformation(img2, srcPoints, dstPoints);//透视变换
            img3 = perspectiveTransformation(img3, srcPoints, dstPoints);//透视变换

            //imshow("cam1", img1);
            //imshow("cam2", img2);
            //imshow("cam3", img3);

            Mat finalStitch = smartStitch(img1, img2, d);//拼接
            finalStitch = smartStitch(finalStitch, img3, d);//拼接

            // // 获取图片的大小
            // cv::Size size = finalStitch.size();
            // std::cout << "Width: " << size.width << ", Height: " << size.height << std::endl;

            imshow("stitch", finalStitch);
        }
        else {
            cout << "摄像头读取失败，等待中..." << endl;
        }


        // 结束计时
        auto end_2 = std::chrono::high_resolution_clock::now();
        auto elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count() / 1000.0;

        // 算法1：计算读取3张图片并处理的总耗时
        // 总时间
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_1).count() / 1000.0;
        // 计算帧率
        auto fps = 1000.0f / duration;

        // 如果计算帧率，输出帧率
        // 输出时间：读取视频帧的时间、处理运行时间、总时间
        printf("Method1 Time: %fms, %fms, %fms\n", elapsed_1, elapsed_2, duration);
        // 输出帧率：读取视频帧的帧率、处理运行帧率、总帧率
        printf("Method1 FPS: %f, %f, %f\n", 1000.0 / elapsed_1, 1000.0 / elapsed_2, fps);



        if (waitKey(delay) == 27) { // Press Esc to exit
            stop = true;
            cout << "程序结束！" << endl;
            cout << "*** ***" << endl;
        }
    }

    cap1.release();
    cap2.release();
    cap3.release();
    destroyAllWindows();
    return 0;
}
