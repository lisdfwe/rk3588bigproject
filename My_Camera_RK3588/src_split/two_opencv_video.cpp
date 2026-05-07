#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 初始化摄像头
    cv::VideoCapture cap1(0);
    cv::VideoCapture cap2(2);

    // 使用V4L2后端并设置MJPG格式（避免GStreamer警告）
    // cap1.open(0, cv::CAP_V4L2);
    // cap2.open(2, cv::CAP_V4L2);
    
    // 设置MJPG格式
    cap1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    // 设置分辨率
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // 帧率和延迟
    int rate = 60;
    int delay = static_cast<int>(1000 / rate);
    bool stop = false;
    int d = 100;  // 渐入渐出融合宽度（减小）
    cv::Mat homography;
    int k = 0;

    // 创建窗口
    cv::namedWindow("cam1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("cam2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("stitch", cv::WINDOW_AUTOSIZE);

    // 检查摄像头是否打开
    if (cap1.isOpened() && cap2.isOpened()) {
        std::cout << "*** ***\n";
        std::cout << "摄像头已启动！\n";
    }
    else {
        std::cout << "*** ***\n";
        std::cout << "警告：请检查摄像头是否安装好!\n";
        std::cout << "程序结束！\n";
        return -1;
    }

    // 调整摄像头的焦距（某些摄像头不支持此属性，忽略警告）
    cap1.set(cv::CAP_PROP_FOCUS, 0);
    cap2.set(cv::CAP_PROP_FOCUS, 0);

    while (!stop) {
        cv::Mat img1, img2;
        bool ret1 = cap1.read(img1);
        bool ret2 = cap2.read(img2);

        if (ret1 && ret2) {
            // 检查图像是否为空
            if (img1.empty() || img2.empty()) {
                std::cout << "读取到空帧，跳过...\n";
                continue;
            }

            // 显示摄像头画面
            cv::imshow("cam1", img1);
            cv::imshow("cam2", img2);

            // 计算单应矩阵
            if (k < 1 || cv::waitKey(delay) == 13) {
                std::cout << "正在匹配...\n";

                // 使用ORB进行特征检测和描述符计算
                cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);  // 增加特征点数量

                std::vector<cv::KeyPoint> keypoints1, keypoints2;
                cv::Mat descriptors1, descriptors2;

                orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
                orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

                // 确保描述符存在
                if (descriptors1.empty() || descriptors2.empty()) {
                    std::cout << "未检测到特征点，跳过当前帧...\n";
                    continue;
                }

                // 检查特征点数量是否足够
                if (keypoints1.size() < 10 || keypoints2.size() < 10) {
                    std::cout << "特征点数量不足，跳过当前帧...\n";
                    continue;
                }

                // 使用KNN匹配 + 比率测试
                cv::BFMatcher bf(cv::NORM_HAMMING);
                std::vector<std::vector<cv::DMatch>> knnMatches;
                bf.knnMatch(descriptors2, descriptors1, knnMatches, 2);  // 注意：img2匹配到img1

                // 比率测试筛选好的匹配
                std::vector<cv::DMatch> goodMatches;
                const float ratioThresh = 0.75f;
                for (const auto& m : knnMatches) {
                    if (m.size() >= 2 && m[0].distance < ratioThresh * m[1].distance) {
                        goodMatches.push_back(m[0]);
                    }
                }

                std::cout << "好的匹配点数: " << goodMatches.size() << "\n";

                // 检查匹配点数量
                if (goodMatches.size() < 10) {
                    std::cout << "匹配点数量不足(" << goodMatches.size() << ")，跳过当前帧...\n";
                    continue;
                }

                // 筛选匹配点 (img2的点 -> img1的点)
                std::vector<cv::Point2f> srcPoints, dstPoints;
                for (const auto& match : goodMatches) {
                    srcPoints.push_back(keypoints2[match.queryIdx].pt);  // img2的点
                    dstPoints.push_back(keypoints1[match.trainIdx].pt);  // img1的点
                }

                // 计算单应性矩阵 (img2 -> img1)
                std::vector<uchar> inliersMask;
                cv::Mat H = cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 4.0, inliersMask);
                
                // 统计内点数量
                int inliersCount = cv::countNonZero(inliersMask);
                std::cout << "RANSAC内点数: " << inliersCount << "\n";

                // 检查单应矩阵是否有效
                if (H.empty()) {
                    std::cout << "单应矩阵计算失败，跳过当前帧...\n";
                    continue;
                }

                // 检查单应矩阵是否合理（放宽条件）
                double det = cv::determinant(H(cv::Rect(0, 0, 2, 2)));
                std::cout << "单应矩阵行列式: " << det << "\n";
                
                if (std::abs(det) < 0.01 || std::abs(det) > 100.0 || inliersCount < 8) {
                    std::cout << "单应矩阵不合理，跳过当前帧...\n";
                    continue;
                }

                homography = H.clone();
                std::cout << "匹配成功！\n";
                k++;
            }

            // 图像拼接 (简单左右拼接)
            if (!homography.empty()) {
                // 将img2变换到img1的坐标系
                int result_width = img1.cols + img2.cols;
                int result_height = img1.rows;

                // 创建平移矩阵，将img1放到右侧
                cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
                translation.at<double>(0, 2) = img2.cols;  // x方向平移

                // 组合变换
                cv::Mat H_combined = translation * homography;

                cv::Mat result = cv::Mat::zeros(result_height, result_width, img1.type());
                
                // 先放置img2在左侧
                img2.copyTo(result(cv::Rect(0, 0, img2.cols, img2.rows)));
                
                // 将img1变换后放置
                cv::Mat warped;
                cv::warpPerspective(img1, warped, H_combined, cv::Size(result_width, result_height));
                
                // 融合：简单叠加（非零像素覆盖）
                for (int y = 0; y < result.rows; y++) {
                    for (int x = 0; x < result.cols; x++) {
                        cv::Vec3b pixel = warped.at<cv::Vec3b>(y, x);
                        if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0) {
                            // 在重叠区域做融合
                            if (x < img2.cols && x >= img2.cols - d) {
                                float alpha = static_cast<float>(img2.cols - x) / d;
                                cv::Vec3b orig = result.at<cv::Vec3b>(y, x);
                                result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                                    static_cast<uchar>(alpha * orig[0] + (1 - alpha) * pixel[0]),
                                    static_cast<uchar>(alpha * orig[1] + (1 - alpha) * pixel[1]),
                                    static_cast<uchar>(alpha * orig[2] + (1 - alpha) * pixel[2])
                                );
                            } else if (x >= img2.cols) {
                                result.at<cv::Vec3b>(y, x) = pixel;
                            }
                        }
                    }
                }

                cv::imshow("stitch", result);
            }

        }
        else {
            std::cout << "----------------------\n";
            std::cout << "等待中...\n";
        }

        if (cv::waitKey(1) == 27) {  // 按下Esc键退出
            stop = true;
            std::cout << "程序结束！\n*** ***\n";
        }
    }

    cap1.release();
    cap2.release();
    cv::destroyAllWindows();

    return 0;
}

