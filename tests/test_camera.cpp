// tests/test_camera.cpp
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "Camera.h"

TEST(CameraTest, CaptureFrame) {
    Camera camera(0); // 디바이스 ID 0번 카메라 사용
    cv::Mat frame = camera.getFrame();
    EXPECT_FALSE(frame.empty()); // 프레임이 비어있지 않은지 확인
}
