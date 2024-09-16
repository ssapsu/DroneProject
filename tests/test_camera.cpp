// tests/test_camera.cpp
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "Camera.h"

TEST(CameraTest, CaptureFrame_Webcam) {
    Camera camera(0, CameraType::WEBCAM);
    cv::Mat frame = camera.getFrame();
    EXPECT_FALSE(frame.empty()) << "웹캠에서 프레임을 가져올 수 없습니다.";
}

TEST(CameraTest, CaptureFrame_CSI) {
    // CSI 카메라가 있는 경우에만 테스트
    // 환경 변수나 설정 파일을 통해 CSI 카메라 사용 여부를 결정할 수 있습니다.
    Camera camera(0, CameraType::CSI);
    cv::Mat frame = camera.getFrame();
    EXPECT_FALSE(frame.empty()) << "CSI 카메라에서 프레임을 가져올 수 없습니다.";
}
