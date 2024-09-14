// src/Camera.cpp
#include "Camera.h"

Camera::Camera(int deviceID) {
    cap.open(deviceID);
    if (!cap.isOpened()) {
        throw std::runtime_error("카메라를 열 수 없습니다.");
    }
}

cv::Mat Camera::getFrame() {
    cv::Mat frame;
    cap >> frame;
    return frame;
}
