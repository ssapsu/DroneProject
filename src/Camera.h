// src/Camera.h
#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>

class Camera {
public:
    explicit Camera(int deviceID = 0);
    cv::Mat getFrame();
private:
    cv::VideoCapture cap;
};

#endif // CAMERA_H
