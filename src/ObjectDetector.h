// src/ObjectDetector.h
#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelPath, const std::string& classNamesPath, float confThreshold = 0.5f, float nmsThreshold = 0.4f);
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    float confThreshold;
    float nmsThreshold;

    void loadClassNames(const std::string& classNamesPath);
};

#endif // OBJECT_DETECTOR_H
