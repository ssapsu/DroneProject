#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>  // TorchScript를 위한 헤더 추가
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class ObjectDetector {
public:
    // 생성자에서 TorchScript 모델 경로와 클래스 이름 파일 경로를 받음
    ObjectDetector(const std::string& modelPath, const std::string& classNamesPath, float confThreshold = 0.5f, float nmsThreshold = 0.4f);

    // 객체 탐지를 수행하는 함수
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    // TorchScript 모델을 위한 변수
    torch::jit::script::Module model;

    // 클래스 이름 저장
    std::vector<std::string> classNames;

    // 임계값
    float confThreshold;
    float nmsThreshold;

    // 클래스 이름 로드 함수
    void loadClassNames(const std::string& classNamesPath);
};

#endif // OBJECT_DETECTOR_H
