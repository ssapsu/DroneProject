// src/ObjectDetector.cpp
#include "ObjectDetector.h"
#include <fstream>
#include <stdexcept>

ObjectDetector::ObjectDetector(const std::string& modelPath, const std::string& classNamesPath, float confThreshold, float nmsThreshold)
    : confThreshold(confThreshold), nmsThreshold(nmsThreshold) {

    // 모델 로드
    net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        throw std::runtime_error("모델을 불러올 수 없습니다: " + modelPath);
    }

    // 클래스 이름 로드
    loadClassNames(classNamesPath);
}

void ObjectDetector::loadClassNames(const std::string& classNamesPath) {
    std::ifstream ifs(classNamesPath.c_str());
    if (!ifs.is_open()) {
        throw std::runtime_error("클래스 이름 파일을 열 수 없습니다: " + classNamesPath);
    }

    std::string line;
    while (std::getline(ifs, line)) {
        classNames.push_back(line);
    }
}

std::vector<Detection> ObjectDetector::detect(const cv::Mat& frame) {
    // 입력 이미지 전처리
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);

    // 추론
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 후처리
    std::vector<Detection> detections;
    // outputs 파싱 로직은 YOLOv8 모델의 출력 형식에 따라 작성해야 합니다.
    // 아래는 일반적인 YOLO 모델의 출력 형식에 따른 예시입니다.

    const int dimensions = 85; // 클래스 수 + 5
    const int rows = outputs[0].size[1];

    float* data = (float*)outputs[0].data;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= confThreshold) {
            // 클래스 확률
            float* classScores = data + 5;
            cv::Mat scores(1, classNames.size(), CV_32FC1, classScores);
            cv::Point classIdPoint;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

            if (maxClassScore > confThreshold) {
                // 바운딩 박스 좌표 계산
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((cx - w / 2) * frame.cols);
                int top = static_cast<int>((cy - h / 2) * frame.rows);
                int width = static_cast<int>(w * frame.cols);
                int height = static_cast<int>(h * frame.rows);

                Detection det;
                det.box = cv::Rect(left, top, width, height);
                det.confidence = static_cast<float>(maxClassScore);
                det.class_id = classIdPoint.x;

                detections.push_back(det);
            }
        }
        data += dimensions;
    }

    // NMS(Non-Maximum Suppression) 적용
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (const auto& det : detections) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }

    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<Detection> nmsDetections;
    for (int idx : indices) {
        nmsDetections.push_back(detections[idx]);
    }

    return nmsDetections;
}
