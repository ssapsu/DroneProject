// tests/test_object_detector.cpp
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.h"

TEST(ObjectDetectorTest, DetectObjects) {
    // 모델 파일 경로와 클래스 이름 파일 경로를 지정합니다.
    std::string modelPath = "models/yolov8n.onnx";
    std::string classNamesPath = "models/coco.names";

    // 샘플 이미지를 불러옵니다.
    cv::Mat image = cv::imread("tests/sample.jpg");
    ASSERT_FALSE(image.empty()) << "샘플 이미지를 불러올 수 없습니다.";

    // ObjectDetector 인스턴스를 생성합니다.
    ObjectDetector detector(modelPath, classNamesPath);

    // 이미지를 입력하여 물체를 감지합니다.
    std::vector<Detection> detections = detector.detect(image);

    // 감지된 객체가 있는지 확인합니다.
    EXPECT_GT(detections.size(), 0) << "감지된 객체가 없습니다.";
}
