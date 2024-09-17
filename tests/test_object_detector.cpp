#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.h"
#include "utils.h"
#include <string>

TEST(ObjectDetectorTest, DetectObjects) {
    // 프로젝트 루트 디렉토리를 사용하여 경로 설정
    std::string projectRoot = "../";

    // TorchScript 모델과 클래스 이름 경로 설정
    std::string modelPath = projectRoot + "models/yolov8s.torchscript";  // TorchScript 모델 경로
    std::string classNamesPath = projectRoot + "models/classes.txt";  // 클래스 이름 경로
    std::string imagePath = projectRoot + "tests/images/bus.jpg";  // 테스트 이미지 경로

    // 경로 출력 (디버깅용)
    std::cout << "Model path: " << modelPath << std::endl;
    std::cout << "Class Names path: " << classNamesPath << std::endl;
    std::cout << "Image path: " << imagePath << std::endl;

    // 샘플 이미지를 불러옵니다.
    cv::Mat image = cv::imread(imagePath);
    ASSERT_FALSE(image.empty()) << "샘플 이미지를 불러올 수 없습니다.";

    // ObjectDetector 인스턴스를 생성합니다.
    ObjectDetector detector(modelPath, classNamesPath);

    // 이미지를 입력하여 물체를 감지합니다.
    std::vector<Detection> detections = detector.detect(image);

    // 감지된 객체가 있는지 확인합니다.
    EXPECT_GT(detections.size(), 0) << "감지된 객체가 없습니다.";

    // 감지된 객체 출력 (디버깅용)
    for (const auto& det : detections) {
        std::cout << "Detection: "
                  << "Class ID: " << det.class_id
                  << ", Confidence: " << det.confidence
                  << ", Box: [" << det.box.x << ", " << det.box.y
                  << ", " << det.box.width << ", " << det.box.height << "]"
                  << std::endl;
    }
}
