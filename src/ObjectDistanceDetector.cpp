#include "ObjectDistanceDetector.h"
#include "ObjectDetector.h"
#include "CameraConstants.h"  // 카메라 관련 상수들
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 센서 크기와 해상도를 이용하여 픽셀 단위의 초점 거리 계산
void calculateFocalLengths(float& focal_length_px, float& focal_length_py) {
    // 가로 방향 초점 거리 계산
    focal_length_px = (FOCAL_LENGTH_MM * SENSOR_RESOLUTION_X) / SENSOR_WIDTH_MM;
    // 세로 방향 초점 거리 계산
    focal_length_py = (FOCAL_LENGTH_MM * SENSOR_RESOLUTION_Y) / SENSOR_HEIGHT_MM;
}

// 물체의 이미지에서 측정된 길이를 사용하여 거리를 계산
float distanceToCamera(float knownDimension, float focalLength, float perDimension) {
    return (knownDimension * focalLength) / perDimension;
}

// 검출된 객체들을 처리하고 각 객체의 거리를 계산
void calculateParcelDistance(const std::vector<Detection>& detections, cv::Mat& image) {
    // 픽셀 단위의 초점 거리 계산
    float focal_length_px, focal_length_py;
    calculateFocalLengths(focal_length_px, focal_length_py);

    // 각 검출된 객체에 대해 반복
    for (const auto& detection : detections) {
        // Extract the width of the bounding box
        float box_width_px = std::max(detection.box.width, detection.box.height);

        // 더 긴 변과 짧은 변 선택
        float longer_side_px = std::max(box_width_px, box_height_px);
        float shorter_side_px = std::min(box_width_px, box_height_px);

        // 더 긴 변이 가로 방향인지 세로 방향인지 확인
        bool is_width_longer = (box_width_px >= box_height_px);

        // 적절한 초점 거리와 실제 크기 선택
        float focal_length = is_width_longer ? focal_length_px : focal_length_py;
        float known_dimension = PARCEL_WIDTH;  // 물체의 실제 폭
        if (!is_width_longer) {
            known_dimension = PARCEL_HEIGHT;  // 물체의 실제 높이
        }

        // 물체의 거리를 계산
        float distance = distanceToCamera(known_dimension, focal_length, longer_side_px);

        // 결과 출력
        std::cout << "Class ID: " << detection.class_id
                  << ", Confidence: " << detection.confidence
                  << ", Distance to object: " << distance << " cm" << std::endl;

        // 바운딩 박스 그리기
        cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), 2);

        // 거리 정보 표시
        std::string label = "Distance: " + std::to_string(static_cast<int>(distance)) + " cm";
        cv::putText(image, label, cv::Point(detection.box.x, detection.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}
