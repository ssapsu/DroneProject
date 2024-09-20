#include "Camera.h"
#include "ObjectDetector.h"

#include "Camera.h"
#include "ObjectDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        // 카메라 초기화 (웹캠 사용 예시)
        Camera camera(0, CameraType::WEBCAM);  // USB 카메라 또는 CSI 카메라 선택 가능

        // ObjectDetector 초기화
        ObjectDetector detector("../models/best.torchscript", "../models/parcel.txt", 0.5f, 0.4f);  // 모델 경로와 클래스 이름 경로

        while (true) {
            // 카메라에서 프레임 읽기
            cv::Mat frame = camera.getFrame();
            if (frame.empty()) {
                std::cerr << "빈 프레임을 받았습니다. 카메라 연결을 확인하세요." << std::endl;
                break;
            }

            // 객체 디텍션 수행
            std::vector<Detection> detections = detector.detect(frame);


            // 디텍션 결과를 프레임에 그리기
            for (const auto& detection : detections) {
                cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);  // 바운딩 박스 그리기
                std::string label = detector.getClassNames()[detection.class_id] + ": " + cv::format("%.2f", detection.confidence);
                cv::putText(frame, label, detection.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }

            // 결과 프레임을 디스플레이
            cv::imshow("Object Detection", frame);

            // 'q' 키를 누르면 종료
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
    }

    return 0;
}
