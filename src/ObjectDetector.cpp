#include "ObjectDetector.h"
#include "utils.h"
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
using torch::indexing::Slice; // 추가된 부분
using torch::indexing::None;  // 추가된 부분

// ObjectDetector 생성자
ObjectDetector::ObjectDetector(const std::string& modelPath, const std::string& classNamesPath, float confThreshold, float nmsThreshold)
    : confThreshold(confThreshold), nmsThreshold(nmsThreshold) {

    // TorchScript 모델 로드
    try {
        model = torch::jit::load(modelPath);  // 모델 로드
        model.eval();  // 평가 모드 설정
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);  // CUDA 또는 CPU 선택
        model.to(device);
    } catch (const c10::Error& e) {
        throw std::runtime_error("모델을 로드하는 데 실패했습니다: " + modelPath);
    }

    // 클래스 이름 로드
    loadClassNames(classNamesPath);
}

void ObjectDetector::loadClassNames(const std::string& classNamesPath) {
    std::ifstream ifs(classNamesPath);
    if (!ifs.is_open()) {
        throw std::runtime_error("클래스 이름 파일을 열 수 없습니다: " + classNamesPath);
    }

    std::string line;
    while (std::getline(ifs, line)) {
        classNames.push_back(line);
    }

    // // 클래스 이름을 출력하여 확인
    // for (const auto& className : classNames) {
    //     std::cout << className << std::endl;
    // }
}

// 객체 탐지 함수
std::vector<Detection> ObjectDetector::detect(const cv::Mat& frame) {
    // 이미지 전처리
    cv::Mat input_image;
    float resize_scale = letterbox(frame, input_image, {640, 640});

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);  // CUDA 또는 CPU 선택

    torch::Tensor image_tensor = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols, 3}, torch::kByte).to(device);
    image_tensor = image_tensor.permute({0, 3, 1, 2}).toType(torch::kFloat32).div(255);

    // 모델 추론
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image_tensor);
    auto output = model.forward(inputs).toTensor();

    // 추론 결과 후처리 (NMS 포함)
    auto keep = non_max_suppression(output, confThreshold, nmsThreshold)[0];

    // 감지 결과를 저장
    std::vector<Detection> detections;
    for (int i = 0; i < keep.size(0); ++i) {
        auto box = keep[i].slice(0, 0, 4);
        auto conf = keep[i][4].item<float>();
        auto class_id = static_cast<int>(keep[i][5].item<float>());

        float x1 = box[0].item<float>() * frame.cols;
        float y1 = box[1].item<float>() * frame.rows;
        float x2 = box[2].item<float>() * frame.cols;
        float y2 = box[3].item<float>() * frame.rows;

        Detection detection;
        detection.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        detection.confidence = conf;
        detection.class_id = class_id;

        detections.push_back(detection);
    }

    return detections;
}
