#include "Camera.h"
#include "ObjectDetector.h"
#include <opencv2/opencv.hpp>
#include "ObjectDistanceDetector.h"  // Include the distance calculation functions
#include <iostream>
#include <filesystem>

int main() {
    try {
        // Initialize camera
        std::cout << "Initializing camera..." << std::endl;
        Camera camera(0, CameraType::CSI);

        // Initialize ObjectDetector
        std::cout << "Initializing object detector..." << std::endl;
        std::string projectRoot = PROJECT_ROOT_DIR;
        std::string model_path = projectRoot + "/models/best.torchscript";
        std::string class_names_path = projectRoot + "/models/parcel.txt";

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file not found at " + model_path);
        }
        if (!std::filesystem::exists(class_names_path)) {
            throw std::runtime_error("Class names file not found at " + class_names_path);
        }

        ObjectDetector detector(model_path, class_names_path, 0.5f, 0.4f);

        while (true) {
            try {
                // Capture frame
                cv::Mat frame = camera.getFrame();

                if (frame.empty()) {
                    std::cerr << "Received empty frame. Skipping." << std::endl;
                    continue;
                }

                // Perform object detection
                std::vector<Detection> detections = detector.detect(frame);

                // Draw detections on the frame
                for (const auto& detection : detections) {
                    if (detection.box.area() <= 0 || detection.class_id < 0 ||
                        detection.class_id >= detector.getClassNames().size()) {
                        continue;
                    }
                    cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);
                    std::string label = detector.getClassNames()[detection.class_id] +
                                        ": " + cv::format("%.2f", detection.confidence);
                    cv::putText(frame, label, detection.box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                                0.5, cv::Scalar(255, 255, 255), 1);
                }
                processDetections(detections, frame);
                // Display the frame
                cv::imshow("Object Detection", frame);

                // Exit if 'q' is pressed
                if (cv::waitKey(1) == 'q') {
                    break;
                }

            } catch (const std::exception& e) {
                std::cerr << "Error in loop: " << e.what() << std::endl;
                // Optionally, sleep or wait before continuing
                continue;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
    }

    return 0;
}
