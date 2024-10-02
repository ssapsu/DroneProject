#include "Camera.h"
#include "ObjectDetector.h"
#include "ObjectDistanceDetector.h"  // Include the distance calculation functions
#include "CameraConstants.h"         // Include the camera constants
#include <opencv2/opencv.hpp>
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
        std::string model_path = projectRoot + "/models/best_ringnParcel.torchscript";
        std::string class_names_path = projectRoot + "/models/parcel.txt";

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file not found at " + model_path);
        }
        if (!std::filesystem::exists(class_names_path)) {
            throw std::runtime_error("Class names file not found at " + class_names_path);
        }

        ObjectDetector detector(model_path, class_names_path, 0.5f, 0.4f);

        // Set up camera matrix and distortion coefficients for undistortion
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
            FOCAL_LENGTH_PX, 0, PRINCIPAL_POINT_X,
            0, FOCAL_LENGTH_PY, PRINCIPAL_POINT_Y,
            0, 0, 1);

        cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) <<
            DISTORTION_COEFFS[0],
            DISTORTION_COEFFS[1],
            DISTORTION_COEFFS[2],
            DISTORTION_COEFFS[3],
            DISTORTION_COEFFS[4]);

        // Compute the optimal new camera matrix (optional, but can improve results)
        cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs,
                                                                cv::Size(SENSOR_RESOLUTION_X, SENSOR_RESOLUTION_Y),
                                                                1,
                                                                cv::Size(SENSOR_RESOLUTION_X, SENSOR_RESOLUTION_Y),
                                                                0);

        while (true) {
            try {
                // Capture frame
                cv::Mat frame = camera.getFrame();

                if (frame.empty()) {
                    std::cerr << "Received empty frame. Skipping." << std::endl;
                    continue;
                }

                // Undistort the frame
                cv::Mat undistortedFrame;
                cv::undistort(frame, undistortedFrame, cameraMatrix, distCoeffs, newCameraMatrix);

                // Perform object detection on the undistorted frame
                std::vector<Detection> detections = detector.detect(undistortedFrame);

                // Calculate object distances and draw results
                calculateObjectDistances(detections, undistortedFrame);

                // Display the frame
                cv::imshow("Object Detection", undistortedFrame);

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
