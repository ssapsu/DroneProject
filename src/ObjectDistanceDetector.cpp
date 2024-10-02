// ObjectDistanceDetector.cpp

#include "ObjectDistanceDetector.h"
#include "ObjectDetector.h"
#include "CameraConstants.h"
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

// Function to calculate the distance to an object
float distanceToCamera(float knownDimension, float focalLength, float perDimension) {
    return (knownDimension * focalLength) / perDimension;
}

// Function to undistort the image (if you choose to undistort the entire image)
// Optional: Only necessary if you decide to undistort the whole image before processing
cv::Mat undistortImage(const cv::Mat& distortedImage) {
    // Set up the camera matrix
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        FOCAL_LENGTH_PX, 0, PRINCIPAL_POINT_X,
        0, FOCAL_LENGTH_PY, PRINCIPAL_POINT_Y,
        0, 0, 1);

    // Set up the distortion coefficients
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) <<
        DISTORTION_COEFFS[0],
        DISTORTION_COEFFS[1],
        DISTORTION_COEFFS[2],
        DISTORTION_COEFFS[3],
        DISTORTION_COEFFS[4]);

    cv::Mat undistortedImage;
    cv::undistort(distortedImage, undistortedImage, cameraMatrix, distCoeffs);
    return undistortedImage;
}

// Function to process detections and calculate distances
void calculateObjectDistances(const std::vector<Detection>& detections, cv::Mat& image) {
    // Camera matrix and distortion coefficients
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

    // Iterate over each detection
    for (const auto& detection : detections) {
        // Extract bounding box corners
        std::vector<cv::Point2f> distortedPoints = {
            cv::Point2f(detection.box.x, detection.box.y),  // Top-left
            cv::Point2f(detection.box.x + detection.box.width, detection.box.y),  // Top-right
            cv::Point2f(detection.box.x, detection.box.y + detection.box.height),  // Bottom-left
            cv::Point2f(detection.box.x + detection.box.width, detection.box.y + detection.box.height)  // Bottom-right
        };

        // Undistort points
        std::vector<cv::Point2f> undistortedPoints;
        cv::undistortPoints(distortedPoints, undistortedPoints, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix);

        // Calculate undistorted bounding box dimensions
        float undistortedWidth = std::abs(undistortedPoints[1].x - undistortedPoints[0].x);
        float undistortedHeight = std::abs(undistortedPoints[2].y - undistortedPoints[0].y);

        // Determine the longer side
        float longer_side_px = std::max(undistortedWidth, undistortedHeight);

        // Variables for known dimension and label
        float known_dimension = 0.0f;
        std::string object_label;

        // Determine object type based on class ID
        if (detection.class_id == CLASS_ID_PARCEL) {
            // For parcels, use width or height
            bool is_width_longer = (undistortedWidth >= undistortedHeight);
            known_dimension = is_width_longer ? PARCEL_WIDTH : PARCEL_HEIGHT;
            object_label = "Parcel";
        } else if (detection.class_id == CLASS_ID_RING) {
            // For rings, use diameter
            known_dimension = PARCEL_DIAMETER;
            object_label = "Ring";
        } else {
            // Unknown object class
            std::cerr << "Unknown class ID: " << detection.class_id << std::endl;
            continue; // Skip this detection
        }

        // Choose appropriate focal length based on orientation
        float focal_length = (undistortedWidth >= undistortedHeight) ? FOCAL_LENGTH_PX : FOCAL_LENGTH_PY;

        // Calculate the distance
        float distance = distanceToCamera(known_dimension, focal_length, longer_side_px);

        // Output results
        std::cout << "Object: " << object_label
                  << ", Class ID: " << detection.class_id
                  << ", Confidence: " << detection.confidence
                  << ", Distance: " << distance << " cm" << std::endl;

        // Draw bounding box on the original image
        cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), 2);

        // Display distance information
        std::string label = object_label + ": " + std::to_string(static_cast<int>(distance)) + " cm";
        cv::putText(image, label, cv::Point(detection.box.x, detection.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}
