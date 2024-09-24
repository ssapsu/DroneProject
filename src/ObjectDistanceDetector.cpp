#include "ObjectDistanceDetector.h"
#include "ObjectDetector.h"
#include "CameraConstants.h"  // For camera-related constants
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

// Calculate focal length in pixels based on sensor size, resolution, and focal length in mm
float calculateFocalLength() {
    float pixel_size_mm = SENSOR_WIDTH_MM / SENSOR_RESOLUTION_X;  // Pixel size in mm
    return FOCAL_LENGTH_MM / pixel_size_mm;  // Focal length in pixels
}

// Calculate distance to an object using its perceived width in the image
float distanceToCamera(float knownWidth, float focalLength, float perWidth) {
    return (knownWidth * focalLength) / perWidth;
}

// Process detections and calculate distance for each detected object
void processDetections(const std::vector<Detection>& detections, const cv::Mat& image) {
    // Calculate focal length in pixels
    float focal_length_px = calculateFocalLength();

    // Iterate over each detected object
    for (const auto& detection : detections) {
        // Extract the width of the bounding box
        float box_width_px = std::min(detection.box.width, detection.box.height);

        // Calculate the distance to the object
        float distance = distanceToCamera(PARCEL_WIDTH, focal_length_px, box_width_px);

        // Display the results
        std::cout << "Class ID: " << detection.class_id
                  << ", Confidence: " << detection.confidence
                  << ", Distance to object: " << distance << " cm" << std::endl;

        // Draw the bounding box and distance on the image
        cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Distance: " + std::to_string(static_cast<int>(distance)) + " cm";
        cv::putText(image, label, cv::Point(detection.box.x, detection.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}
