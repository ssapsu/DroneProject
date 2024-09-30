// ObjectDistanceDetector.cpp

#include "ObjectDistanceDetector.h"
#include "ObjectDetector.h"
#include "CameraConstants.h"
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

// Function to calculate focal lengths in pixels
void calculateFocalLengths(float& focal_length_px, float& focal_length_py) {
    focal_length_px = (FOCAL_LENGTH_MM * SENSOR_RESOLUTION_X) / SENSOR_WIDTH_MM;
    focal_length_py = (FOCAL_LENGTH_MM * SENSOR_RESOLUTION_Y) / SENSOR_HEIGHT_MM;
}

// Function to calculate distance to an object
float distanceToCamera(float knownDimension, float focalLength, float perDimension) {
    return (knownDimension * focalLength) / perDimension;
}

// Function to process detections and calculate distances
void calculateObjectDistances(const std::vector<Detection>& detections,const cv::Mat& image) {
    // Calculate focal lengths
    float focal_length_px, focal_length_py;
    calculateFocalLengths(focal_length_px, focal_length_py);

    // Iterate over each detection
    for (const auto& detection : detections) {
        // Extract bounding box dimensions
        float box_width_px = detection.box.width;
        float box_height_px = detection.box.height;

        // Determine the longer side
        float longer_side_px = std::max(box_width_px, box_height_px);
        float shorter_side_px = std::min(box_width_px, box_height_px);

        float focal_length = focal_length_px; // Default to horizontal focal length

        // Variables for known dimension and label
        float known_dimension = 0.0f;
        std::string object_label;

        // Determine object type based on class ID
        if (detection.class_id == CLASS_ID_PARCEL) {
            // For parcels, use width or height
            bool is_width_longer = (box_width_px >= box_height_px);
            known_dimension = is_width_longer ? PARCEL_WIDTH : PARCEL_HEIGHT;
            object_label = "Parcel";
        } else if (detection.class_id == CLASS_ID_RING) {
            // For rings, use diameter (average of width and height)
            known_dimension = PARCEL_DIAMETER;
            // For more accuracy, you might consider averaging the width and height
            longer_side_px = std::max(box_width_px, box_height_px);
            object_label = "Ring";
        } else {
            // Unknown object class
            std::cerr << "Unknown class ID: " << detection.class_id << std::endl;
            continue; // Skip this detection
        }

        // Calculate the distance
        float distance = distanceToCamera(known_dimension, focal_length, longer_side_px);

        // Output results
        std::cout << "Object: " << object_label
                  << ", Class ID: " << detection.class_id
                  << ", Confidence: " << detection.confidence
                  << ", Distance: " << distance << " cm" << std::endl;

        // Draw bounding box
        cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), 2);

        // Display distance information
        std::string label = object_label + ": " + std::to_string(static_cast<int>(distance)) + " cm";
        cv::putText(image, label, cv::Point(detection.box.x, detection.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}
