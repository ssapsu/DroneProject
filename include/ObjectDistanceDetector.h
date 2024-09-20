#ifndef OBJECT_DISTANCE_DETECTOR_H
#define OBJECT_DISTANCE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "CameraConstants.h"  // Camera-related constants
#include <vector>
#include "ObjectDetector.h"   // Include the Detection struct

// Function declarations for calculating focal length and distance to object

// Calculate focal length in pixels based on sensor size, resolution, and focal length in mm
float calculateFocalLength();

// Calculate distance to an object using its perceived width in the image
float distanceToCamera(float knownWidth, float focalLength, float perWidth);

// Process detections and calculate distance for each detected object
void processDetections(const std::vector<Detection>& detections, const cv::Mat& image);

#endif  // OBJECT_DISTANCE_DETECTOR_H
