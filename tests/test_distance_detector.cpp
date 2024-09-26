#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.h"
#include "ObjectDistanceDetector.h"  // Include the distance calculation functions

// TEST(ObjectDistanceTest, EstimateDistance) {
//     // Load the test image
//     cv::Mat image = cv::imread("../images/image_17.jpg");
//     ASSERT_FALSE(image.empty()) << "Failed to load image.";

//     // Initialize ObjectDetector and run detection
//     ObjectDetector detector("../models/best.torchscript", "../models/parcel.txt");
//     std::vector<Detection> detections = detector.detect(image);

//     // Ensure objects were detected
//     ASSERT_GT(detections.size(), 0) << "No objects detected.";

//     // Process the detections to calculate the distance and draw the results
//     calculateParcelDistance(detections, image);

//     // Save the output image with bounding boxes and distance
//     cv::imwrite("../output/detected_objects_with_distance.jpg", image);

//     // Test assertion: at least one object must have a valid distance
//     EXPECT_GT(detections[0].box.width, 0) << "Bounding box width should be greater than 0.";
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
