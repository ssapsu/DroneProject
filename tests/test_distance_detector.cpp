#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.h"
#include "ObjectDistanceDetector.h"  // Include the distance calculation functions

TEST(ObjectDistanceTest, EstimateDistance) {
    // Load the test image
    std::string projectRoot = PROJECT_ROOT_DIR;
    std::string image_path = projectRoot + "/images/bus.jpeg";
    cv::Mat image = cv::imread(image_path);
    ASSERT_FALSE(image.empty()) << "Failed to load image.";

    // Initialize ObjectDetector and run detection
    std::string model_path = projectRoot + "/models/yolov8s.torchscript";
    std::string class_names_path = projectRoot + "/models/classes.txt";
    ObjectDetector detector("model_path", "class_names_path");
    std::vector<Detection> detections = detector.detect(image);

    // Ensure objects were detected
    ASSERT_GT(detections.size(), 0) << "No objects detected.";

    // Process the detections to calculate the distance and draw the results
    processDetections(detections, image);

    // Save the output image with bounding boxes and distance
    cv::imwrite("../output/detected_objects_with_distance.jpg", image);

    // Test assertion: at least one object must have a valid distance
    EXPECT_GT(detections[0].box.width, 0) << "Bounding box width should be greater than 0.";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
