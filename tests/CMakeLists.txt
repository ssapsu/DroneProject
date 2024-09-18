# tests/CMakeLists.txt

# Find GTest
find_package(GTest REQUIRED)

# Test for ObjectDetector
add_executable(TestObjectDetector test_object_detector.cpp)
target_include_directories(TestObjectDetector PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(TestObjectDetector PRIVATE Camera GTest::GTest GTest::Main)

# Register tests
add_test(NAME ObjectDetectorTest COMMAND TestObjectDetector)
