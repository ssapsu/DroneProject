# tests/CMakeLists.txt

# Find GTest
find_package(GTest REQUIRED)

add_executable(TestCamera test_camera.cpp)
target_include_directories(TestCamera PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(TestCamera PRIVATE Camera GTest::GTest GTest::Main)

# Test for ObjectDetector
add_executable(TestObjectDetector test_object_detector.cpp)
target_include_directories(TestObjectDetector PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(TestObjectDetector PRIVATE ${OpenCV_LIBS} GTest::GTest GTest::Main ${TORCH_LIBRARIES} utils ObjectDetector)

# Test for ObjectDistanceDetector
add_executable(TestObjectDistanceDetector test_distance_detector.cpp)
target_include_directories(TestObjectDistanceDetector PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(TestObjectDistanceDetector PRIVATE GTest::GTest GTest::Main ${OpenCV_LIBS} ${TORCH_LIBRARIES} utils ObjectDetector ObjectDistanceDetector)

# Register tests
add_test(NAME ObjectDetectorTest COMMAND TestObjectDetector)
