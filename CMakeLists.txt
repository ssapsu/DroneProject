cmake_minimum_required(VERSION 3.10)
project(CameraProject)

# C++17 사용 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV와 GTest 패키지 찾기
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

# 테스트 활성화
enable_testing()

# Camera 라이브러리 생성
add_library(Camera src/Camera.cpp)
target_include_directories(Camera PUBLIC src/)
target_link_libraries(Camera PUBLIC ${OpenCV_LIBS})

# TestCamera 실행 파일 생성
add_executable(TestCamera tests/test_camera.cpp)
target_include_directories(TestCamera PRIVATE src/)
target_link_libraries(TestCamera PRIVATE Camera GTest::GTest GTest::Main)

# 테스트 등록
add_test(NAME CameraTest COMMAND TestCamera)
