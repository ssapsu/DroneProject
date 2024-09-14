# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(CameraProject)

# C++17 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV와 GTest 패키지 찾기
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

# include 디렉토리 설정
include_directories(${OpenCV_INCLUDE_DIRS})

# 소스 파일 추가
add_library(Camera src/Camera.cpp)

# Camera 라이브러리에 헤더 파일 경로 추가
target_include_directories(Camera PUBLIC src/)

# 테스트 실행 파일 추가
add_executable(TestCamera tests/test_camera.cpp)

# TestCamera 대상에 헤더 파일 경로 추가
target_include_directories(TestCamera PRIVATE src/)

# 라이브러리 링크
target_link_libraries(Camera ${OpenCV_LIBS})
target_link_libraries(TestCamera Camera ${OpenCV_LIBS} GTest::GTest GTest::Main)

# 테스트 등록
enable_testing()
add_test(NAME CameraTest COMMAND TestCamera)
