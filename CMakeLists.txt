cmake_minimum_required(VERSION 3.10)
project(CameraProject)

# C++17 사용 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV와 GTest, Torch 패키지 찾기
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Torch REQUIRED)

# 테스트 활성화
enable_testing()

# Camera 라이브러리 생성
add_subdirectory(src)
add_subdirectory(tests)
