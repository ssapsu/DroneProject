cmake_minimum_required(VERSION 3.10)
project(CameraProject)


# Get the absolute path to the project root directory
set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Optionally, print the project root for debugging purposes
message(STATUS "Project root directory: ${PROJECT_ROOT_DIR}")

set(OpenCV_DIR "/usr/lib/cmake/opencv")
set(Torch_DIR "/home/singleuav_causlab/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")

# C++17 사용 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV와 GTest, Torch 패키지 찾기
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Torch REQUIRED PATHS ${Torch_DIR})

# Torch 패키지 경로를 CMAKE_PREFIX_PATH에 추가
list(APPEND CMAKE_PREFIX_PATH ${Torch_DIR})

# 테스트 활성화
enable_testing()

# Camera 라이브러리 생성
add_subdirectory(src)
add_subdirectory(tests)
