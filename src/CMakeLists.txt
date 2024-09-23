# Camera 라이브러리 생성
add_library(Camera Camera.cpp ObjectDetector.cpp utils.cpp)
target_include_directories(Camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Camera PUBLIC ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# ObjectDetector 라이브러리 생성
add_library(ObjectDetector ObjectDetector.cpp utils.cpp)
target_include_directories(ObjectDetector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ObjectDetector PUBLIC ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# utils 라이브러리 생성
add_library(utils utils.cpp)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(utils PUBLIC ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# ObjectDistanceDetector 라이브러리 생성
add_library(ObjectDistanceDetector ObjectDistanceDetector.cpp utils.cpp)
target_include_directories(ObjectDistanceDetector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ObjectDistanceDetector PUBLIC ${OpenCV_LIBS} ${TORCH_LIBRARIES} utils ObjectDetector)

# OpenCV 라이브러리 링크
target_link_libraries(Camera PUBLIC ${OpenCV_LIBS})
target_link_libraries(ObjectDetector PUBLIC ${OpenCV_LIBS})
target_link_libraries(utils PUBLIC ${OpenCV_LIBS})
target_link_libraries(ObjectDistanceDetector PUBLIC ${OpenCV_LIBS})

# include 경로 설정
target_include_directories(Camera PUBLIC ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_include_directories(ObjectDetector PUBLIC ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_include_directories(ObjectDistanceDetector PUBLIC ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PUBLIC ${OpenCV_LIBS} ${TORCH_LIBRARIES} Camera ObjectDetector ObjectDistanceDetector)
target_compile_definitions(main PRIVATE PROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")
