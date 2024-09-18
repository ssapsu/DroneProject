# Camera 라이브러리 생성
add_library(Camera Camera.cpp ObjectDetector.cpp utils.cpp)
target_include_directories(Camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Camera PUBLIC ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# OpenCV 라이브러리 링크
target_link_libraries(Camera PUBLIC ${OpenCV_LIBS})

# include 경로 설정
target_include_directories(Camera PUBLIC ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
