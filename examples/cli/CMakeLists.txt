add_executable(facedetect-cli main.cpp)
target_link_libraries(facedetect-cli PRIVATE facedetect)
target_include_directories(facedetect-cli PRIVATE ${CMAKE_SOURCE_DIR}/src)
