cmake_minimum_required(VERSION 3.10)

project(nn-rust LANGUAGES NONE)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/target/release)
target_link_libraries(${PROJECT_NAME} INTERFACE libnn320)
