project(Tester_L2)
cmake_minimum_required(VERSION 2.8)

include_directories(../Spherical_Hashing/)

add_executable(${PROJECT_NAME} Tester_L2_Main.cpp)
target_link_libraries(${PROJECT_NAME} Spherical_Hashing)
