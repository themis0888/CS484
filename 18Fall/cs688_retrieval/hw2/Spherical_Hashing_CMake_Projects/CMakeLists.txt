project(Spherical_Hashing_CMake_Projects)
cmake_minimum_required(VERSION 2.8)

if( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    set(CMAKE_C_COMPILER /usr/local/bin/gcc-5)
    set(CMAKE_CXX_COMPILER /usr/local/bin/g++-5)
endif()

if( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp" )

add_subdirectory(Spherical_Hashing)
add_subdirectory(Tester_L2)

