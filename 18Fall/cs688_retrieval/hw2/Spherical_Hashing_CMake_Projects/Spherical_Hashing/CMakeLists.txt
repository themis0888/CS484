project(Spherical_Hashing)
cmake_minimum_required(VERSION 2.8)

set(SRC_LIST
    Common.cpp
    Common.h
    Evaluation.h
    Kernel.cpp
    Kernel.h
    Kernel_SphericalHashing.cpp
    Kernel_SphericalHashing.h
    LSH.h
    Parameters.h
    Points.h
    SphericalHashing.cpp
    SphericalHashing.h
    Stopwatch.cpp
    Stopwatch.hpp
)

add_library(${PROJECT_NAME} ${SRC_LIST})
