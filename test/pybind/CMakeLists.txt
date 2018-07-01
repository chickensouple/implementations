cmake_minimum_required(VERSION 3.5)   # CMake version check
project(pybind_example)
set(CMAKE_CXX_STANDARD 11)            # Enable c++11 standard


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(pybind11)
pybind11_add_module(example src/example.cpp)

#set(SOURCE_FILES src/example.cpp)            # Add main.cpp file of project root directory as source file
#add_executable(pybind_example ${SOURCE_FILES})       # Add executable target with source files listed in SOURCE_FILES variable

