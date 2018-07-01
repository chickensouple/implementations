cmake_minimum_required (VERSION 2.6)
project (hashheap)

set(CMAKE_CXX_STANDARD 11)


include_directories(include)
add_executable(hashheaptest tests/test.cpp include/hashheap/hashheap.hpp)


##############
## BINDINGS ##
##############
add_subdirectory(pybind11)
pybind11_add_module(HashHeap bindings/python/hashheap_python_bindings.cpp)


