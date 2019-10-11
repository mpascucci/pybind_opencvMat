project( cvMat_pybind )
cmake_minimum_required(VERSION 3.15)
set (CMAKE_CXX_STANDARD 11)
set(PYBIND11_CPP_STANDARD -std=c++11)

# specify here which python interpreter you want to compile this module for
set(PYTHON_EXECUTABLE "~/miniconda3/bin/python3")

# find the system installed OpenCV libraries
find_package(OpenCV 4.0.0 REQUIRED)

# The following requires pybind installed in the compiling system
add_subdirectory(pybind11)
pybind11_add_module(cvMat main.cpp)

# Link the OpenCV libraries
target_link_libraries(cvMat PRIVATE ${OpenCV_LIBS})