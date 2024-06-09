cmake_minimum_required(VERSION 3.17)
project(infer-engine LANGUAGES C CXX CUDA)

# 
# Options
# 
option(ENABLE_GPU "Enable GPU support" ON)
option(BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")

include(FetchContent)

include_directories(include)

add_library(tensor SHARED src/data/tensor.cc)

if (BUILD_TESTS)
    message(STATUS "Building tests")
    add_subdirectory(tests)
endif()


