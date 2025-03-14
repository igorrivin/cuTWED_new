cmake_minimum_required(VERSION 3.18...3.26)
project(cuTWED 
        VERSION 3.0.0
        DESCRIPTION "A linear memory CUDA algorithm for solving Time Warp Edit Distance"
        LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add support for newer CUDA architectures (Pascal, Volta, Turing, Ampere, Hopper)
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;90")

# Options
option(CUTWED_BUILD_TESTS "Build cuTWED tests" ON)
option(CUTWED_USE_DOUBLE "Build cuTWED with double precision" ON)
option(CUTWED_BUILD_PYTHON "Build Python bindings" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library sources
set(CUTWED_SOURCES
    src/cuTWED.cu
)

# Define the library
add_library(cuTWED SHARED ${CUTWED_SOURCES})
set_target_properties(cuTWED PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Find CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuTWED PRIVATE CUDA::cudart CUDA::cublas)

# Add compile definitions
if(CUTWED_USE_DOUBLE)
    target_compile_definitions(cuTWED PRIVATE CUTWED_USE_DOUBLE)
endif()

# Installation rules
install(TARGETS cuTWED
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Add tests if requested
if(CUTWED_BUILD_TESTS)
    # Enable testing
    enable_testing()
    add_subdirectory(tests)
endif()

# Add Python bindings if requested
if(CUTWED_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Print information
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUTWED_USE_DOUBLE: ${CUTWED_USE_DOUBLE}")
message(STATUS "CUTWED_BUILD_TESTS: ${CUTWED_BUILD_TESTS}")
message(STATUS "CUTWED_BUILD_PYTHON: ${CUTWED_BUILD_PYTHON}")