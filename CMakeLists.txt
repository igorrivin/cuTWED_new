cmake_minimum_required(VERSION 3.18...3.26)
project(cuTWED 
        VERSION 3.0.0
        DESCRIPTION "A linear memory CUDA algorithm for solving Time Warp Edit Distance"
        LANGUAGES CXX)

# Make CUDA optional
option(CUTWED_USE_CUDA "Build with CUDA support" ON)

if(CUTWED_USE_CUDA)
    enable_language(CUDA)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard if enabled
if(CUTWED_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Add support for newer CUDA architectures (Pascal, Volta, Turing, Ampere, Hopper)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;90")
endif()

# Options
option(CUTWED_BUILD_TESTS "Build cuTWED tests" ON)
option(CUTWED_USE_DOUBLE "Build cuTWED with double precision" ON)
option(CUTWED_BUILD_PYTHON "Build Python bindings" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library sources
if(CUTWED_USE_CUDA)
    set(CUTWED_SOURCES
        src/cuTWED.cu
    )
else()
    # Create empty library for CPU-only backends
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp "// Empty file for CPU-only builds\n")
    set(CUTWED_SOURCES
        ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp
    )
endif()

# Define the library
add_library(cuTWED SHARED ${CUTWED_SOURCES})
set_target_properties(cuTWED PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Set CUDA-specific properties if enabled
if(CUTWED_USE_CUDA)
    set_target_properties(cuTWED PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Find CUDA libraries if enabled
if(CUTWED_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(cuTWED PRIVATE CUDA::cudart CUDA::cublas)
endif()

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
message(STATUS "CUTWED_USE_CUDA: ${CUTWED_USE_CUDA}")
if(CUTWED_USE_CUDA)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "CUTWED_USE_DOUBLE: ${CUTWED_USE_DOUBLE}")
message(STATUS "CUTWED_BUILD_TESTS: ${CUTWED_BUILD_TESTS}")
message(STATUS "CUTWED_BUILD_PYTHON: ${CUTWED_BUILD_PYTHON}")