# Tests for cuTWED

# Enable testing
enable_testing()

# Only add CUDA-specific tests if CUDA is enabled
if(CUTWED_USE_CUDA)
    # Add test executable for CUDA
    add_executable(test_cutwed test_cutwed.cu)
    target_link_libraries(test_cutwed PRIVATE cuTWED CUDA::cudart CUDA::cublas)
    set_target_properties(test_cutwed PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Add CUDA tests
    add_test(NAME TestBasic COMMAND test_cutwed --basic)
    add_test(NAME TestBatch COMMAND test_cutwed --batch)
endif()

# Add Python tests if Python bindings are enabled
if(CUTWED_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME TestPython
        COMMAND ${Python3_EXECUTABLE} -m pytest python_test.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()