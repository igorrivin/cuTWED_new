# Python bindings for cuTWED
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Install Python files
install(
    FILES
        cutwed.py
        setup.py
    DESTINATION 
        ${CMAKE_INSTALL_PREFIX}/python
)

# Generate a setup.cfg file with correct paths
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.cfg
    @ONLY
)

# Copy generated setup.cfg to install directory
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/setup.cfg
    DESTINATION 
        ${CMAKE_INSTALL_PREFIX}/python
)

# Add a custom target to build Python package
add_custom_target(python_package
    COMMAND ${Python3_EXECUTABLE} setup.py build
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/python
    COMMENT "Building Python package"
    DEPENDS cuTWED
)

# Add a custom target to install Python package
add_custom_target(python_install
    COMMAND ${Python3_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/python
    COMMENT "Installing Python package"
    DEPENDS python_package
)