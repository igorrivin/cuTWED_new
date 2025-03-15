# CMake generated Testfile for 
# Source directory: /Users/igorrivin/devel/cuTWED_new/tests
# Build directory: /Users/igorrivin/devel/cuTWED_new/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[TestPython]=] "/opt/anaconda3/bin/python3.12" "-m" "pytest" "python_test.py" "-v")
set_tests_properties([=[TestPython]=] PROPERTIES  WORKING_DIRECTORY "/Users/igorrivin/devel/cuTWED_new/tests" _BACKTRACE_TRIPLES "/Users/igorrivin/devel/cuTWED_new/tests/CMakeLists.txt;23;add_test;/Users/igorrivin/devel/cuTWED_new/tests/CMakeLists.txt;0;")
