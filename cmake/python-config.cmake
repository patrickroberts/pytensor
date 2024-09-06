include_guard(GLOBAL)

set(PYTHON_VERSION
    3.8
    CACHE STRING "Python package version to find")
set(PYTHON_EXACT
    OFF
    CACHE BOOL "Python package version must match exactly")

if(PYTHON_EXACT)
  find_package(
    Python ${PYTHON_VERSION}
    COMPONENTS Interpreter Development.Module
    EXACT REQUIRED)
else()
  find_package(
    Python ${PYTHON_VERSION}
    COMPONENTS Interpreter Development.Module
    REQUIRED)
endif()
