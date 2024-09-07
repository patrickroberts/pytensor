cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(
  PyTensor
  VERSION 1.0.0
  LANGUAGES CXX)

include(cmake/python-config.cmake)
include(cmake/deps-config.cmake)

add_library(tensor_flags INTERFACE)
target_compile_options(tensor_flags INTERFACE -Wall -Wextra -Werror)
target_include_directories(tensor_flags
                           INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(tensor_flags INTERFACE boost_mp11 fmt::fmt-header-only
                                             magic_enum mdspan)

nanobind_add_module(_tt src/tt.cpp)
target_link_libraries(_tt PRIVATE tensor_flags)

install(TARGETS _tt LIBRARY DESTINATION tt)

set(TT_EXAMPLES
    ON
    CACHE BOOL "Build targets in examples")

if(TT_EXAMPLES)
  add_subdirectory(examples)
endif()
