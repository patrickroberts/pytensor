cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(
  PyTensor
  VERSION 1.0.0
  LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  boost_mp11
  GIT_REPOSITORY https://github.com/boostorg/mp11.git
  GIT_TAG boost-1.86.0)
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.5)
FetchContent_MakeAvailable(boost_mp11 mdspan pybind11)

add_library(tensor_flags INTERFACE)
target_compile_options(tensor_flags INTERFACE -Wall -Wextra -Werror)
target_include_directories(tensor_flags
                           INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(tensor_flags INTERFACE boost_mp11 mdspan)

pybind11_add_module(tt lib/tt.cpp)
target_link_libraries(tt PRIVATE tensor_flags)
add_library(tensor::lib ALIAS tt)

add_subdirectory(examples)
