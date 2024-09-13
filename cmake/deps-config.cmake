include_guard(GLOBAL)

include(FetchContent)

FetchContent_Declare(
  boost_mp11
  GIT_REPOSITORY https://github.com/boostorg/mp11.git
  GIT_TAG boost-1.86.0
  EXCLUDE_FROM_ALL)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.0.2
  EXCLUDE_FROM_ALL)
FetchContent_Declare(
  magic_enum
  GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
  GIT_TAG v0.9.6
  EXCLUDE_FROM_ALL)
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable
  EXCLUDE_FROM_ALL)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.1.0
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(boost_mp11 fmt magic_enum mdspan nanobind)
