#pragma once

#include <cstddef>

namespace tt::inline core {

enum class DType : std::size_t {
  Float32,
  Float64,
  BFloat16,
  Uint8,
  Int8,
  Int16,
  Int32,
  Int64,
  Bool,
};

} // namespace tt::inline core
