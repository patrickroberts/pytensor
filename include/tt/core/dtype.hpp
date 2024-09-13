#pragma once

#include <tt/core/float.hpp>
#include <tt/core/int.hpp>

namespace tt {
inline namespace core {

enum class dtype {
  Float32,
  Float64,
  BFloat16,
  UInt8,
  Int8,
  Int16,
  Int32,
  Int64,
  Bool,
};

template <class T, tt::dtype V>
struct dtype_traits {
  using type = T;
  static constexpr auto value = V;
};

struct dtypes : dtype_traits<tt::Float32, tt::dtype::Float32>,
                dtype_traits<tt::Float64, tt::dtype::Float64>,
                dtype_traits<tt::BFloat16, tt::dtype::BFloat16>,
                dtype_traits<tt::UInt8, tt::dtype::UInt8>,
                dtype_traits<tt::Int8, tt::dtype::Int8>,
                dtype_traits<tt::Int16, tt::dtype::Int16>,
                dtype_traits<tt::Int32, tt::dtype::Int32>,
                dtype_traits<tt::Int64, tt::dtype::Int64>,
                dtype_traits<tt::Bool, tt::dtype::Bool> {
  template <class T, tt::dtype V>
  using fn = dtype_traits<T, V>;
};

} // namespace core
} // namespace tt
