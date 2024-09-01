#pragma once

#include <tt/core/concepts.hpp>

namespace tt::inline core {

template <tt::arithmetic T>
struct dtype_t {};

template <tt::arithmetic T>
inline constexpr dtype_t<T> dtype{};

} // namespace tt::inline core
