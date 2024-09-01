#pragma once

#include <tt/operators/detail/fill.hpp>

namespace tt::inline operators {

template <tt::arithmetic T>
inline constexpr detail::fill_fn<T, 1> ones{};

} // namespace tt::inline operators
