#pragma once

#include <tt/operators/detail/fill.hpp>

namespace tt {
inline namespace operators {

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
inline constexpr detail::fill_fn<T, 1> ones{};

} // namespace operators
} // namespace tt
