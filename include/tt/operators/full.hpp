#pragma once

#include <tt/core/memory.hpp>
#include <tt/core/tensor.hpp>

namespace tt {
inline namespace operators {

struct full_fn {
  template <class T, class... TIndices>
  constexpr auto operator()(T fill_value, TIndices... extents) const
      -> TT_REQUIRES(tt::arithmetic<T>,
                     tt::RowMajorTensor<T, tt::extents_from<TIndices...>>) {
    const std::size_t size = (1 * ... * extents);

    using explicit_t = decltype(operator()(fill_value, extents...));
    return explicit_t{tt::make_shared<T[]>(size, fill_value), extents...};
  }
};

inline constexpr tt::full_fn full{};

} // namespace operators
} // namespace tt
