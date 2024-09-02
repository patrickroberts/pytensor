#pragma once

#include <tt/core/tensor.hpp>

namespace tt::inline operators {

struct full_fn {
  template <tt::arithmetic T, class... TIndices>
  constexpr tt::RowMajorTensor<T, tt::extents_from<TIndices...>>
  operator()(T fill_value, TIndices... extents) const {
    const std::size_t size = (1 * ... * extents);
    auto data_handle = std::make_shared<T[]>(size, fill_value);

    using explicit_t = decltype(operator()(fill_value, extents...));
    return explicit_t{std::move(data_handle), extents...};
  }
};

inline constexpr tt::full_fn full{};

} // namespace tt::inline operators
