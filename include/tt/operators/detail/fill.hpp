#pragma once

#include <tt/operators/full.hpp>

namespace tt::inline operators {
namespace detail {

template <tt::arithmetic T, int V>
struct fill_fn {
private:
  static constexpr T fill_value{V};

public:
  template <class... TIndices>
  constexpr tt::row_major_tensor<T, tt::extents_from<TIndices...>>
  operator()(TIndices... extents) const {
    return tt::full(fill_value, extents...);
  }
};

} // namespace detail
} // namespace tt::inline operators
