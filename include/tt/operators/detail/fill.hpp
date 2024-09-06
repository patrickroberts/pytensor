#pragma once

#include <tt/operators/full.hpp>

namespace tt {
inline namespace operators {
namespace detail {

template <class T, int V, class = TT_REQUIRES(tt::arithmetic<T>)>
struct fill_fn {
private:
  static constexpr T fill_value{V};

public:
  template <class... TIndices>
  constexpr tt::RowMajorTensor<T, tt::extents_from<TIndices...>>
  operator()(TIndices... extents) const {
    return tt::full(fill_value, extents...);
  }
};

} // namespace detail
} // namespace operators
} // namespace tt
