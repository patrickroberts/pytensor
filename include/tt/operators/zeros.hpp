#pragma once

#include <tt/operators/full.hpp>

namespace tt {
inline namespace operators {

template <auto... Vs, class... TIndices>
constexpr auto zeros(TIndices... extents) {
  return tt::full<Vs...>(0.f, extents...);
}

} // namespace operators
} // namespace tt
