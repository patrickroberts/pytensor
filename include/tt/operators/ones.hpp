#pragma once

#include <tt/operators/full.hpp>

namespace tt {
inline namespace operators {

template <auto... Vs, class... TIndices>
constexpr auto ones(TIndices... extents) {
  return tt::full<Vs...>(1.f, extents...);
}

} // namespace operators
} // namespace tt
