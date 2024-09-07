#pragma once

#include <tt/core/layout.hpp>
#include <tt/core/memory.hpp>
#include <tt/core/tensor.hpp>

namespace tt {
inline namespace operators {

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
struct empty_fn {
  template <class... TIndices>
  constexpr auto operator()(TIndices... extents) const
      -> tt::RowMajorTensor<T, tt::extents_from<TIndices...>> {
    using extents_type = tt::extents_from<TIndices...>;

    const tt::RowMajor::template mapping<extents_type> mapping{
        extents_type{extents...}};
    const auto size = mapping.required_span_size();

    using explicit_t = decltype(operator()(extents...));
    return explicit_t{tt::make_shared_for_overwrite<T[]>(size), mapping};
  }
};

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
inline constexpr empty_fn<T> empty{};

} // namespace operators
} // namespace tt
