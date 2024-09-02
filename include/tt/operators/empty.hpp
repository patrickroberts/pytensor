#pragma once

#include <tt/core/layout.hpp>
#include <tt/core/tensor.hpp>

namespace tt::inline operators {

template <tt::arithmetic T>
struct empty_fn {
  template <class... TIndices>
  constexpr tt::RowMajorTensor<T, tt::extents_from<TIndices...>>
  operator()(TIndices... extents) const {
    using extents_type = tt::extents_from<TIndices...>;

    const tt::RowMajor::template mapping<extents_type> mapping{
        extents_type{extents...}};
    auto data_handle =
        std::make_shared_for_overwrite<T[]>(mapping.required_span_size());

    using explicit_t = decltype(operator()(extents...));
    return explicit_t{std::move(data_handle), mapping};
  }
};

template <tt::arithmetic T>
inline constexpr empty_fn<T> empty{};

} // namespace tt::inline operators
