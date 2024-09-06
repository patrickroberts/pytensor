#pragma once

#include <tt/core/layout.hpp>
#include <tt/core/tensor.hpp>
#include <tt/operators/detail/delete.hpp>

namespace tt {
inline namespace operators {

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
struct empty_fn {
  template <class... TIndices>
  constexpr tt::RowMajorTensor<T, tt::extents_from<TIndices...>>
  operator()(TIndices... extents) const {
    using extents_type = tt::extents_from<TIndices...>;

    const tt::RowMajor::template mapping<extents_type> mapping{
        extents_type{extents...}};
    const auto size = mapping.required_span_size();
    auto allocator = std::allocator<T>{};
    auto data_handle = std::shared_ptr<T[]>{
        allocator.allocate(size), detail::allocator_delete{allocator, size}};

    using explicit_t = decltype(operator()(extents...));
    return explicit_t{std::move(data_handle), mapping};
  }
};

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
inline constexpr empty_fn<T> empty{};

} // namespace operators
} // namespace tt
