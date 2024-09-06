#pragma once

#include <tt/core/tensor.hpp>
#include <tt/operators/detail/delete.hpp>

#include <algorithm>

namespace tt {
inline namespace operators {

struct full_fn {
  template <class T, class... TIndices, class = TT_REQUIRES(tt::arithmetic<T>)>
  constexpr tt::RowMajorTensor<T, tt::extents_from<TIndices...>>
  operator()(T fill_value, TIndices... extents) const {
    const std::size_t size = (1 * ... * extents);
    auto allocator = std::allocator<T>{};
    auto data_pointer = allocator.allocate(size);

    std::fill_n(data_pointer, size, fill_value);

    auto data_handle = std::shared_ptr<T[]>{
        data_pointer, detail::allocator_delete{allocator, size}};

    using explicit_t = decltype(operator()(fill_value, extents...));
    return explicit_t{std::move(data_handle), extents...};
  }
};

inline constexpr tt::full_fn full{};

} // namespace operators
} // namespace tt
