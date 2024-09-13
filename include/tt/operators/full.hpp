#pragma once

#include <tt/core/dtype.hpp>
#include <tt/core/memory.hpp>
#include <tt/core/tensor.hpp>

namespace tt {
inline namespace operators {

template <auto... Vs, class T, class... TIndices>
constexpr auto full(T fill_value, TIndices... extents) {
  constexpr auto default_dtype = tt::value_v<tt::dtypes, T>;
  using element_type = tt::type_t<tt::dtypes, default_dtype, Vs...>;
  using extents_type = tt::extents_from<TIndices...>;
  using layout_type = tt::type_t<tt::layouts, tt::layout::RowMajor, Vs...>;

  const std::size_t size = (1 * ... * extents);

  return tt::Tensor<element_type, extents_type, layout_type>{
      tt::make_shared<element_type[]>(size, fill_value), extents...};
}

} // namespace operators
} // namespace tt
