#pragma once

#include <tt/core/dtype.hpp>
#include <tt/core/float.hpp>
#include <tt/core/layout.hpp>
#include <tt/core/memory.hpp>
#include <tt/core/tensor.hpp>

namespace tt {
inline namespace operators {

template <auto... Vs, class... TIndices>
constexpr auto empty(TIndices... extents) {
  using extents_type = tt::extents_from<TIndices...>;
  using element_type = tt::type_t<tt::dtypes, tt::dtype::Float32, Vs...>;
  using layout_type = tt::type_t<tt::layouts, tt::layout::RowMajor, Vs...>;

  const typename layout_type::template mapping<extents_type> mapping{
      extents_type{extents...}};
  const auto size = mapping.required_span_size();

  return tt::Tensor<element_type, extents_type, layout_type>{
      tt::make_shared_for_overwrite<element_type[]>(size), mapping};
}

} // namespace operators
} // namespace tt
