#pragma once

#include <tt/core/dtype.hpp>
#include <tt/core/memory.hpp>
#include <tt/core/tensor.hpp>

#include <cassert>

namespace tt {
inline namespace operators {

template <class TExtents>
struct reshape_view {
private:
  TExtents extents;

public:
  constexpr reshape_view() noexcept = default;
  constexpr reshape_view(const TExtents &extents) : extents{extents} {}

  template <class TInput, class = std::enable_if_t<tt::tensor<TInput>>>
  friend constexpr auto operator|(const TInput &input,
                                  const reshape_view &view) {
    using element_type = tt::element_type_t<TInput>;
    using mapping_type = tt::mapping_type_t<TInput, TExtents>;
    using layout_type = tt::layout_type_t<TInput>;
    using output_type = tt::Tensor<element_type, TExtents, layout_type>;

    const mapping_type mapping{view.extents};

    assert(mapping.required_span_size() <=
           input.mapping().required_span_size());

    return output_type{input.data_handle(), mapping};
  }
};

template <class... TIndices,
          class = std::enable_if_t<(... and tt::index<TIndices>)>>
constexpr auto reshape(TIndices... extents)
    -> tt::reshape_view<tt::extents_from<TIndices...>> {
  return extents_from<TIndices...>{extents...};
}

} // namespace operators
} // namespace tt
