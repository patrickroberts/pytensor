#pragma once

#include <tt/core/tensor.hpp>

#include <cassert>

namespace tt {
inline namespace operators {

template <class TExtents, class = TT_REQUIRES(tt::extents<TExtents>)>
struct reshape_view {
private:
  TExtents extents;

public:
  constexpr reshape_view(const TExtents &extents) : extents{extents} {}

  template <class TInput, class = TT_REQUIRES(tt::tensor<TInput>)>
  friend constexpr tt::Tensor<tt::element_type_t<TInput>, TExtents,
                              tt::layout_type_t<TInput>>
  operator|(const TInput &input, const reshape_view &view) {
    const tt::mapping_type_t<TInput, TExtents> mapping{view.extents};
    assert(mapping.required_span_size() <=
           input.mapping().required_span_size());

    using explicit_t = decltype(input | view);
    return explicit_t{input.data_handle(), mapping};
  }
};

struct reshape_fn {
  template <class TExtents, class = TT_REQUIRES(tt::extents<TExtents>)>
  constexpr tt::reshape_view<TExtents>
  operator()(const TExtents &extents) const {
    return extents;
  }

  template <class... TIndices>
  constexpr tt::reshape_view<tt::extents_from<TIndices...>>
  operator()(TIndices... extents) const {
    return extents_from<TIndices...>{extents...};
  }

  template <class TInput, class TExtents,
            class = TT_REQUIRES(tt::tensor<TInput> and tt::extents<TExtents>)>
  constexpr tt::Tensor<tt::element_type_t<TInput>, TExtents,
                       tt::layout_type_t<TInput>>
  operator()(const TInput &input, const TExtents &extents) const {
    return input | extents;
  }

  template <class TInput, class... TIndices,
            class = TT_REQUIRES(tt::tensor<TInput>)>
  constexpr tt::Tensor<tt::element_type_t<TInput>,
                       tt::extents_from<TIndices...>, tt::layout_type_t<TInput>>
  operator()(const TInput &input, TIndices... extents) const {
    return input | tt::reshape_fn{}(extents...);
  }
};

inline constexpr tt::reshape_fn reshape{};

} // namespace operators
} // namespace tt
