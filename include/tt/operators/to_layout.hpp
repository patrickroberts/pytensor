#pragma once

#include <tt/core/layout.hpp>
#include <tt/core/memory.hpp>
#include <tt/core/tensor.hpp>

namespace tt {
inline namespace operators {

template <class TLayout>
struct to_layout_view {};

template <class TInput, class TLayout,
          class = std::enable_if_t<tt::tensor<TInput>>>
constexpr auto operator|(const TInput &input, const to_layout_view<TLayout> &) {
  using element_type = tt::element_type_t<TInput>;
  using extents_type = tt::extents_type_t<TInput>;
  using mapping_type = typename TLayout::template mapping<extents_type>;
  using output_type = tt::Tensor<element_type, extents_type, TLayout>;
  using index_type = tt::index_type_t<output_type>;

  const mapping_type mapping{input.extents()};
  const auto count = mapping.required_span_size();
  const output_type output{
      mapping.is_exhaustive()
          ? tt::make_shared_for_overwrite<element_type[]>(count)
          : tt::make_shared<element_type[]>(count),
      mapping};

  const auto recur = [&](const auto &recur, auto... indices) {
    constexpr auto rank = sizeof...(indices);

    if constexpr (rank == output_type::rank()) {
      output(indices...) = input(indices...);
    } else {
      for (index_type index = 0; index < output.extent(rank); ++index) {
        recur(recur, indices..., index);
      }
    }
  };

  recur(recur);

  return output;
}

template <tt::layout Layout, class TLayout = tt::type_t<tt::layouts, Layout>>
constexpr auto to_layout() -> tt::to_layout_view<TLayout> {
  return {};
}

template <tt::layout Layout, class TInput,
          class = std::enable_if_t<tt::tensor<TInput>>>
constexpr auto to_layout(const TInput &input) {
  return input | to_layout<Layout>();
}

constexpr auto to_row_major() { return to_layout<tt::layout::RowMajor>(); }

template <class TInput, class = std::enable_if_t<tt::tensor<TInput>>>
constexpr auto to_row_major(const TInput &input) {
  return input | to_row_major();
}

constexpr auto to_tiled() { return to_layout<tt::layout::Tiled>(); }

template <class TInput, class = std::enable_if_t<tt::tensor<TInput>>>
constexpr auto to_tiled(const TInput &input) {
  return input | to_tiled();
}

} // namespace operators
} // namespace tt
