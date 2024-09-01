#pragma once

#include <tt/core/shared_tensor.hpp>
#include <tt/core/tiled_layout.hpp>

namespace tt::inline operators {

template <class TLayout>
struct to_layout_view {
  template <tt::tensor TInput>
  friend constexpr tt::shared_tensor<tt::element_type_t<TInput>,
                                     tt::extents_type_t<TInput>, TLayout>
  operator|(const TInput &input, to_layout_view to_layout) {
    using output_type = decltype(input | to_layout);
    using element_type = tt::element_type_t<output_type>;
    using extents_type = tt::extents_type_t<output_type>;
    using index_type = tt::index_type_t<output_type>;
    using mapping_type = TLayout::template mapping<extents_type>;

    const mapping_type mapping{input.extents()};
    const auto required_span_size = mapping.required_span_size();
    const output_type output{
        mapping.is_exhaustive()
            ? std::make_shared_for_overwrite<element_type[]>(required_span_size)
            : std::make_shared<element_type[]>(required_span_size),
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
};

template <class TLayout>
struct to_layout_fn {
  constexpr tt::to_layout_view<TLayout> operator()() const { return {}; }

  template <tt::tensor TInput>
  constexpr tt::shared_tensor<tt::element_type_t<TInput>,
                              tt::extents_type_t<TInput>, TLayout>
  operator()(const TInput &input) const {
    return input | to_layout_fn{}();
  }
};

template <class TLayout>
inline constexpr tt::to_layout_fn<TLayout> to_layout{};

inline constexpr tt::to_layout_fn<tt::tiled_layout> to_tiled{};

inline constexpr tt::to_layout_fn<tt::row_major_layout> to_row_major{};

} // namespace tt::inline operators
