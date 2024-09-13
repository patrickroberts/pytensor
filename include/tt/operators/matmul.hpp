#pragma once

#include <tt/operators/dot.hpp>
#include <tt/operators/empty.hpp>

namespace tt {
inline namespace operators {
namespace detail {
namespace {

template <std::size_t Index, class T, class = std::enable_if_t<tt::tensor<T>>>
constexpr auto get_extent(const T &t) noexcept {
  static_assert(Index < T::rank());

  if constexpr (T::static_extent(Index) == std::dynamic_extent) {
    return t.extent(Index);
  } else {
    return tt::size_constant<T::static_extent(Index)>{};
  }
}

} // namespace
} // namespace detail

template <class TLhs, class TRhs, class = void>
inline constexpr bool has_matrix_product = false;

template <class TLhs, class TRhs>
inline constexpr bool has_matrix_product<
    TLhs, TRhs,
    std::enable_if_t<tt::matrix<TLhs> and tt::matrix<TRhs> and
                     tt::common_extent_with<TLhs::static_extent(1),
                                            TRhs::static_extent(0)>>> = true;

template <auto... Vs, class TLhs, class TRhs,
          class = std::enable_if_t<tt::has_matrix_product<TLhs, TRhs>>>
constexpr auto matmul(const TLhs &lhs, const TRhs &rhs) {
  assert(lhs.extent(1) == rhs.extent(0));

  constexpr auto common_dtype =
      tt::value_v<tt::dtypes, tt::common_element_type_t<TLhs, TRhs>>;
  using element_type = tt::type_t<tt::dtypes, common_dtype, Vs...>;
  constexpr auto dtype = tt::value_v<tt::dtypes, element_type>;

  const auto rows = detail::get_extent<0>(lhs);
  const auto cols = detail::get_extent<1>(rhs);
  const auto result = tt::empty<dtype>(rows, cols);

  for (std::size_t col = 0; col < cols; ++col) {
    const auto rhs_col = std::submdspan(rhs, std::full_extent, col);

    for (std::size_t row = 0; row < rows; ++row) {
      const auto lhs_row = std::submdspan(lhs, row, std::full_extent);

      result(row, col) = tt::dot(lhs_row, rhs_col);
    }
  }

  return result;
}

} // namespace operators
} // namespace tt
