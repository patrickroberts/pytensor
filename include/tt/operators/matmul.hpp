#pragma once

#include <tt/operators/dot.hpp>
#include <tt/operators/empty.hpp>

namespace tt::inline operators {

template <class TLhs, class TRhs>
concept has_matrix_product =
    tt::matrix<TLhs> and tt::matrix<TRhs> and
    tt::common_element_type_with<TLhs, TRhs> and
    tt::common_extent_with<TLhs::static_extent(1), TRhs::static_extent(0)>;

template <class TLhs, class TRhs>
  requires tt::has_matrix_product<TLhs, TRhs>
using matrix_product_result_t = tt::row_major_matrix<
    tt::common_element_type_t<TLhs, TRhs>,
    std::extents<std::size_t, TLhs::static_extent(0), TRhs::static_extent(1)>>;

struct matmul_fn {
private:
  template <std::size_t Index, tt::tensor T>
  static constexpr tt::index auto get_extent(const T &t) noexcept {
    static_assert(Index < T::rank());

    if constexpr (T::static_extent(Index) == std::dynamic_extent) {
      return t.extent(Index);
    } else {
      return tt::size_constant<T::static_extent(Index)>{};
    }
  }

public:
  template <class TLhs, class TRhs>
  constexpr tt::matrix_product_result_t<TLhs, TRhs>
  operator()(const TLhs &lhs, const TRhs &rhs) const {
    assert(lhs.extent(1) == rhs.extent(0));

    using result_type = decltype(matmul_fn{}(lhs, rhs));

    const auto rows = matmul_fn::get_extent<0>(lhs);
    const auto cols = matmul_fn::get_extent<1>(rhs);
    const auto result = tt::empty<tt::element_type_t<result_type>>(rows, cols);

    for (std::size_t col = 0; col < cols; ++col) {
      const auto rhs_col = std::submdspan(rhs, std::full_extent, col);

      for (std::size_t row = 0; row < rows; ++row) {
        const auto lhs_row = std::submdspan(lhs, row, std::full_extent);

        result(row, col) = tt::dot(lhs_row, rhs_col);
      }
    }

    return result;
  }
};

inline constexpr tt::matmul_fn matmul{};

} // namespace tt::inline operators
