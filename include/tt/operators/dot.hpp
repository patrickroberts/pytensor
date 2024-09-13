#pragma once

#include <tt/core/concepts.hpp>

namespace tt {
inline namespace operators {

template <class TLhs, class TRhs, class = void>
inline constexpr bool has_dot_product = false;

template <class TLhs, class TRhs>
inline constexpr bool has_dot_product<
    TLhs, TRhs,
    std::enable_if_t<tt::vector<TLhs> and tt::vector<TRhs> and
                     tt::common_extent_with<TLhs::static_extent(0),
                                            TRhs::static_extent(0)>>> = true;

template <class TLhs, class TRhs>
using dot_product_result_t =
    std::enable_if_t<tt::has_dot_product<TLhs, TRhs>,
                     tt::common_element_type_t<TLhs, TRhs>>;

template <class TLhs, class TRhs>
constexpr auto dot(const TLhs &lhs, const TRhs &rhs) {
  assert(lhs.size() == rhs.size());

  const auto size = lhs.size();
  tt::dot_product_result_t<TLhs, TRhs> result{};

  for (std::size_t index = 0; index < size; ++index) {
    result += lhs[index] * rhs[index];
  }

  return result;
}

} // namespace operators
} // namespace tt
