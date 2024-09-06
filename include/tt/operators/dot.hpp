#pragma once

#include <tt/core/concepts.hpp>

namespace tt {
inline namespace operators {

template <class TLhs, class TRhs>
TT_CONCEPT has_dot_product =
    tt::vector<TLhs> and tt::vector<TRhs> and
    tt::common_extent_with<TLhs::static_extent(0), TRhs::static_extent(0)>;

template <class TLhs, class TRhs,
          class = TT_REQUIRES(tt::has_dot_product<TLhs, TRhs>)>
using dot_product_result_t = tt::common_element_type_t<TLhs, TRhs>;

struct dot_fn {
  template <class TLhs, class TRhs>
  constexpr tt::dot_product_result_t<TLhs, TRhs>
  operator()(const TLhs &lhs, const TRhs &rhs) const {
    assert(lhs.size() == rhs.size());

    using result_type = decltype(dot_fn{}(lhs, rhs));

    const auto size = lhs.size();
    result_type result{};

    for (std::size_t index = 0; index < size; ++index) {
      result += lhs[index] * rhs[index];
    }

    return result;
  }
};

inline constexpr tt::dot_fn dot{};

} // namespace operators
} // namespace tt
