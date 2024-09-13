#pragma once

#include <tt/operators/zeros.hpp>

namespace tt {
inline namespace operators {

template <auto... Vs, class TRows, class TCols>
constexpr auto eye(TRows rows, TCols cols) {
  using element_type = tt::type_t<tt::dtypes, tt::dtype::Float32, Vs...>;

  constexpr element_type one{1};
  const auto result = zeros<Vs...>(rows, cols);
  const auto diagonal_size = std::min<std::size_t>(rows, cols);

  for (std::size_t index = 0; index < diagonal_size; ++index) {
    result(index, index) = one;
  }

  return result;
}

template <auto... Vs, class TIndex>
constexpr auto eye(TIndex extent) {
  return eye<Vs...>(extent, extent);
}

} // namespace operators
} // namespace tt
