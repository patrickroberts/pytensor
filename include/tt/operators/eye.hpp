#pragma once

#include <tt/operators/zeros.hpp>

namespace tt::inline operators {

template <tt::arithmetic T>
struct eye_fn {
private:
  static constexpr T one{1};

public:
  template <class TRows, class TCols>
  constexpr tt::row_major_matrix<T, tt::extents_from<TRows, TCols>>
  operator()(TRows rows, TCols cols) const {
    const auto result = zeros<T>(rows, cols);
    const auto diagonal_size = std::min<std::size_t>(rows, cols);

    for (std::size_t index = 0; index < diagonal_size; ++index) {
      result(index, index) = one;
    }

    return result;
  }

  template <class TIndex>
  constexpr tt::row_major_matrix<T, tt::extents_from<TIndex, TIndex>>
  operator()(TIndex extent) const {
    return eye_fn{}(extent, extent);
  }
};

template <tt::arithmetic T>
inline constexpr tt::eye_fn<T> eye{};

} // namespace tt::inline operators
