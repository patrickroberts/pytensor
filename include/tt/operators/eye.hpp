#pragma once

#include <tt/operators/zeros.hpp>

namespace tt {
inline namespace operators {

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
struct eye_fn {
private:
  static constexpr T one{1};

public:
  template <class TRows, class TCols>
  constexpr auto operator()(TRows rows, TCols cols) const
      -> tt::RowMajorMatrix<T, tt::extents_from<TRows, TCols>> {
    const auto result = zeros<T>(rows, cols);
    const auto diagonal_size = std::min<std::size_t>(rows, cols);

    for (std::size_t index = 0; index < diagonal_size; ++index) {
      result(index, index) = one;
    }

    return result;
  }

  template <class TIndex>
  constexpr auto operator()(TIndex extent) const
      -> tt::RowMajorMatrix<T, tt::extents_from<TIndex, TIndex>> {
    return eye_fn{}(extent, extent);
  }
};

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
inline constexpr tt::eye_fn<T> eye{};

} // namespace operators
} // namespace tt
