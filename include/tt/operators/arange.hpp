#pragma once

#include <tt/operators/empty.hpp>

namespace tt {
inline namespace operators {

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
struct arange_fn {
  template <class TStep, class = TT_REQUIRES(tt::arithmetic<TStep>)>
  constexpr tt::RowMajorVector<T> operator()(T start, T end, TStep step) const {
    const std::size_t size = static_cast<T>(end - start - 1) / step + 1;
    const auto result = tt::empty<T>(size);

    for (std::size_t index = 0; index < size; ++index) {
      result[index] = start + index * step;
    }

    return result;
  }

  constexpr tt::RowMajorVector<T> operator()(T start, T end) const {
    constexpr T step{1};
    return arange_fn{}(start, end, step);
  }

  constexpr tt::RowMajorVector<T> operator()(T end) const {
    constexpr T start{0};
    return arange_fn{}(start, end);
  }
};

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
inline constexpr tt::arange_fn<T> arange{};

} // namespace operators
} // namespace tt
