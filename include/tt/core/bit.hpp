#pragma once

#if __cplusplus >= 202002L

#include <bit>

namespace tt {
inline namespace core {

using std::bit_cast;
using std::has_single_bit;

} // namespace core
} // namespace tt

#else

#include <tt/core/preprocessor.hpp>

#include <type_traits>

namespace tt {
inline namespace core {

template <class TTo, class TFrom>
[[nodiscard]] constexpr auto bit_cast(const TFrom &from) noexcept
    -> TT_REQUIRES(std::is_trivially_copyable_v<TTo>
                           and std::is_trivially_copyable_v<TFrom> and
                       sizeof(TTo) == sizeof(TFrom),
                   TTo) {
  return __builtin_bit_cast(TTo, from);
}

template <class T>
[[nodiscard]] constexpr auto has_single_bit(T value) noexcept
    -> TT_REQUIRES(std::is_unsigned_v<T> and std::is_integral_v<T>, bool) {
  return __builtin_popcountll(value) == 1;
}

} // namespace core
} // namespace tt

#endif
