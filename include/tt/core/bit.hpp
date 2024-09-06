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

template <class TTo, class TFrom,
          class = TT_REQUIRES(std::is_trivially_copyable_v<TTo> and
                              std::is_trivially_copyable_v<TFrom> and
                              sizeof(TTo) == sizeof(TFrom))>
[[nodiscard]] constexpr TTo bit_cast(const TFrom &from) noexcept {
  return __builtin_bit_cast(TTo, from);
}

template <class T,
          class = TT_REQUIRES(std::is_unsigned_v<T> and std::is_integral_v<T>)>
constexpr bool has_single_bit(T value) noexcept {
  return __builtin_popcountll(value) == 1;
}

} // namespace core
} // namespace tt

#endif
