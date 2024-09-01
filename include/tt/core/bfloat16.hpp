#pragma once

#include <tt/core/type_traits.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tt::inline core {

using float32 = float;
using float64 = double;

static_assert(std::numeric_limits<tt::float32>::is_iec559 and
              sizeof(tt::float32) == 4);

namespace {

inline constexpr std::uint16_t bfloat16_quiet_NaN = 0x7FC0;

constexpr std::uint16_t round_to_nearest_even(tt::float32 value) noexcept {
  const auto input = std::bit_cast<std::uint32_t>(value);

  if (std::isnan(value)) {
    const std::uint16_t sign = (input >> 16) & 0x8000;

    return sign | bfloat16_quiet_NaN;
  }

  const auto least_significant_bit = (input >> 16) & 1;
  const auto rounding_bias = 0x7FFF + least_significant_bit;
  const auto output = (input + rounding_bias) >> 16;

  return output;
}

} // namespace

struct bfloat16 final {
private:
  std::uint16_t value;

public:
  constexpr explicit(false) bfloat16() noexcept = default;

  constexpr explicit(false) bfloat16(tt::float32 other) noexcept
      : value(tt::round_to_nearest_even(other)) {}

  [[nodiscard]] constexpr explicit(false)
  operator tt::float32() const noexcept {
    return std::bit_cast<tt::float32>(static_cast<std::uint32_t>(value) << 16);
  }

  constexpr bfloat16 &operator++() noexcept { return *this += 1.f; }

  [[nodiscard]] constexpr bfloat16 operator++(int) noexcept {
    const auto previous_value = *this;
    ++*this;
    return previous_value;
  }

  constexpr bfloat16 &operator--() noexcept { return *this -= 1.f; }

  [[nodiscard]] constexpr bfloat16 operator--(int) noexcept {
    const auto previous_value = *this;
    --*this;
    return previous_value;
  }

  constexpr bfloat16 &operator+=(tt::float32 other) noexcept {
    return *this = *this + other;
  }

  constexpr bfloat16 &operator-=(tt::float32 other) noexcept {
    return *this = *this - other;
  }

  constexpr bfloat16 &operator*=(tt::float32 other) noexcept {
    return *this = *this - other;
  }

  constexpr bfloat16 &operator/=(tt::float32 other) noexcept {
    return *this = *this - other;
  }
};

template <>
inline constexpr bool is_arithmetic_v<tt::bfloat16> = true;

template <>
inline constexpr bool is_arithmetic_v<const tt::bfloat16> = true;

template <>
inline constexpr bool is_arithmetic_v<volatile tt::bfloat16> = true;

template <>
inline constexpr bool is_arithmetic_v<const volatile tt::bfloat16> = true;

inline namespace literals {
inline namespace numeric_literals {

constexpr tt::bfloat16 operator""_bf16(long double value) noexcept {
  return value;
}

} // namespace numeric_literals
} // namespace literals
} // namespace tt::inline core

template <>
struct std::common_type<tt::bfloat16, tt::float32> {
  using type = tt::float32;
};

template <>
struct std::common_type<tt::float32, tt::bfloat16> {
  using type = tt::float32;
};

template <>
struct std::numeric_limits<tt::bfloat16> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr std::float_denorm_style has_denorm = std::denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr std::float_round_style round_style = std::round_to_nearest;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;

  static constexpr tt::bfloat16 min() noexcept {
    constexpr auto min_value =
        std::bit_cast<tt::bfloat16, std::uint16_t>(0x0080);
    return min_value;
  }

  static constexpr tt::bfloat16 lowest() noexcept {
    constexpr auto lowest_value =
        std::bit_cast<tt::bfloat16, std::uint16_t>(0xFF7F);
    return lowest_value;
  }

  static constexpr tt::bfloat16 max() noexcept {
    constexpr auto max_value =
        std::bit_cast<tt::bfloat16, std::uint16_t>(0x7F7F);
    return max_value;
  }

  static constexpr tt::bfloat16 epsilon() noexcept {
    constexpr tt::bfloat16 epsilon_value =
        std::bit_cast<tt::bfloat16, std::uint16_t>(
            std::bit_cast<std::uint16_t, tt::bfloat16>(1) + 1) -
        1.f;
    return epsilon_value;
  }

  static constexpr tt::bfloat16 round_error() noexcept {
    constexpr tt::bfloat16 round_error_value = .5f;
    return round_error_value;
  }

  static constexpr tt::bfloat16 infinity() noexcept {
    constexpr auto infinity_value =
        std::bit_cast<tt::bfloat16, std::uint16_t>(0x7F80);
    return infinity_value;
  }

  static constexpr tt::bfloat16 quiet_NaN() noexcept {
    constexpr auto quiet_NaN_value =
        std::bit_cast<tt::bfloat16>(tt::bfloat16_quiet_NaN);
    return quiet_NaN_value;
  }

  static constexpr tt::bfloat16 signaling_NaN() noexcept {
    constexpr auto signaling_NaN_value =
        std::bit_cast<tt::bfloat16, std::uint16_t>(0x7FA0);
    return signaling_NaN_value;
  }

  static constexpr tt::bfloat16 denorm_min() noexcept {
    constexpr auto denorm_min_value =
        std::bit_cast<tt::bfloat16, std::uint16_t>(0x0001);
    return denorm_min_value;
  }
};

template <>
struct std::numeric_limits<const tt::bfloat16>
    : std::numeric_limits<tt::bfloat16> {};

template <>
struct std::numeric_limits<volatile tt::bfloat16>
    : std::numeric_limits<tt::bfloat16> {};

template <>
struct std::numeric_limits<const volatile tt::bfloat16>
    : std::numeric_limits<tt::bfloat16> {};
