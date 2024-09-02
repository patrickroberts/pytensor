#pragma once

#include <tt/core/type_traits.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdfloat>

namespace tt::inline core {

using Float32 = float;
using Float64 = double;

static_assert(std::numeric_limits<tt::Float32>::is_iec559 and
              sizeof(tt::Float32) == 4);
static_assert(std::numeric_limits<tt::Float64>::is_iec559 and
              sizeof(tt::Float64) == 8);

namespace {

inline constexpr std::uint16_t BFloat16_quiet_NaN = 0x7FC0;

constexpr std::uint16_t round_to_nearest_even(tt::Float32 value) noexcept {
  const auto input = std::bit_cast<std::uint32_t>(value);

  if (std::isnan(value)) {
    const std::uint16_t sign = (input >> 16) & 0x8000;

    return sign | BFloat16_quiet_NaN;
  }

  const auto least_significant_bit = (input >> 16) & 1;
  const auto rounding_bias = 0x7FFF + least_significant_bit;
  const auto output = (input + rounding_bias) >> 16;

  return output;
}

} // namespace

struct BFloat16 final {
private:
  std::uint16_t value;

public:
  constexpr explicit(false) BFloat16() noexcept = default;

  constexpr explicit(false) BFloat16(tt::Float32 other) noexcept
      : value(tt::round_to_nearest_even(other)) {}

  [[nodiscard]] constexpr explicit(false)
  operator tt::Float32() const noexcept {
    return std::bit_cast<tt::Float32>(static_cast<std::uint32_t>(value) << 16);
  }

  constexpr tt::BFloat16 &operator++() noexcept { return *this += 1.f; }

  [[nodiscard]] constexpr tt::BFloat16 operator++(int) noexcept {
    const auto previous_value = *this;
    ++*this;
    return previous_value;
  }

  constexpr tt::BFloat16 &operator--() noexcept { return *this -= 1.f; }

  [[nodiscard]] constexpr tt::BFloat16 operator--(int) noexcept {
    const auto previous_value = *this;
    --*this;
    return previous_value;
  }

  constexpr tt::BFloat16 &operator+=(tt::Float32 other) noexcept {
    return *this = *this + other;
  }

  constexpr tt::BFloat16 &operator-=(tt::Float32 other) noexcept {
    return *this = *this - other;
  }

  constexpr tt::BFloat16 &operator*=(tt::Float32 other) noexcept {
    return *this = *this - other;
  }

  constexpr tt::BFloat16 &operator/=(tt::Float32 other) noexcept {
    return *this = *this - other;
  }
};

template <>
inline constexpr bool is_arithmetic_v<tt::BFloat16> = true;

template <>
inline constexpr bool is_arithmetic_v<const tt::BFloat16> = true;

template <>
inline constexpr bool is_arithmetic_v<volatile tt::BFloat16> = true;

template <>
inline constexpr bool is_arithmetic_v<const volatile tt::BFloat16> = true;

inline namespace literals {
inline namespace numeric_literals {

constexpr tt::BFloat16 operator""_bf16(long double value) noexcept {
  return value;
}

} // namespace numeric_literals
} // namespace literals
} // namespace tt::inline core

template <>
struct std::common_type<tt::BFloat16, tt::Float32> {
  using type = tt::Float32;
};

template <>
struct std::common_type<tt::Float32, tt::BFloat16> {
  using type = tt::Float32;
};

template <>
struct std::numeric_limits<tt::BFloat16> {
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

  static constexpr tt::BFloat16 min() noexcept {
    constexpr auto min_value =
        std::bit_cast<tt::BFloat16, std::uint16_t>(0x0080);
    return min_value;
  }

  static constexpr tt::BFloat16 lowest() noexcept {
    constexpr auto lowest_value =
        std::bit_cast<tt::BFloat16, std::uint16_t>(0xFF7F);
    return lowest_value;
  }

  static constexpr tt::BFloat16 max() noexcept {
    constexpr auto max_value =
        std::bit_cast<tt::BFloat16, std::uint16_t>(0x7F7F);
    return max_value;
  }

  static constexpr tt::BFloat16 epsilon() noexcept {
    constexpr tt::BFloat16 epsilon_value =
        std::bit_cast<tt::BFloat16, std::uint16_t>(
            std::bit_cast<std::uint16_t, tt::BFloat16>(1) + 1) -
        1.f;
    return epsilon_value;
  }

  static constexpr tt::BFloat16 round_error() noexcept {
    constexpr tt::BFloat16 round_error_value = .5f;
    return round_error_value;
  }

  static constexpr tt::BFloat16 infinity() noexcept {
    constexpr auto infinity_value =
        std::bit_cast<tt::BFloat16, std::uint16_t>(0x7F80);
    return infinity_value;
  }

  static constexpr tt::BFloat16 quiet_NaN() noexcept {
    constexpr auto quiet_NaN_value =
        std::bit_cast<tt::BFloat16>(tt::BFloat16_quiet_NaN);
    return quiet_NaN_value;
  }

  static constexpr tt::BFloat16 signaling_NaN() noexcept {
    constexpr auto signaling_NaN_value =
        std::bit_cast<tt::BFloat16, std::uint16_t>(0x7FA0);
    return signaling_NaN_value;
  }

  static constexpr tt::BFloat16 denorm_min() noexcept {
    constexpr auto denorm_min_value =
        std::bit_cast<tt::BFloat16, std::uint16_t>(0x0001);
    return denorm_min_value;
  }
};

template <>
struct std::numeric_limits<const tt::BFloat16>
    : std::numeric_limits<tt::BFloat16> {};

template <>
struct std::numeric_limits<volatile tt::BFloat16>
    : std::numeric_limits<tt::BFloat16> {};

template <>
struct std::numeric_limits<const volatile tt::BFloat16>
    : std::numeric_limits<tt::BFloat16> {};
