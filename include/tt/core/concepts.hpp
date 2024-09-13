#pragma once

#include <tt/core/type_traits.hpp>

namespace tt {
inline namespace core {

template <class T>
inline constexpr bool arithmetic = tt::is_arithmetic_v<T>;

template <std::size_t N1, std::size_t N2>
inline constexpr bool common_extent_with =
    N1 == N2 or N1 == std::dynamic_extent or N2 == std::dynamic_extent;

template <std::size_t N1, std::size_t N2,
          class = std::enable_if_t<common_extent_with<N1, N2>>>
inline constexpr std::size_t common_extent_v =
    N1 == N2 ? N1 : std::dynamic_extent;

template <class T>
inline constexpr bool extents = tt::is_extents_v<T>;

template <class T, std::size_t Rank, class = void>
inline constexpr bool has_rank = false;

template <class T, std::size_t Rank>
inline constexpr bool has_rank<T, Rank, std::enable_if_t<T::rank() == Rank>> =
    true;

template <class T>
inline constexpr bool index =
    std::is_integral_v<T> or tt::is_integral_constant_like_v<T>;

template <class M, class L, class = void>
inline constexpr bool mapping = false;

template <class M, class L>
inline constexpr bool
    mapping<M, L,
            std::enable_if_t<std::is_same_v<tt::layout_type_t<M>, L> and
                             tt::extents<tt::extents_type_t<M>>>> = true;

template <class T>
inline constexpr bool tensor = tt::is_tensor_v<T>;

template <class T>
inline constexpr bool matrix = tt::tensor<T> and tt::has_rank<T, 2>;

template <class T>
inline constexpr bool vector = tt::tensor<T> and tt::has_rank<T, 1>;

} // namespace core
} // namespace tt
