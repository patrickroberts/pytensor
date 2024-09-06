#pragma once

#include <tt/core/type_traits.hpp>

namespace tt {
inline namespace core {

template <class T>
TT_CONCEPT arithmetic = tt::is_arithmetic_v<T>;

template <std::size_t N1, std::size_t N2>
TT_CONCEPT common_extent_with =
    N1 == N2 or N1 == std::dynamic_extent or N2 == std::dynamic_extent;

template <std::size_t N1, std::size_t N2,
          class = TT_REQUIRES(common_extent_with<N1, N2>)>
inline constexpr std::size_t common_extent_v =
    N1 == N2 ? N1 : std::dynamic_extent;

template <class T>
TT_CONCEPT extents = tt::is_extents_v<T>;

template <class T, std::size_t Rank>
TT_CONCEPT has_rank = T::rank() == Rank;

template <class T>
TT_CONCEPT index = std::is_integral_v<T> or tt::is_integral_constant_like_v<T>;

template <class M, class L>
TT_CONCEPT mapping = std::is_same_v<tt::layout_type_t<M>, L> and
                     tt::extents<tt::extents_type_t<M>>;

template <class T>
TT_CONCEPT tensor = tt::is_tensor_v<T>;

template <class T>
TT_CONCEPT matrix = tt::tensor<T> and tt::has_rank<T, 2>;

template <class T>
TT_CONCEPT vector = tt::tensor<T> and tt::has_rank<T, 1>;

} // namespace core
} // namespace tt
