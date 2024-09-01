#pragma once

#include <tt/core/type_traits.hpp>

namespace tt::inline core {

template <class T>
concept arithmetic = tt::is_arithmetic_v<T>;

template <class T1, class T2>
concept common_element_type_with =
    std::common_with<tt::element_type_t<T1>, tt::element_type_t<T2>>;

template <std::size_t N1, std::size_t N2>
concept common_extent_with =
    N1 == N2 or N1 == std::dynamic_extent or N2 == std::dynamic_extent;

template <std::size_t N1, std::size_t N2>
  requires common_extent_with<N1, N2>
inline constexpr std::size_t common_extent_v =
    N1 == N2 ? N1 : std::dynamic_extent;

template <class T>
concept extents = tt::is_instantiation_of_typename_values<T, std::extents>();

template <class T, std::size_t Rank>
concept has_rank = T::rank() == Rank;

template <class T>
concept index = std::integral<T> or tt::integral_constant_like<T>;

template <class M, class L>
concept mapping = std::same_as<tt::layout_type_t<M>, L> and
                  tt::extents<tt::extents_type_t<M>>;

template <class L, class E>
concept layout = tt::mapping<typename L::template mapping<E>, L>;

template <class T>
concept tensor = tt::is_instantiation_of_typenames<T, std::mdspan>();

template <class T>
concept matrix = tt::tensor<T> and tt::has_rank<T, 2>;

template <class T>
concept vector = tt::tensor<T> and tt::has_rank<T, 1>;

} // namespace tt::inline core
