#pragma once

#include <tt/core/preprocessor.hpp>

#include <experimental/mdspan>

namespace tt {
inline namespace core {

using std::experimental::dims;

template <class T>
inline constexpr bool is_arithmetic_v = std::is_arithmetic_v<T>;

template <class T>
inline constexpr bool is_extents_v = false;

template <class TIndex, auto... Extents>
inline constexpr bool is_extents_v<std::extents<TIndex, Extents...>> = true;

template <class T>
inline constexpr bool is_tensor_v = false;

template <class TElement, class TExtents, class TLayout, class TAccessor>
inline constexpr bool
    is_tensor_v<std::mdspan<TElement, TExtents, TLayout, TAccessor>> = true;

template <class T>
using element_type_t = typename T::element_type;

template <class... Ts>
using common_element_type_t = std::common_type_t<tt::element_type_t<Ts>...>;

template <class T>
using extents_type_t = typename T::extents_type;

template <class T>
using index_type_t = typename T::index_type;

template <class T>
using layout_type_t = typename T::layout_type;

template <class T, class TExtents = tt::extents_type_t<T>>
using mapping_type_t =
    typename tt::layout_type_t<T>::template mapping<TExtents>;

template <class T>
using rank_type_t = typename T::rank_type;

template <class T>
using size_type_t = typename T::size_type;

template <class TIndex, class = void>
struct extent_from;

template <class TIndex>
struct extent_from<TIndex, TT_REQUIRES(std::is_integral_v<TIndex>)> {
  static constexpr std::size_t value = std::dynamic_extent;
};

template <class T, class = void>
inline constexpr bool is_integral_constant_like_v = false;

template <class T>
inline constexpr bool is_integral_constant_like_v<
    T, TT_REQUIRES(
           std::is_integral_v<decltype(T::value)> and
           not std::is_same_v<bool, std::remove_const_t<decltype(T::value)>> and
           std::is_convertible_v<T, decltype(T::value)> and
           std::bool_constant<T() == T::value>::value and
           std::bool_constant<static_cast<decltype(T::value)>(T()) ==
                              T::value>::value)> = true;

template <class TIndex>
struct extent_from<TIndex,
                   TT_REQUIRES(tt::is_integral_constant_like_v<TIndex>)> {
  static constexpr std::size_t value = TIndex::value;
};

template <class... TIndices>
using extents_from =
    std::extents<std::size_t, tt::extent_from<TIndices>::value...>;

template <std::size_t N>
using size_constant = std::integral_constant<std::size_t, N>;

} // namespace core
} // namespace tt
