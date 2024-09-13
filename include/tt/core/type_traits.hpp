#pragma once

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
struct extent_from<TIndex, std::enable_if_t<std::is_integral_v<TIndex>>> {
  static constexpr std::size_t value = std::dynamic_extent;
};

template <class T, class = void>
inline constexpr bool is_integral_constant_like_v = false;

template <class T>
inline constexpr bool is_integral_constant_like_v<
    T, std::enable_if_t<
           std::is_integral_v<decltype(T::value)> and
           not std::is_same_v<bool, std::remove_const_t<decltype(T::value)>> and
           std::is_convertible_v<T, decltype(T::value)> and
           std::bool_constant<T() == T::value>::value and
           std::bool_constant<static_cast<decltype(T::value)>(T()) ==
                              T::value>::value>> = true;

template <class TIndex>
struct extent_from<TIndex,
                   std::enable_if_t<tt::is_integral_constant_like_v<TIndex>>> {
  static constexpr std::size_t value = TIndex::value;
};

template <class... TIndices>
using extents_from =
    std::extents<std::size_t, tt::extent_from<TIndices>::value...>;

template <auto N>
using constant = std::integral_constant<decltype(N), N>;

template <std::size_t N>
using size_constant = constant<N>;

template <auto Default, auto... Vs>
struct value_or : value_or<Default, Vs>... {};

template <auto Default, auto V>
struct value_or<Default, V> {};

template <auto Default, decltype(Default) V>
struct value_or<Default, V> {
  static constexpr auto value = V;
};

namespace detail {

template <auto Default, class = void, auto... Vs>
inline constexpr auto value_or_v = Default;

template <auto Default, auto... Vs>
inline constexpr auto value_or_v<
    Default, std::void_t<decltype(tt::value_or<Default, Vs...>::value)>,
    Vs...> = tt::value_or<Default, Vs...>::value;

template <class T, template <class, auto> class Traits, auto V>
constexpr auto as_value(Traits<T, V>) noexcept {
  return V;
}

template <auto V, template <class, auto> class Traits, class T>
constexpr auto as_type(Traits<T, V>) noexcept -> T;

} // namespace detail

template <auto Default, auto... Vs>
inline constexpr auto value_or_v = detail::value_or_v<Default, void, Vs...>;

template <class Traits, class T>
inline constexpr auto value_v =
    detail::as_value<T, Traits::template fn>(Traits{});

template <class Traits, auto Default, auto... Vs>
using type_t = decltype(detail::as_type<tt::value_or_v<Default, Vs...>,
                                        Traits::template fn>(Traits{}));

} // namespace core
} // namespace tt
