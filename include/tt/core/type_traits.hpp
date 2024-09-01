#pragma once

#include <experimental/mdspan>

#include <concepts>

namespace tt::inline core {

using std::experimental::dims;

template <class T>
inline constexpr bool is_arithmetic_v = std::is_arithmetic_v<T>;

template <class T, template <class...> class F>
constexpr bool is_instantiation_of_typenames() {
  constexpr auto deduce = []<class... Ts>(F<Ts...> *) {};
  return requires { deduce(static_cast<T *>(nullptr)); };
}

template <class T, template <class, auto...> class F>
constexpr bool is_instantiation_of_typename_values() {
  constexpr auto deduce = []<class U, auto... Vs>(F<U, Vs...> *) {};
  return requires { deduce(static_cast<T *>(nullptr)); };
}

template <class T, template <class, auto...> class F>
constexpr auto instantiation_for_typename_values() {
  constexpr auto deduce =
      []<class U, auto... Vs>(
          F<U, Vs...> *) -> std::type_identity<F<U, Vs...>> { return {}; };

  return deduce(static_cast<T *>(nullptr));
}

template <class T>
using element_type_t = T::element_type;

template <class... Ts>
using common_element_type_t = std::common_type_t<tt::element_type_t<Ts>...>;

template <class T>
using extents_type_t = T::extents_type;

template <class T>
using index_type_t = T::index_type;

template <class T>
using layout_type_t = T::layout_type;

template <class T, class TExtents = tt::extents_type_t<T>>
using mapping_type_t = tt::layout_type_t<T>::template mapping<TExtents>;

template <class T>
using rank_type_t = T::rank_type;

template <class T>
using size_type_t = T::size_type;

template <class TIndex>
struct extent_from;

template <std::integral TIndex>
struct extent_from<TIndex> {
  static constexpr std::size_t value = std::dynamic_extent;
};

template <class T>
concept integral_constant_like =
    std::is_integral_v<decltype(T::value)> and
    not std::is_same_v<bool, std::remove_const_t<decltype(T::value)>> and
    std::convertible_to<T, decltype(T::value)> and
    std::equality_comparable_with<T, decltype(T::value)> and
    std::bool_constant<T() == T::value>::value and
    std::bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value;

template <tt::integral_constant_like TIndex>
struct extent_from<TIndex> {
  static constexpr std::size_t value = TIndex::value;
};

template <class... TIndices>
using extents_from =
    std::extents<std::size_t, tt::extent_from<TIndices>::value...>;

template <std::size_t N>
using size_constant = std::integral_constant<std::size_t, N>;

} // namespace tt::inline core
