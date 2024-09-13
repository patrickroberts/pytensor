#pragma once

#include <tt/core/dtype.hpp>
#include <tt/operators/empty.hpp>

namespace tt {
inline namespace operators {

template <auto... Vs, class TStart, class TEnd, class TStep>
constexpr auto arange(TStart start, TEnd end, TStep step) {
  using common_type = std::common_type_t<TStart, TEnd, TStep>;
  using element_type =
      tt::type_t<tt::dtypes, tt::value_v<tt::dtypes, common_type>, Vs...>;

  constexpr auto dtype = tt::value_v<tt::dtypes, element_type>;

  const std::size_t size =
      static_cast<element_type>(end - start - 1) / step + 1;
  const auto result = tt::empty<dtype>(size);

  for (std::size_t index = 0; index < size; ++index) {
    result[index] = start + index * step;
  }

  return result;
}

template <auto... Vs, class TStart, class TEnd>
constexpr auto arange(TStart start, TEnd end) {
  constexpr std::common_type_t<TStart, TEnd> step{1};
  return arange<Vs...>(start, end, step);
}

template <auto... Vs, class TEnd>
constexpr auto arange(TEnd end) {
  constexpr TEnd start{0};
  return arange<Vs...>(start, end);
}

} // namespace operators
} // namespace tt
