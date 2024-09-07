#pragma once

#if __cplusplus >= 202002L

#include <memory>

namespace tt {
inline namespace core {

using std::make_shared;
using std::make_shared_for_overwrite;

} // namespace core
} // namespace tt

#else

#include <tt/core/detail/delete.hpp>
#include <tt/core/preprocessor.hpp>

#include <memory>
#include <type_traits>

namespace tt {
inline namespace core {

template <class T>
constexpr auto make_shared(std::size_t count) noexcept
    -> TT_REQUIRES(std::is_array_v<T> and std::extent_v<T> == 0,
                   std::shared_ptr<T>) {
  auto alloc = std::allocator<std::remove_extent_t<T>>{};
  const auto pointer = alloc.allocate(count);

  std::uninitialized_value_construct_n(pointer, count);

  return {pointer, detail::allocator_delete{alloc, count}};
}

template <class T>
constexpr auto make_shared(std::size_t count,
                           const std::remove_extent_t<T> &value) noexcept
    -> TT_REQUIRES(std::is_array_v<T> and std::extent_v<T> == 0,
                   std::shared_ptr<T>) {
  auto alloc = std::allocator<std::remove_extent_t<T>>{};
  const auto pointer = alloc.allocate(count);

  std::uninitialized_fill_n(pointer, count, value);

  return {pointer, detail::allocator_delete{alloc, count}};
}

template <class T>
constexpr auto make_shared_for_overwrite(std::size_t count) noexcept
    -> TT_REQUIRES(std::is_array_v<T> and std::extent_v<T> == 0,
                   std::shared_ptr<T>) {
  auto alloc = std::allocator<std::remove_extent_t<T>>{};
  const auto pointer = alloc.allocate(count);

  return {pointer, detail::allocator_delete{alloc, count}};
}

} // namespace core
} // namespace tt

#endif
