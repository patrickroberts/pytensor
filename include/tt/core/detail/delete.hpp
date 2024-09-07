#pragma once

#include <memory>

namespace tt {
inline namespace core {
namespace detail {

template <class T>
class allocator_delete {
  std::allocator<T> alloc;
  std::size_t elements;

public:
  constexpr allocator_delete(const allocator_delete &other) noexcept = default;

  constexpr allocator_delete(const std::allocator<T> &alloc,
                             const std::size_t elements) noexcept
      : alloc(alloc), elements(elements) {}

  constexpr auto operator()(T *ptr) -> void { alloc.deallocate(ptr, elements); }
};

} // namespace detail
} // namespace core
} // namespace tt
