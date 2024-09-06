#pragma once

#include <memory>

namespace tt {
inline namespace operators {
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

  constexpr void operator()(T *ptr) { alloc.deallocate(ptr, elements); }
};

} // namespace detail
} // namespace operators
} // namespace tt
