#pragma once

#include <tt/core/offset_policy.hpp>

#include <cassert>
#include <memory>

namespace tt {
inline namespace core {

template <class T, class TEnable>
struct weak_accessor {
  using offset_policy = weak_accessor;
  using element_type = T;
  using reference = T &;
  using data_handle_type = std::weak_ptr<T[]>;

  constexpr weak_accessor() noexcept = default;

  template <class TOtherElement, class = std::enable_if_t<std::is_convertible_v<
                                     TOtherElement (*)[], element_type (*)[]>>>
  constexpr weak_accessor(
      const shared_accessor<TOtherElement, weak_offset_policy> &) noexcept {}

  template <class TOtherElement, class = std::enable_if_t<std::is_convertible_v<
                                     TOtherElement (*)[], element_type (*)[]>>>
  constexpr weak_accessor(const weak_accessor<TOtherElement> &) noexcept {}

  static constexpr auto access(const data_handle_type &data_handle,
                               std::size_t index) noexcept -> reference {
    const auto locked_data_handle = data_handle.lock();

    assert(locked_data_handle);

    return locked_data_handle[index];
  }

  static constexpr auto
  offset(const data_handle_type &data_handle,
         std::size_t index) noexcept -> offset_policy::data_handle_type {
    const auto locked_data_handle = data_handle.lock();

    assert(locked_data_handle);

    return std::shared_ptr<T[]>{
        locked_data_handle,
        locked_data_handle.get() + index,
    };
  }
};

} // namespace core
} // namespace tt
