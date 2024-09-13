#pragma once

#include <tt/core/offset_policy.hpp>

#include <memory>

namespace tt {
inline namespace core {

template <class T, class TOffsetPolicy, class TEnable>
struct shared_accessor {
  using offset_policy = typename TOffsetPolicy::template offset<T>;
  using element_type = T;
  using reference = T &;
  using data_handle_type = std::shared_ptr<T[]>;

  constexpr shared_accessor() noexcept = default;

  template <class TOtherElement, class = std::enable_if_t<std::is_convertible_v<
                                     TOtherElement (*)[], element_type (*)[]>>>
  constexpr shared_accessor(
      const shared_accessor<TOtherElement, TOffsetPolicy> &) noexcept {}

  static constexpr auto access(const data_handle_type &data_handle,
                               std::size_t index) noexcept -> reference {
    return data_handle[index];
  }

  static constexpr auto offset(const data_handle_type &data_handle,
                               std::size_t index) noexcept ->
      typename offset_policy::data_handle_type {
    return data_handle_type{data_handle, data_handle.get() + index};
  }
};

} // namespace core
} // namespace tt
