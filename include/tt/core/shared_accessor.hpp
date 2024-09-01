#pragma once

#include <tt/core/offset_policy.hpp>

#include <memory>

namespace tt::inline core {

template <tt::arithmetic T, class TOffsetPolicy>
struct shared_accessor {
  using offset_policy = TOffsetPolicy::template offset<T>;
  using element_type = T;
  using reference = T &;
  using data_handle_type = std::shared_ptr<T[]>;

  constexpr shared_accessor() noexcept = default;

  template <class TOtherElement>
    requires std::is_convertible_v<TOtherElement (*)[], element_type (*)[]>
  constexpr shared_accessor(
      const shared_accessor<TOtherElement, TOffsetPolicy> &) noexcept {}

  static constexpr reference access(const data_handle_type &data_handle,
                                    std::size_t index) noexcept {
    return data_handle[index];
  }

  static constexpr offset_policy::data_handle_type
  offset(const data_handle_type &data_handle, std::size_t index) noexcept {
    return data_handle_type{data_handle, data_handle.get() + index};
  }
};

} // namespace tt::inline core
