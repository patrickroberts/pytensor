#pragma once

#include <tt/core/concepts.hpp>

namespace tt::inline core {

using default_offset_policy = struct shared_offset_policy;

template <tt::arithmetic T, class TOffsetPolicy = tt::default_offset_policy>
struct shared_accessor;

struct shared_offset_policy {
  template <tt::arithmetic T>
  using offset = tt::shared_accessor<T, tt::shared_offset_policy>;
};

template <tt::arithmetic T>
struct weak_accessor;

struct weak_offset_policy {
  template <tt::arithmetic T>
  using offset = tt::weak_accessor<T>;
};

} // namespace tt::inline core
