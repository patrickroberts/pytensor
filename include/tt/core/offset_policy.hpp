#pragma once

#include <tt/core/concepts.hpp>

namespace tt {
inline namespace core {

using default_offset_policy = struct shared_offset_policy;

template <class T, class TOffsetPolicy = tt::default_offset_policy,
          class = TT_REQUIRES(tt::arithmetic<T>)>
struct shared_accessor;

struct shared_offset_policy {
  template <class T>
  using offset = tt::shared_accessor<T, tt::shared_offset_policy>;
};

template <class T, class = TT_REQUIRES(tt::arithmetic<T>)>
struct weak_accessor;

struct weak_offset_policy {
  template <class T>
  using offset = tt::weak_accessor<T>;
};

} // namespace core
} // namespace tt
