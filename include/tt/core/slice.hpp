#pragma once

#include <tt/core/concepts.hpp>

#include <experimental/mdspan>

namespace tt {
inline namespace core {

struct slice_fn {
private:
  using default_offset_type = tt::size_constant<0>;
  using default_stride_type = tt::size_constant<1>;

  static constexpr default_offset_type default_offset{};
  static constexpr default_stride_type default_stride{};

public:
  template <class TOffset, class TExtent, class TStride,
            class = TT_REQUIRES(tt::index<TOffset> and tt::index<TExtent> and
                                tt::index<TStride>)>
  constexpr std::strided_slice<TOffset, TExtent, TStride>
  operator()(TOffset offset, TExtent extent, TStride stride) const {
    return {offset, extent, stride};
  }

  template <class TOffset, class TExtent,
            class = TT_REQUIRES(tt::index<TOffset> and tt::index<TExtent>)>
  constexpr std::strided_slice<TOffset, TExtent, default_stride_type>
  operator()(TOffset offset, TExtent extent) const {
    return tt::slice_fn{}(offset, extent, default_stride);
  }

  template <class TExtent, class = TT_REQUIRES(tt::index<TExtent>)>
  constexpr std::strided_slice<default_offset_type, TExtent,
                               default_stride_type>
  operator()(TExtent extent) const {
    return tt::slice_fn{}(default_offset, extent);
  }
};

inline constexpr tt::slice_fn slice{};

} // namespace core
} // namespace tt
