#pragma once

#include <tt/core/concepts.hpp>
#include <tt/core/layout.hpp>
#include <tt/core/shared_accessor.hpp>

namespace tt {
inline namespace core {

template <class T, class TExtents, class TLayout,
          class = std::enable_if_t<tt::arithmetic<T> and tt::extents<TExtents>>>
using Tensor = std::mdspan<T, TExtents, TLayout, tt::shared_accessor<T>>;

template <class T, class TExtents>
using RowMajorTensor = tt::Tensor<T, TExtents, tt::RowMajor>;

template <class T, class TExtents = tt::dims<1>,
          class = std::enable_if_t<tt::has_rank<TExtents, 1>>>
using RowMajorVector = tt::RowMajorTensor<T, TExtents>;

template <class T, class TExtents = tt::dims<2>,
          class = std::enable_if_t<tt::has_rank<TExtents, 2>>>
using RowMajorMatrix = tt::RowMajorTensor<T, TExtents>;

template <class T, class TExtents>
using TiledTensor = tt::Tensor<T, TExtents, tt::Tiled>;

template <class T, class TExtents = tt::dims<1>,
          class = std::enable_if_t<tt::has_rank<TExtents, 1>>>
using TiledVector = tt::TiledTensor<T, TExtents>;

template <class T, class TExtents = tt::dims<2>,
          class = std::enable_if_t<tt::has_rank<TExtents, 2>>>
using TiledMatrix = tt::TiledTensor<T, TExtents>;

} // namespace core
} // namespace tt
