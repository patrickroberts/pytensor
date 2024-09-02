#pragma once

#include <tt/core/concepts.hpp>
#include <tt/core/layout.hpp>
#include <tt/core/shared_accessor.hpp>

namespace tt::inline core {

template <tt::arithmetic T, tt::extents TExtents, class TLayout>
using Tensor = std::mdspan<T, TExtents, TLayout, tt::shared_accessor<T>>;

template <tt::arithmetic T, tt::extents TExtents>
using RowMajorTensor = tt::Tensor<T, TExtents, tt::RowMajor>;

template <tt::arithmetic T,
          tt::has_rank<1> TExtents = std::dextents<std::size_t, 1>>
using RowMajorVector = tt::RowMajorTensor<T, TExtents>;

template <tt::arithmetic T,
          tt::has_rank<2> TExtents = std::dextents<std::size_t, 2>>
using RowMajorMatrix = tt::RowMajorTensor<T, TExtents>;

template <tt::arithmetic T, tt::extents TExtents>
using TiledTensor = tt::Tensor<T, TExtents, tt::Tiled>;

template <tt::arithmetic T,
          tt::has_rank<1> TExtents = std::dextents<std::size_t, 1>>
using TiledVector = tt::TiledTensor<T, TExtents>;

template <tt::arithmetic T,
          tt::has_rank<2> TExtents = std::dextents<std::size_t, 2>>
using TiledMatrix = tt::TiledTensor<T, TExtents>;

} // namespace tt::inline core
