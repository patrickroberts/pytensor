#pragma once

#include <tt/core/concepts.hpp>
#include <tt/core/shared_accessor.hpp>
#include <tt/core/tiled_layout.hpp>

namespace tt::inline core {

template <tt::arithmetic T, tt::extents TExtents, tt::layout<TExtents> TLayout>
using shared_tensor = std::mdspan<T, TExtents, TLayout, tt::shared_accessor<T>>;

template <tt::arithmetic T, tt::extents TExtents>
using row_major_tensor = tt::shared_tensor<T, TExtents, tt::row_major_layout>;

template <tt::arithmetic T,
          tt::has_rank<1> TExtents = std::dextents<std::size_t, 1>>
using row_major_vector = tt::row_major_tensor<T, TExtents>;

template <tt::arithmetic T,
          tt::has_rank<2> TExtents = std::dextents<std::size_t, 2>>
using row_major_matrix = tt::row_major_tensor<T, TExtents>;

template <tt::arithmetic T, tt::extents TExtents>
using tiled_tensor = tt::shared_tensor<T, TExtents, tt::tiled_layout>;

template <tt::arithmetic T,
          tt::has_rank<1> TExtents = std::dextents<std::size_t, 1>>
using tiled_vector = tt::tiled_tensor<T, TExtents>;

template <tt::arithmetic T,
          tt::has_rank<2> TExtents = std::dextents<std::size_t, 2>>
using tiled_matrix = tt::tiled_tensor<T, TExtents>;

} // namespace tt::inline core
