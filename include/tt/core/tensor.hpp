#pragma once

#include <tt/core/concepts.hpp>
#include <tt/core/layout.hpp>
#include <tt/core/shared_accessor.hpp>

namespace tt {
inline namespace core {

template <class T, class TExtents, class TLayout>
using Tensor =
    TT_REQUIRES(tt::arithmetic<T> and tt::extents<TExtents>,
                std::mdspan<T, TExtents, TLayout, tt::shared_accessor<T>>);

template <class T, class TExtents>
using RowMajorTensor = tt::Tensor<T, TExtents, tt::RowMajor>;

template <class T, class TExtents = tt::dims<1>>
using RowMajorVector = TT_REQUIRES(tt::has_rank<TExtents, 1>,
                                   tt::RowMajorTensor<T, TExtents>);

template <class T, class TExtents = tt::dims<2>>
using RowMajorMatrix = TT_REQUIRES(tt::has_rank<TExtents, 2>,
                                   tt::RowMajorTensor<T, TExtents>);

template <class T, class TExtents>
using TiledTensor = tt::Tensor<T, TExtents, tt::Tiled>;

template <class T, class TExtents = tt::dims<1>>
using TiledVector = TT_REQUIRES(tt::has_rank<TExtents, 1>,
                                tt::TiledTensor<T, TExtents>);

template <class T, class TExtents = tt::dims<2>>
using TiledMatrix = TT_REQUIRES(tt::has_rank<TExtents, 2>,
                                tt::TiledTensor<T, TExtents>);

} // namespace core
} // namespace tt
