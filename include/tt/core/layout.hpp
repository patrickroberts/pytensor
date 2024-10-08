#pragma once

#include <tt/core/bit.hpp>
#include <tt/core/concepts.hpp>

#include <experimental/mdspan>

namespace tt {
inline namespace core {
namespace detail {
namespace {

constexpr auto find_next_multiple(std::size_t alignment,
                                  std::size_t offset) noexcept -> std::size_t {
  if (offset == std::dynamic_extent) {
    return offset;
  }

  return ((alignment - 1 + offset) / alignment) * alignment;
}

} // namespace
} // namespace detail

inline constexpr std::size_t default_tile_extent = 4;

template <std::size_t TileHeight = tt::default_tile_extent,
          std::size_t TileWidth = TileHeight>
struct layout_right_tiled {
private:
  static constexpr std::size_t tile_height = TileHeight;
  static constexpr std::size_t tile_width = TileWidth;
  static constexpr std::size_t tile_size = tile_height * tile_width;

  template <class TExtents>
  using padding = std::extents<
      std::size_t,
      detail::find_next_multiple(
          tile_height, TExtents::rank() >= 2
                           ? TExtents::static_extent(TExtents::rank() - 2)
                           : tile_height),
      detail::find_next_multiple(
          tile_width, TExtents::rank() >= 1
                          ? TExtents::static_extent(TExtents::rank() - 1)
                          : tile_width)>;

  static_assert(tile_height != std::dynamic_extent and
                tile_width != std::dynamic_extent);
  static_assert(tt::has_single_bit(tile_size));

public:
  template <class TExtents, class = std::enable_if_t<tt::extents<TExtents>>>
  struct mapping {
    using extents_type = TExtents;
    using index_type = tt::index_type_t<extents_type>;
    using size_type = tt::size_type_t<extents_type>;
    using rank_type = tt::rank_type_t<extents_type>;
    using layout_type = layout_right_tiled;

  private:
    using padding_type = layout_right_tiled::padding<TExtents>;

    static constexpr auto
    make_padding(const extents_type &exts) noexcept -> padding_type {
      return padding_type{
          detail::find_next_multiple(tile_height,
                                     TExtents::rank() >= 2
                                         ? exts.extent(TExtents::rank() - 2)
                                         : tile_height),
          detail::find_next_multiple(tile_width,
                                     TExtents::rank() >= 1
                                         ? exts.extent(TExtents::rank() - 1)
                                         : tile_width),
      };
    }

    [[no_unique_address]] extents_type exts;
    [[no_unique_address]] padding_type pads;

    constexpr auto
    accumulate_offset(index_type value, index_type row,
                      index_type col) const noexcept -> index_type {
      return this->pads.extent(1) *
                 (value + (row / tile_height) * tile_height) +
             (row % tile_height) * tile_width + (col / tile_width) * tile_size +
             (col % tile_width);
    }

    constexpr auto
    accumulate_offset(index_type value) const noexcept -> index_type {
      return accumulate_offset(value, 0, 0);
    }

    constexpr auto
    accumulate_offset(index_type value,
                      index_type col) const noexcept -> index_type {
      return accumulate_offset(value, 0, col);
    }

    template <class... TIndices>
    constexpr auto
    accumulate_offset(index_type value, index_type index,
                      TIndices... indices) const noexcept -> index_type {
      if constexpr (sizeof...(indices) == 2) {
        return this->accumulate_offset(this->pads.extent(0) * (value + index),
                                       indices...);
      } else {
        constexpr rank_type r = extents_type::rank() - sizeof...(indices);

        return this->accumulate_offset(this->exts.extent(r) * (value + index),
                                       indices...);
      }
    }

  public:
    constexpr mapping() noexcept : mapping(extents_type{}) {}

    constexpr mapping(const mapping &) noexcept = default;

    constexpr auto operator=(const mapping &) noexcept -> mapping & = default;

    constexpr mapping(const extents_type &exts) noexcept
        : exts(exts), pads(mapping::make_padding(exts)) {}

    template <class TMapping>
    constexpr auto operator==(const TMapping &rhs) const noexcept
        -> std::enable_if_t<tt::mapping<TMapping, layout_type> and
                                TMapping::extents_type::rank() ==
                                    extents_type::rank(),
                            bool> {
      return this->exts == rhs.extents();
    }

    constexpr auto extents() const noexcept -> const extents_type & {
      return exts;
    }

    template <class... TIndices>
    constexpr auto operator()(TIndices... indices) const noexcept
        -> std::enable_if_t<(... and tt::index<TIndices>), index_type> {
      return this->accumulate_offset(0, static_cast<index_type>(indices)...);
    }

    constexpr auto required_span_size() const noexcept -> index_type {
      if constexpr (extents_type::rank() < 2) {
        return this->pads.extent(0) * this->pads.extent(1);
      } else {
        return this->exts.extent(0) * this->stride(0);
      }
    }

    constexpr auto stride(rank_type r) const noexcept -> index_type {
      constexpr rank_type unpadded_ranks =
          extents_type::rank() >= 2 ? extents_type::rank() - 2 : 0;

      assert(r < unpadded_ranks);

      index_type value = this->pads.extent(0) * this->pads.extent(1);

      for (rank_type i = r + 1; i < unpadded_ranks; ++i) {
        value *= this->exts.extent(i);
      }

      return value;
    }

    static constexpr auto is_always_unique() noexcept -> bool { return true; }

    static constexpr auto is_always_strided() noexcept -> bool { return false; }

    static constexpr auto is_always_exhaustive() noexcept -> bool {
      constexpr rank_type rank = extents_type::rank();

      if constexpr (rank < 2) {
        return false;
      } else if constexpr (extents_type::static_extent(rank - 2) ==
                               std::dynamic_extent or
                           extents_type::static_extent(rank - 1) ==
                               std::dynamic_extent) {
        return false;
      } else {
        return extents_type::static_extent(rank - 2) ==
                   padding_type::static_extent(0) and
               extents_type::static_extent(rank - 1) ==
                   padding_type::static_extent(1);
      }
    }

    static constexpr auto is_unique() noexcept -> bool { return true; }

    constexpr auto is_exhaustive() const noexcept -> bool {
      constexpr rank_type rank = extents_type::rank();

      if constexpr (rank < 2) {
        return false;
      } else {
        return this->exts.extent(rank - 2) == this->pads.extent(0) and
               this->exts.extent(rank - 1) == this->pads.extent(1);
      }
    }

    static constexpr auto is_strided() noexcept -> bool { return false; }
  };
};

using RowMajor = std::layout_right;
using Strided = std::layout_stride;
using Tiled = tt::layout_right_tiled<>;

enum class layout {
  RowMajor,
  Strided,
  Tiled,
};

template <class T, tt::layout V>
struct layout_traits {
  using type = T;
  static constexpr auto value = V;
};

struct layouts : layout_traits<tt::RowMajor, tt::layout::RowMajor>,
                 layout_traits<tt::Strided, tt::layout::Strided>,
                 layout_traits<tt::Tiled, tt::layout::Tiled> {
  template <class T, tt::layout V>
  using fn = layout_traits<T, V>;
};

} // namespace core
} // namespace tt
