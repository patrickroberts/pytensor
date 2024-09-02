#pragma once

#include <tt/core/concepts.hpp>

#include <experimental/mdspan>

#include <bit>

namespace tt::inline core {
namespace {

constexpr std::size_t find_next_multiple(std::size_t alignment,
                                         std::size_t offset) {
  if (offset == std::dynamic_extent) {
    return offset;
  }

  return ((alignment - 1 + offset) / alignment) * alignment;
}

} // namespace

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
      tt::find_next_multiple(tile_height,
                             TExtents::static_extent(TExtents::rank() - 2)),
      tt::find_next_multiple(tile_width,
                             TExtents::static_extent(TExtents::rank() - 1))>;

  static_assert(tile_height != std::dynamic_extent and
                tile_width != std::dynamic_extent);
  static_assert(std::has_single_bit(tile_size));

public:
  template <tt::extents TExtents>
  struct mapping {
    static_assert(TExtents::rank() >= 2);

    using extents_type = TExtents;
    using index_type = tt::index_type_t<extents_type>;
    using size_type = tt::size_type_t<extents_type>;
    using rank_type = tt::rank_type_t<extents_type>;
    using layout_type = layout_right_tiled;

  private:
    using padding_type = layout_right_tiled::padding<TExtents>;

    static constexpr padding_type
    make_padding(const extents_type &exts) noexcept {
      return padding_type{
          tt::find_next_multiple(tile_height,
                                 exts.extent(TExtents::rank() - 2)),
          tt::find_next_multiple(tile_width, exts.extent(TExtents::rank() - 1)),
      };
    }

    [[no_unique_address]] extents_type exts;
    [[no_unique_address]] padding_type pads;

    constexpr index_type accumulate_offset(index_type value, index_type row,
                                           index_type col) const noexcept {
      return this->pads.extent(1) *
                 (value + (row / tile_height) * tile_height) +
             (row % tile_height) * tile_width + (col / tile_width) * tile_size +
             (col % tile_width);
    }

    constexpr index_type
    accumulate_offset(index_type value, index_type index,
                      std::same_as<index_type> auto... indices) const noexcept {
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

    constexpr mapping &operator=(const mapping &) noexcept = default;

    constexpr mapping(const extents_type &exts) noexcept
        : exts(exts), pads(mapping::make_padding(exts)) {}

    template <tt::mapping<layout_type> TMapping>
      requires(TMapping::extents_type::rank() == extents_type::rank())
    constexpr bool operator==(const TMapping &rhs) const noexcept {
      return this->exts == rhs.extents();
    }

    constexpr const extents_type &extents() const noexcept { return exts; }

    constexpr index_type operator()(tt::index auto... indices) const noexcept {
      return this->accumulate_offset(0, static_cast<index_type>(indices)...);
    }

    constexpr index_type required_span_size() const noexcept {
      return this->exts.extent(0) * this->stride(0);
    }

    constexpr index_type stride(rank_type r) const noexcept {
      constexpr rank_type unpadded_ranks = extents_type::rank() - 2;

      assert(r < unpadded_ranks);

      index_type value = this->pads.extent(0) * this->pads.extent(1);

      for (rank_type i = r + 1; i < unpadded_ranks; ++i) {
        value *= this->exts.extent(i);
      }

      return value;
    }

    static constexpr bool is_always_unique() noexcept { return true; }

    static constexpr bool is_always_strided() noexcept { return false; }

    static constexpr bool is_always_exhaustive() noexcept {
      constexpr rank_type rank = extents_type::rank();

      if constexpr (extents_type::static_extent(rank - 2) ==
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

    static constexpr bool is_unique() noexcept { return true; }

    constexpr bool is_exhaustive() const noexcept {
      constexpr rank_type rank = extents_type::rank();

      return this->exts.extent(rank - 2) == this->pads.extent(0) and
             this->exts.extent(rank - 1) == this->pads.extent(1);
    }

    static constexpr bool is_strided() noexcept { return false; }
  };
};

using RowMajor = std::layout_right;
using Strided = std::layout_stride;
using Tiled = tt::layout_right_tiled<>;

enum class Layout {
  RowMajor,
  Strided,
  Tiled,
};

} // namespace tt::inline core
