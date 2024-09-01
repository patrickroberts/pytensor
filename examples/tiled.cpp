#include <tt/core/bfloat16.hpp>
#include <tt/core/format.hpp>
#include <tt/operators/arange.hpp>
#include <tt/operators/reshape.hpp>
#include <tt/operators/to_layout.hpp>

#include <format>
#include <iostream>

int main() {
  // create a 3x5x7 tiled tensor of bfloat16...
  const auto tiled_3d =
      tt::arange<tt::bfloat16>(1, 106) | tt::reshape(3, 5, 7) | tt::to_tiled();
  // ...where tile size is 4x4
  static_assert(tt::default_tile_extent == 4);

  // create a padded view of the tiled tensor
  const auto tiled_padded = tiled_3d | tt::reshape(3, 8, 8);

  // create a 3x2x2x4x4 view of the tiled tensor
  const auto tiled_5d = tiled_3d | tt::reshape(3, 2, 2, 4, 4);

  std::format_to(std::ostreambuf_iterator<char>(std::cout),
                 "3x5x7:\n{:>3}\n"
                 "3x8x8:\n{:>3}\n"
                 "3x2x2x4x4:\n{:>3}\n",
                 tiled_3d, tiled_padded, tiled_5d);
}
