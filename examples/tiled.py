from tt import arange, default_tile_extent, dtype, reshape, to_tiled


def main():
    # create a 3x5x7 tiled tensor of bfloat16...
    tiled_3d = arange(1, 106, dtype=dtype.BFloat16) | reshape(3, 5, 7) | to_tiled()
    # ...where tile size is 4x4
    assert default_tile_extent() == 4

    # create a padded view of the tiled tensor
    tiled_padded = tiled_3d | reshape(3, 8, 8)

    # create a 3x2x2x4x4 view of the tiled tensor
    tiled_5d = tiled_3d | reshape(3, 2, 2, 4, 4)

    print(
        f"3x5x7:\n{tiled_3d}\n"
        + f"3x8x8:\n{tiled_padded}\n"
        + f"3x2x2x4x4:\n{tiled_5d}\n"
    )


if __name__ == "__main__":
    main()
