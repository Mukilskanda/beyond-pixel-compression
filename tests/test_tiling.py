from PIL import Image

from pixelpath.tiling import _grid_dims, tile_images


def test_grid_dims_perfect_squares():
    assert _grid_dims(4) == (2, 2)
    assert _grid_dims(9) == (3, 3)


def test_grid_dims_non_square():
    assert _grid_dims(2) == (1, 2)
    assert _grid_dims(6) == (2, 3)


def test_tile_images_produces_correct_grid_shape(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (100, 100), (i * 50, 0, 0)).save(p)
        paths.append(p)

    grid = tile_images(paths, tile_size=(100, 100), label_tiles=False, separator_px=4)
    # 2x2 grid, each cell 100+4 wide, +4 border
    assert grid.width == 4 + 2 * (100 + 4)
    assert grid.height == 4 + 2 * (100 + 4)


def test_tile_images_single_image():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/img.png"
        Image.new("RGB", (50, 50), "blue").save(p)
        grid = tile_images([p], tile_size=(50, 50), label_tiles=False)
        assert grid.width > 0 and grid.height > 0


def test_tile_images_rejects_empty_list():
    import pytest

    with pytest.raises(ValueError):
        tile_images([])
