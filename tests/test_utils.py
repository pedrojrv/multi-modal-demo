import io
import pytest
from PIL import Image
from pathlib import Path
from src.utils import get_assets_path, get_random_image_path, display_binary_image
from unittest.mock import patch


def test_get_assets_path_env_var_set(monkeypatch):
    monkeypatch.setenv('ASSETS_PATH', '/some/path')
    assert get_assets_path() == Path('/some/path')


def test_get_assets_path_env_var_not_set(monkeypatch):
    monkeypatch.delenv('ASSETS_PATH', raising=False)
    with pytest.raises(ValueError, match='ASSETS_PATH environment variable not set'):
        get_assets_path()


def test_get_random_image_path(monkeypatch, tmp_path):
    # Set up a temporary directory with some jpg files
    assets_path = tmp_path / "assets"
    assets_path.mkdir()
    (assets_path / "image1.jpg").write_text("image1")
    (assets_path / "image2.jpg").write_text("image2")
    (assets_path / "image3.jpg").write_text("image3")

    # Set the ASSETS_PATH environment variable to the temporary directory
    monkeypatch.setenv('ASSETS_PATH', str(assets_path))

    # Call the function and check that the returned path is one of the jpg files
    image_path = get_random_image_path()
    assert image_path in [assets_path / "image1.jpg", assets_path / "image2.jpg", assets_path / "image3.jpg"]


def test_get_random_image_path_no_images(monkeypatch, tmp_path):
    # Set up a temporary directory with no jpg files
    assets_path = tmp_path / "assets"
    assets_path.mkdir()

    # Set the ASSETS_PATH environment variable to the temporary directory
    monkeypatch.setenv('ASSETS_PATH', str(assets_path))

    # Call the function and check that it raises a ValueError
    with pytest.raises(ValueError, match='No images found in the assets path'):
        get_random_image_path()


def test_display_binary_image():
    # Create a simple image using PIL
    image = Image.new('RGB', (10, 10), color='red')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    with patch('matplotlib.pyplot.show') as show_mock:
        # Call the function
        display_binary_image(image_bytes)

        # Check that plt.show was called
        assert show_mock.called
