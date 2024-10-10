import os
import random
from pathlib import Path


def get_assets_path() -> Path:
    path = os.environ.get('ASSETS_PATH', '')
    if not path:
        raise ValueError('ASSETS_PATH environment variable not set')

    return Path(path)


def get_random_image_path() -> Path:
    # search for all jpg and return a random one
    assets_path = get_assets_path()
    images = list(assets_path.glob('*.jpg'))
    if not images:
        raise ValueError('No images found in the assets path')

    return random.choice(images)
