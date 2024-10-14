import io
import os
import random
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


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


def display_binary_image(image):
    # Convert the bytes to a BytesIO object
    image_stream = io.BytesIO(image)

    # Open the image from the BytesIO object
    image = Image.open(image_stream)

    # Display the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.show()
