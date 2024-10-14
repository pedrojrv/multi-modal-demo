import torch
import logging

from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from src.utils import get_random_image_path

logger = logging.getLogger(__name__)


class ViTImageEmbeddings:
    def __init__(self):
        # Load model and processor from HuggingFace Hub
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model.eval()

    def invoke(self, images, batch_size=10):
        images = [images] if isinstance(images, str) else images
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            logger.info(f"Processing batch {i + 1} - {i + len(batch)}")

            batch_image = [Image.open(img) for img in batch]

            # Process image
            inputs = self.processor(images=batch_image, return_tensors="pt")

            # Compute image embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # last_hidden_states = outputs.last_hidden_state
                image_embeddings = outputs.pooler_output

            # Normalize embeddings
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
            image_embeddings_lst = image_embeddings.tolist()
            all_embeddings.extend(image_embeddings_lst)

        return all_embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    image_path = get_random_image_path()

    vit = ViTImageEmbeddings()
    embeddings = vit.invoke([image_path] * 10, batch_size=5)
    logger.info(f"Lenght of embeddings: {len(embeddings)}")
    logger.info(f"Length of each embedding: {len(embeddings[0])}")
