import torch
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class HFBGESmallEnV15:
    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
        self.model.eval()

    def invoke(self, texts, instruction=False, batch_size=10):
        texts = [texts] if isinstance(texts, str) else texts
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i + 1} - {i + len(batch)}")

            # Tokenize sentences
            if instruction:
                encoded_input = self.tokenizer(
                    [instruction + q for q in batch], padding=True, truncation=True, return_tensors='pt'
                )
            else:
                encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0]

            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings = sentence_embeddings.tolist()
            all_embeddings.extend(sentence_embeddings)

        return all_embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bge = HFBGESmallEnV15()
    outputs = bge.invoke(["Hello, world!", "Goodbye, world!"] * 10)
    print(len(outputs))
