import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

os.environ['HF_HOME'] = str(Path(__file__).parent / "models")


class HFBGELargeEnV15:
    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
        self.model.eval()

    def invoke(self, texts, instruction=False):
        # Tokenize sentences
        if instruction:
            encoded_input = self.tokenizer(
                [instruction + q for q in texts], padding=True, truncation=True, return_tensors='pt'
            )
        else:
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]

        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()


if __name__ == "__main__":
    bge = HFBGELargeEnV15()
    print(bge.invoke(["Hello, world!"]))

    outputs = bge.invoke(["Hello, world!", "Goodbye, world!"])
    print(len(outputs))
