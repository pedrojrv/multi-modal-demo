from huggingface_hub import InferenceClient
from transformers import AutoProcessor
from huggingface_hub import login
from PIL import Image
import requests

# img_urls =["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
#            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"]
# images = [Image.open(requests.get(img_urls[0], stream=True).raw),
#           Image.open(requests.get(img_urls[1], stream=True).raw)]


login(token="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")

client = InferenceClient(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    token="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",
)


# processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")


# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "What do we see in this image?"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [
#             {"type": "text", "text": "In this image we can see two cats on the nets."},
#         ]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "And how about this image?"},
#         ]
#     },
# ]

# prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(text=prompt, images=[images[0], images[1]], return_tensors="pt")


# test = client.chat_completion(
#     messages=messages,
#     max_tokens=500,
#     stream=False,
# )

# for message in client.chat_completion(
#     messages=[{"role": "user", "content": "What is the capital of France?"}],
#     max_tokens=500,
#     stream=True,
# ):
#     print(message.choices[0].delta.content, end="")


import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"
headers = {
    "Authorization": "Bearer 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",
    "x-wait-for-model": "true"
}

response = requests.post(
    API_URL,
    headers=headers,
    json={
        "inputs": "What do we see in this image?",
        # "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }
)

print(response.json())
