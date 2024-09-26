from huggingface_hub import InferenceClient

client = InferenceClient(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    token="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",
)

for message in client.chat_completion(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=500,
    stream=True,
):
    print(message.choices[0].delta.content, end="")

