import base64
from src.utils import get_random_image_path
from groq import Groq


class GroqLlama32Vision:
    def __init__(self):
        self.client = Groq()

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def invoke(self, text, image: str = None):
        messages = [
            {
                "role": "system",
                "content": "Your only job is to provide detailed image captions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]

        if image:
            image_is_url = image.startswith("http")
            if image_is_url:
                input_image = image
            else:
                input_image = self.encode_image(image)
        else:
            input_image = None

        if input_image:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{input_image}" if not image_is_url else "${input_mage}",
                    },
                }
            )

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.2-11b-vision-preview",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return chat_completion.choices[0].message.content


if __name__ == "__main__":
    llama32 = GroqLlama32Vision()
    # print(llama32.invoke("Hi, how are you?"))
    print(llama32.invoke("Create a detailed caption for the attached image.", str(get_random_image_path())))
    print("Done!")
