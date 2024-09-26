import gradio as gr
import random

def random_response(message, history):
    num_files = len(message["files"])
    print(f"You uploaded {num_files} files")
    return random.choice(["Yes", "No"])


demo = gr.ChatInterface(
    random_response,
    chatbot=gr.Chatbot(height=300, placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything"),
    # textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="LlaMA 3.2 Multi-Modal Chatbot",
    description="Ask me any question with and without images.",
    # theme="soft",
    # examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    # cache_examples=True,
    # retry_btn=None,
    # undo_btn="Delete Previous",
    # clear_btn="Clear",
    multimodal=True,
)

demo.launch()
