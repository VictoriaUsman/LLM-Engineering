#ChatBot - Using Hugging Face

import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

HF_TOKEN = "HF Token Here"

client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HF_TOKEN
)

# Chat function
def chat_fn(user_message, history):
    if history is None:
        history = []

    messages = [{"role": "system", "content": "You are a funny assistant."}] + history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=messages,
        max_tokens=256,
        temperature=0.7
    )

    reply = response.choices[0].message["content"]

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply}
    ]
    return history, history

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("Funny Assistant")

    chatbot = gr.Chatbot(
        label="Chat",
        type="messages",
        bubble_full_width=False,
        show_copy_button=True
    )

    with gr.Row():
        user_in = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=9
        )
        send_btn = gr.Button("Send", scale=1)

    state = gr.State([])  # keeps conversation history

    send_btn.click(
        fn=chat_fn,
        inputs=[user_in, state],
        outputs=[chatbot, state]
    )
    user_in.submit(
        fn=chat_fn,
        inputs=[user_in, state],
        outputs=[chatbot, state]
    )

demo.launch()