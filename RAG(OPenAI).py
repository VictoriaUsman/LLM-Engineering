# Desc: Simple RAG using OPENAI


import os
import glob
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

MODEL = "gpt-4o-mini"

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'You OPEN AI API KEY')
openai = OpenAI()

Feraros = glob.glob("/Users/iantristancultura/llm_engineering/week5/knowledge-base/Feraro/*")

context = {}

for Feraro in Feraros:
    if os.path.isfile(Feraro):
        name = os.path.splitext(os.path.basename(Feraro))[0]
        with open(Feraro, "r", encoding="utf-8") as f:
            doc = f.read()
        context[name] = doc
    else:
        print(f"⚠️ Skipping folder: {Feraro}")

print("✅ Loaded files:", context.keys())


system_message = "You are an expert in answering accurate questions about Feraro. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."


def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context

def add_context(message):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history
    message = add_context(message)
    messages.append({"role": "user", "content": message})

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

view = gr.ChatInterface(chat, type="messages").launch()




