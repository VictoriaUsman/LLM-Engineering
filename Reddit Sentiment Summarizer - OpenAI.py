import os
import praw
import gradio as gr
import matplotlib.pyplot as plt
from openai import OpenAI
import json
import re

# -----------------------------
# Config
# -----------------------------
os.environ["REDDIT_CLIENT_ID"] = "Your API Here"
os.environ["REDDIT_SECRET"] = "Your API Here"
os.environ["OPENAI_API_KEY"] = "Your API Here"

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent="gradio-reddit-summary"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \"'")
    return text

def extract_json(raw_output: str):
    try:
        return json.loads(raw_output)
    except:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return None
    return None


def fetch_comments(topic, limit=30):
    comments = []
    subreddit = reddit.subreddit("all")
    for post in subreddit.search(topic, limit=limit):
        if post.selftext:
            comments.append(post.selftext[:500])
    return comments


def summarize_topic(topic):
    comments = fetch_comments(topic, limit=30)
    if not comments:
        return "No comments found.", None, None

    joined_comments = "\n\n".join(comments)


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""
Summarize the following Reddit opinions on '{topic}' in under 200 words.
Give the output STRICTLY in JSON:
{{
  "summary": "...",
  "sentiment_score": <number between 1 and 100>
}}

Text:
{joined_comments}
"""}
        ]
    )

    raw_output = response.choices[0].message.content.strip()
    parsed = extract_json(raw_output)

    if parsed:
        summary_text = clean_text(parsed.get("summary", "No summary found."))
        try:
            score = int(parsed.get("sentiment_score", 50))
        except:
            score = 50
    else:
        summary_text = clean_text(raw_output)
        score = 50

    score = max(1, min(100, score))

    # Pie chart (positive vs negative)
    labels = ["Positive", "Negative"]
    sizes = [score, 100 - score]
    colors = ["green", "red"]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    ax.set_title("Sentiment Distribution")

    return summary_text, str(score), fig


with gr.Blocks(css="""
.card {border:1px solid #ddd; border-radius:12px; padding:12px;
       box-shadow: 1px 2px 4px rgba(0,0,0,0.1); margin-bottom:10px;}
.score {font-size: 22px; font-weight: bold; text-align: center;}
""") as demo:
    gr.Markdown("##               🔎     Donkee")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                topic = gr.Textbox(label="Enter a topic")
                btn = gr.Button("Donkee It!", variant="primary")

            with gr.Group(elem_classes="card"):
                score_box = gr.Textbox(label="People's Score (1-100)", elem_classes="score")

            with gr.Group(elem_classes="card"):
                score_plot = gr.Plot(label="Sentiment Graph")

        with gr.Column(scale=2):
            with gr.Group(elem_classes="card"):
                summary = gr.Textbox(label="Summary", lines=20)

    btn.click(summarize_topic, inputs=topic, outputs=[summary, score_box, score_plot])

if __name__ == "__main__":
    demo.launch()



