#%%
import os
import glob
from dotenv import load_dotenv

# Optional UI
import gradio as gr

# Imports for LangChain, Chroma, and Plotly
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# Config
MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Hugging Face embedding model
db_name = "vector_db"

# Load environment variables (optional)
load_dotenv(override=True)

# Load markdown documents from subfolders
text_loader_kwargs = {'encoding': 'utf-8'}
folders = [f for f in glob.glob("Jolibee Global/*") if os.path.isdir(f)]

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        if doc.page_content.strip():  # skip empty docs
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

print(f"Loaded {len(documents)} documents from {len(folders)} folders")

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Filter empty chunks
chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
print(f"Number of chunks after filtering empty ones: {len(chunks)}")

# Show document types
doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

# Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL,
    model_kwargs={"device": "cpu"}  # Change to "cuda" if GPU available
)

# Safely filter chunks that can be embedded
valid_chunks = []
failed_chunks = []
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    if not text:
        failed_chunks.append((i, "Empty text"))
        continue
    try:
        vec = embeddings.embed_documents([text])
        if vec and len(vec[0]) > 0:
            valid_chunks.append(chunk)
        else:
            failed_chunks.append((i, "Empty embedding"))
    except Exception as e:
        failed_chunks.append((i, str(e)))

print(f"Valid chunks for Chroma: {len(valid_chunks)}")
if failed_chunks:
    print(f"Failed chunks: {len(failed_chunks)}")
    for idx, reason in failed_chunks[:5]:  # Show first 5
        print(f"Chunk {idx} failed: {reason}")

# Delete existing Chroma collection if it exists
if os.path.exists(db_name):
    print("Deleting existing Chroma collection...")
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create Chroma vectorstore safely
vectorstore = Chroma.from_documents(
    documents=valid_chunks,
    embedding=embeddings,
    persist_directory=db_name
)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Sample one vector to check dimensions
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

# Prework for visualization
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
colors = [['blue', 'green', 'red', 'orange'][['branches', 'faq', 'products', 'promotions'].index(t)] for t in doc_types]

# 2D t-SNE visualization
n_samples = vectors.shape[0]
perplexity = min(30.0, max(5.0, n_samples - 1))

tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)
fig.show()

# 3D t-SNE visualization
tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)
fig.show()
