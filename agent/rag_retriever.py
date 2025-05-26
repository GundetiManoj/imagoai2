import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_sheet_rag(sheet_name, model=None, base_path="outputs"):
    model = model or SentenceTransformer("all-MiniLM-L6-v2")
    path = os.path.join(base_path, sheet_name)

    # Load embeddings and cast to float32
    embeddings = np.load(os.path.join(path, "embeddings.npy")).astype(np.float32)
    with open(os.path.join(path, "sheet_sentences.json"), "r") as f:
        sentences = json.load(f)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    def retriever(query, k=5):
        query_emb = np.array([model.encode(query)], dtype=np.float32)
        _, top_k = index.search(query_emb, k)
        return "\n".join(sentences[i] for i in top_k[0])

    return retriever
