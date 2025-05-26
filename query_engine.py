import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from llm_agent import answer_question_with_context

def load_embeddings():
    embeddings = np.load("outputs/embeddings.npy")
    with open("outputs/sheet_sentences.json", "r") as f:
        sentences = json.load(f)
    return embeddings, sentences

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_top_k(query, model, index, sentences, k=5):
    query_embedding = model.encode([query])[0]
    _, top_k_indices = index.search(np.array([query_embedding]), k)
    top_k_contexts = [sentences[i] for i in top_k_indices[0]]
    return top_k_contexts

def ask_question(query):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings, sentences = load_embeddings()
    index = create_faiss_index(embeddings)

    top_contexts = retrieve_top_k(query, model, index, sentences)
    return answer_question_with_context(query, top_contexts)
