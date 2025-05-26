from sentence_transformers import SentenceTransformer
import numpy as np
import json

def tabular_to_sentences(df):
    sentences = []
    for _, row in df.iterrows():
        sentence = ", ".join([f"{col} is {val}" for col, val in row.items()])
        sentences.append(sentence)
    return sentences

def get_tabular_embeddings(sentences, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

def save_outputs(sentences, embeddings, out_dir="outputs"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/sheet_sentences.json", "w") as f:
        json.dump(sentences, f, indent=2)

    np.save(f"{out_dir}/embeddings.npy", embeddings)
