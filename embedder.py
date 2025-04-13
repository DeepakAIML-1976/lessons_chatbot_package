from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def generate_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, model

def create_faiss_index(embeddings, texts):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    id_to_text = dict(enumerate(texts))
    return index, id_to_text
