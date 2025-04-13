import numpy as np

def query_lessons(question, model, index, id_to_text, k=3):
    query_embedding = model.encode([question])
    D, I = index.search(np.array(query_embedding), k)
    return [id_to_text[i] for i in I[0]]
