from scripts.data_loader import load_and_prepare_data
from scripts.embedder import generate_embeddings, create_faiss_index
from scripts.retriever import query_lessons

if __name__ == "__main__":
    print("Loading and processing data...")
    texts = load_and_prepare_data("data/02. Lessons Learnt.xlsx")
    embeddings, model = generate_embeddings(texts)
    index, id_to_text = create_faiss_index(embeddings, texts)

    print("\nChatbot is ready! Type 'exit' to quit.\n")
    while True:
        q = input("Ask a question: ")
        if q.lower() == 'exit':
            break
        results = query_lessons(q, model, index, id_to_text)
        for i, res in enumerate(results, 1):
            print(f"\nResult {i}:\n{'-'*40}\n{res}\n")
