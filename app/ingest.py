import os
import pickle

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from app.config import OPENAI_API_KEY

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIMENSION = 1536

client = OpenAI(api_key=OPENAI_API_KEY)


def load_document(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks


def get_embedding(text: str) -> list[float]:
    import numpy as np
    return np.random.rand(1536).tolist()


def build_vector_store() -> None:
    file_path = "data/company_policy.txt"

    text = load_document(file_path)
    print("Document loaded successfully.\n")

    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}\n")

    embeddings = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Creating embedding for chunk {i}...")
        embedding = get_embedding(chunk)
        embeddings.append(embedding)

    embedding_array = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embedding_array)

    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(index, "vector_store/faiss_index.bin")

    with open("vector_store/chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)

    print("\nVector store created successfully.")
    print("Saved files:")
    print("- vector_store/faiss_index.bin")
    print("- vector_store/chunks.pkl")


if __name__ == "__main__":
    build_vector_store()