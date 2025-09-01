# huggingface_rag.py
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# Configure your LLM (Gemini API key)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
# Initialize Hugging Face embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
client = chromadb.Client()
collection_name = "documents"

# Create or get collection
try:
    collection = client.get_collection(collection_name)
except:
    collection = client.create_collection(collection_name)

# Function to add documents to collection
def add_documents(documents):
    embeddings = embedding_model.encode(documents)
    for doc, emb in zip(documents, embeddings):
        collection.add(
            documents=[doc],
            metadatas=[{"source": doc}],
            embeddings=[emb.tolist()]  # Convert numpy array to list
        )

# Function to retrieve top-k relevant documents
def retrieve_documents(query, top_k=3):
    query_embedding = embedding_model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results['documents'][0]  # List of top-k documents

# Function to generate response using LLM
def generate_response(query):
    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs)
    prompt = f"Answer the following question based on these documents:\n{context}\nQuestion: {query}"

    response = model.generate_content(
        model="gemini-pro",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message["content"]

# Example usage
if __name__ == "__main__":
    docs = [
        "RAG combines retrieval and generation to improve answers.",
        "Hugging Face embeddings allow semantic similarity search.",
        "ChromaDB efficiently stores and retrieves embeddings."
    ]

    add_documents(docs)

    query = "What is RAG in AI?"
    answer = generate_response(query)
    print("AI Response:", answer)
