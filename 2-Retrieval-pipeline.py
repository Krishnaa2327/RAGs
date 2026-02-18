import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Verify OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
# embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Check if vector store has documents
num_docs = db._collection.count()
print(f"Documents in vector store: {num_docs}\n")

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

# Also try similarity search with scores
results_with_scores = db.similarity_search_with_score(query, k=5)

print(f"User Query: {query}")
print("--- Context (with similarity scores) ---")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\nDocument {i} (Score: {score:.4f}):")
    print(f"{doc.page_content[:500]}...\n")


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"