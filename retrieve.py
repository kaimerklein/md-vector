from chroma_db_handler import ChromaDBHandler

# Initialize the ChromaDBHandler

chroma_db_handler = ChromaDBHandler(
    model_name="nomic-embed-text",
    embedding_url="http://localhost:11434/api/embeddings",
    db_path="./chroma_db",
    collection_name="TechnologyGuidelines",
    allow_reset=True
)

results = chroma_db_handler.get_chunks(search_term="Metering", n_results=3)

print(results)