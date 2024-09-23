import chromadb

# Initialize the ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.config.Settings(allow_reset=True)
)

# Create or get a collection for storing embeddings
collection_name = "markdown_chunks"

collection = client.get_collection(name=collection_name)

# Retrieve all documents (chunks) from the collection
documents = collection.get(include=['documents', 'metadatas', 'embeddings'])

# Iterate over the documents and print them
for doc, metadata in zip(documents['documents'], documents['metadatas']):
    print("Document ID:", metadata.get('id', 'N/A'))
    print("Content:", doc)
    print("Metadata:", metadata)
    print("-" * 80)
