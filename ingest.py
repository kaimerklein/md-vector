import os
import glob
import chromadb
from datetime import datetime
from langchain.text_splitter import MarkdownTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_ollama import OllamaEmbeddings

# Set up ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.config.Settings(allow_reset=True)
)

# Create or get a collection for storing embeddings
collection_name = "markdown_chunks"
collection = client.create_collection(name=collection_name,get_or_create=True)

collection.add(ids=["id1"], embeddings=[[0]*768], documents=["test"], metadatas=[{"file_name": "test.md", "timestamp": datetime.now().isoformat()}])

# Initialize the Markdown text splitter
text_splitter = MarkdownTextSplitter()

# Initialize the Nomic embeddings model
# embeddings_model = NomicEmbeddings(model="nomic-embed-text", inference_mode="local")

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
)

def read_markdown_files(directory):
    """Recursively read all markdown files in a directory."""
    markdown_files = glob.glob(os.path.join(directory, '**', '*.md'), recursive=True)
    return markdown_files

def process_markdown_file(file_path):
    """Process a single markdown file: split into chunks and store embeddings."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into chunks using MarkdownTextSplitter
    chunks = text_splitter.split_text(content)

    # Generate embeddings for each chunk and store them in ChromaDB
    for i, chunk in enumerate(chunks):
        embedding_vector = embeddings_model.embed_query(chunk)

        print(f"Chunk {i}:", chunk)
        print("Embedding shape:", embedding_vector)

        
        # Add chunk and its embedding to the ChromaDB collection
        collection.add(
            documents=[chunk],
            ids=[f"{file_path}_chunk_{i}"],
            embeddings=[embedding_vector],
            metadatas=[{"file_name": os.path.basename(file_path), "timestamp": datetime.now().isoformat()}]
        )

def main(directory):
    """Main function to read markdown files and process them."""
    markdown_files = read_markdown_files(directory)
    
    for file_path in markdown_files:
        process_markdown_file(file_path)
        print(f"Processed {file_path}")

if __name__ == "__main__":
    # Specify the directory containing markdown files here
    directory_to_process = os.getcwd() #"./markdown_files"  # Change this to your target directory
    main(directory_to_process)