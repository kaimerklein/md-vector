import os
import glob
from langchain.text_splitter import MarkdownTextSplitter

from chroma_db_handler import ChromaDBHandler

# Initialize the ChromaDBHandler

chroma_db_handler = ChromaDBHandler(
    model_name="nomic-embed-text",
    embedding_url="http://localhost:11434/api/embeddings",
    db_path="./chroma_db",
    collection_name="TechnologyGuidelines",
    allow_reset=True
)

# Initialize the Markdown text splitter
text_splitter = MarkdownTextSplitter()

# Initialize the Nomic embeddings model
# embeddings_model = NomicEmbeddings(model="nomic-embed-text", inference_mode="local")


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
        
        # Add chunk and its embedding to the ChromaDB collection
        chroma_db_handler.add_chunk(file_path=file_path, chunk=chunk, i=i)

def main(directory):
    """Main function to read markdown files and process them."""
    markdown_files = read_markdown_files(directory)
    
    for file_path in markdown_files:
        process_markdown_file(file_path)
        print(f"Processed {file_path}")

if __name__ == "__main__":
    # Specify the directory containing markdown files here
    # directory_to_process = os.getcwd() #"./markdown_files"  # Change this to your target directory
    directory_to_process = "/Users/D046675/dev-local/TechnologyGuidelines"
    main(directory_to_process)