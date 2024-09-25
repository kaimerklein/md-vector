import os
import glob
from langchain.text_splitter import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vector_store = Chroma(
    collection_name="TechnologyGuidelines",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # Where to save data locally, remove if not necessary
)

# Initialize the Markdown text splitter
text_splitter = MarkdownTextSplitter()


def read_markdown_files(directory):
    """Recursively read all markdown files in a directory."""
    markdown_files = glob.glob(os.path.join(directory, '**', 'README.md'), recursive=True)
    return markdown_files

def process_markdown_file(file_path):
    """Process a single markdown file: split into chunks and store embeddings."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into chunks using MarkdownTextSplitter
    chunks = text_splitter.split_text(content)

    docs = [Document(page_content=chunk, metadata={"file_path": file_path}) for chunk in chunks]

    ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]

    vector_store.add_documents(documents=docs,ids=ids)


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