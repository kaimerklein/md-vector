import os
import glob
from langchain.text_splitter import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

class GroundingStore:
    def __init__(
        self,
        directory,
        collection_name="TechnologyGuidelines",
        model="nomic-embed-text",
        persist_directory="./chroma_db"
    ):
        """Initialize the MarkdownProcessor with embeddings, vector store, and text splitter."""
        self.directory = directory

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=model)

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        # Initialize the Markdown text splitter
        self.text_splitter = MarkdownTextSplitter()

    def read_markdown_files(self):
        """Recursively read all markdown files in the directory."""
        markdown_files = glob.glob(
            os.path.join(self.directory, '**', 'README.md'), recursive=True
        )
        return markdown_files

    def process_markdown_file(self, file_path):
        """Process a single markdown file: split into chunks and store embeddings."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split the content into chunks using MarkdownTextSplitter
        chunks = self.text_splitter.split_text(content)

        docs = [
            Document(page_content=chunk, metadata={"file_path": file_path})
            for chunk in chunks
        ]

        ids = [
            f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))
        ]

        self.vector_store.add_documents(documents=docs, ids=ids)
        print(f"Processed {file_path}")

    def process_all_files(self):
        """Read all markdown files and process them."""
        markdown_files = self.read_markdown_files()

        for file_path in markdown_files:
            self.process_markdown_file(file_path)

if __name__ == "__main__":
    # Specify the directory containing markdown files

    processor_tg = GroundingStore(directory="/Users/D046675/dev-local/TechnologyGuidelines", collection_name="TechnologyGuidelines")
    processor_tg.process_all_files()

    processor_das = GroundingStore(directory="/Users/D046675/dev-local/das-architecture", collection_name="DASArchitecture")
    processor_das.process_all_files()
