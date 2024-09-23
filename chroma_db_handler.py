import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import os
from datetime import datetime
import logging

class ChromaDBHandler:
    def __init__(self,
                 model_name: str = "nomic-embed-text",
                 embedding_url: str = "http://localhost:11434/api/embeddings",
                 db_path: str = "./chroma_db",
                 collection_name: str = "TechnologyGuidelines",
                 allow_reset: bool = True):
        """
        Initializes the ChromaDBHandler with the specified parameters.

        Args:
            model_name (str): The name of the embedding model.
            embedding_url (str): The URL for the embedding service.
            db_path (str): Path to the persistent ChromaDB storage.
            collection_name (str): Name of the collection to use.
            allow_reset (bool): Whether to allow resetting the database.
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize the embedding function
        self.embedding_function = OllamaEmbeddingFunction(
            model_name=model_name,
            url=embedding_url,
        )
        self.logger.info("Initialized OllamaEmbeddingFunction.")

        # Initialize the ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=chromadb.config.Settings(allow_reset=allow_reset)
        )
        self.logger.info(f"Connected to ChromaDB at {db_path}.")

        # Create or get the collection
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            get_or_create=True
        )
        self.logger.info(f"Accessed collection '{collection_name}'.")

    def get_chunks(self, search_term: str, n_results: int = 3):
        """
        Retrieves chunks from the collection based on the search term.

        Args:
            search_term (str): The term to search for in the collection.
            n_results (int): The number of results to retrieve.

        Returns:
            dict: The query results containing the retrieved chunks.
        """
        try:
            self.logger.info(f"Querying for term: '{search_term}' with {n_results} results.")
            results = self.collection.query(
                query_texts=[search_term],
                n_results=n_results
            )
            self.logger.info("Query successful.")
            return results
        except Exception as e:
            self.logger.error(f"An error occurred while querying: {e}")
            return None

    def add_chunk(self, chunk: str, file_path: str, i: int):
        """
        Adds a new chunk to the collection with associated metadata.

        Args:
            chunk (str): The text content to add as a document.
            file_path (str): The file path from which the chunk originates.
            i (int): The index of the chunk (e.g., chunk number).
            embedding_vector (list, optional): Precomputed embedding vector for the chunk.
                If None, the embedding function will generate it automatically.

        Returns:
            bool: True if the addition was successful, False otherwise.
        """
        try:
            doc_id = f"{os.path.basename(file_path)}_chunk_{i}"

            metadata = {
                "file_name": os.path.basename(file_path),
                "timestamp": datetime.now().isoformat()
            }

            add_params = {
                "documents": [chunk],
                "ids": [doc_id],
                "metadatas": [metadata]
            }

            self.logger.info(f"Adding document ID: {doc_id}")
            self.collection.add(**add_params)
            self.logger.info(f"Document '{doc_id}' added successfully.")
            return True
        
        except Exception as e:
            self.logger.error(f"An error occurred while adding a chunk: {e}")
            return False

# Example Usage
if __name__ == "__main__":
    # Instantiate the handler
    chroma_handler = ChromaDBHandler()

    # Example: Retrieving chunks
    search_term = "Metering"
    num_results = 3
    results = chroma_handler.get_chunks(search_term, num_results)
    print("Query Results:", results)

    # Example: Adding a new chunk
    chunk_text = "This is a sample chunk of text related to metering technology."
    file_path = "/path/to/document.txt"
    chunk_index = 1

    success = chroma_handler.add_chunk(chunk=chunk_text, file_path=file_path, i=chunk_index)
    if success:
        print(f"Chunk {chunk_index} added successfully.")
    else:
        print(f"Failed to add chunk {chunk_index}.")
