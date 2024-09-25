

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from IPython.display import display


# LLM from Ollama
# local_model = "llama3.1-70b-4096:latest"
# local_model = "llama3.1:latest"
# local_model = "gemma:7b"
local_model = "phi3.5:latest"
llm = ChatOllama(model=local_model,num_ctx=32768)



# Initialize the Ollama embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vector_store = Chroma(
    collection_name="TechnologyGuidelines",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # Where to save data locally, remove if not necessary
)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm( 
    vector_store.as_retriever(), ChatOllama(model=local_model),
    prompt=QUERY_PROMPT
)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    # | StrOutputParser()
)

questions = '''Explain the concepts of Metering'''

# Get the retrieved documents
retrieved_docs = retriever.get_relevant_documents(questions)

# Display the retrieved documents
print("Retrieved Documents:")
for idx, doc in enumerate(retrieved_docs, start=1):
    print(f"\nDocument {idx}:")
    print(f"Metadata: {doc.metadata}")
    # print(f"Content:\n{doc.page_content}")
    print('-' * 80)

response = chain.invoke(questions)



print("\nGenerated Answer:")
display(response.content)

