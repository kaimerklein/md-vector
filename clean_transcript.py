

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from IPython.display import display


# LLM from Ollama
# local_model = "llama3.1-70b-4096:latest"
local_model = "llama3.1:latest"
# local_model = "gemma:7b"
# local_model = "phi3.5:latest"
llm = ChatOllama(model=local_model,num_ctx=32768)



# Initialize the Ollama embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

# Open a text file and load the content into one lonf string variable
with open('transcript_acd_walk_through.txt', 'r') as file:
    content = file.read()  





# vector_store = Chroma(
#     collection_name="TechnologyGuidelines",
#     embedding_function=embeddings,
#     persist_directory="./chroma_db",  # Where to save data locally, remove if not necessary
# )

# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from
#     a vector database. By generating multiple perspectives on the user question, your
#     goal is to help the user overcome some of the limitations of the distance-based
#     similarity search. Provide these alternative questions separated by newlines.
#     Original question: {question}""",
# )

# retriever = MultiQueryRetriever.from_llm( 
#     vector_store.as_retriever(), ChatOllama(model=local_model),
#     prompt=QUERY_PROMPT
# )

template = """<INSTRUCTION>
Following is a TEXT transcription of an online meeting.
The purpose of the meeting was that speaker Kai Merklein presented a software architecture document.
Your task is to clean the transcript. There might be wrongly transcribed words. 
Infer the correct words from the surrounding context.
You must not hallucinate.
Work through the text sentence by sentence.
Generate cleansed output for each sentence and each paragraph.
Keep the original length and all level of detail.
</INSTRUCTION>

<TEXT>
{text}
</TEXT>
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"text": RunnablePassthrough()}
    | prompt
    | llm
    # | StrOutputParser()
)



response = chain.invoke(content)



print("\nGenerated Answer:")
display(response.content)

