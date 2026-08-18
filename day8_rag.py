
from langchain_community.document_loaders import TextLoader #Load the document
from langchain_text_splitters import RecursiveCharacterTextSplitter #Add Chunking
from langchain_openai import OpenAIEmbeddings #Generate Embeddings
from langchain_chroma import Chroma #Store the Vectors

from config import Config

# 1. Load document
loader = TextLoader("project_status.txt")
documents = loader.load()
print("Number of documents:", len(documents))
print(documents[0].page_content)
print(documents[0].metadata)

#Now add the Day 4 concept:
#2  Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
    
#3. Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=Config.OPENAI_API_KEY
)

#4. Store chunks + embeddings in vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="vehicle_insurance_project"
)
print("Documents successfully stored in vector database.")