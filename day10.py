from langchain_community.document_loaders import TextLoader #Load the document
from langchain_text_splitters import RecursiveCharacterTextSplitter #Add Chunking
from langchain_openai import OpenAIEmbeddings #Generate Embeddings
from langchain_chroma import Chroma #Store the Vectors

from config import Config
from openai import OpenAI

#Load the document
loader = TextLoader("project_status.txt")
documents = loader.load()

#Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)

#Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=Config.OPENAI_API_KEY
)

#Store chunks + embeddings in vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="vehicle_insurance_project"
)

#Now perform similarity search
query = "Why is the payment gateway delayed?"
results = vector_store.similarity_search(query, k=2)
context = results[0].page_content

# for i, result in enumerate(results):
#     print(f"\n--- Result {i + 1} ---")
#     print(result.page_content)

context = "\n\n".join(
    result.page_content for result in results
)

prompt = f"""
Answwer the user's question using only the provided context.

Context: 
{context}
Question: 
{query}
"""
client = OpenAI() #Creates the API client.
response=client.responses.create(
    model="gpt-5.6-luna", #selects the model.
    input=prompt    
)

print(response.output_text)