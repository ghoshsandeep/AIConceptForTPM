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

#Create the Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

#Now perform similarity search
query = "Why is the payment gateway delayed?"

results = retriever.invoke(query)


for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
    print("Metadata:", result.metadata)



