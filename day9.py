from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import Config


# 1. Load document
loader = TextLoader("project_status.txt")
documents = loader.load()

print("Documents:", len(documents))


# 2. Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = text_splitter.split_documents(documents)

print("Chunks:", len(chunks))


# 3. Create embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=Config.OPENAI_API_KEY
)


# 4. Create vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="vehicle_insurance_project"
)

print("Vector store created.")


# 5. Search
query = "Why is the payment gateway delayed?"
query1="What are the current project defects?"
query2="When is the production deployment planned?"

results = vector_store.similarity_search(
    query1,
    k=1
)


# 6. Display retrieved chunks
for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
    print("Metadata:", result.metadata)
    
# results1 = vector_store.similarity_search_with_score(
#     query,
#     k=2
# )

# for doc, score in results1:
#     print("\n--- Result with_score ---")
#     print(doc.page_content)
#     print("Score:", score)