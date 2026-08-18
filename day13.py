from langchain_community.document_loaders import TextLoader #Load the document
from langchain_text_splitters import RecursiveCharacterTextSplitter #Add Chunking
from langchain_openai import OpenAIEmbeddings #Generate Embeddings
from langchain_chroma import Chroma #Store the Vectors
from config import Config
from langchain_core.prompts import ChatPromptTemplate #Create the Prompt
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough #day 13


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

llm = ChatOpenAI(
    model="gpt-5.6-luna"
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
""")    

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

#Now perform similarity search
#query = "Why is the payment gateway delayed?"
#query = "How long is the payment gateway integration delayed?"
query="What impact could the payment gateway delay have?"

results = retriever.invoke(query)

context = "\n\n".join(
    result.page_content
    for result in results
)


print("Context:", context)

formatted_prompt = prompt.invoke({
    "context": context,
    "question": query
})

response = llm.invoke(formatted_prompt)
print("Response Content:", response.content)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )