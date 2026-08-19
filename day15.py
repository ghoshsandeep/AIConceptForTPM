from langchain_community.document_loaders import TextLoader #Load the document
from langchain_text_splitters import RecursiveCharacterTextSplitter #Add Chunking
from langchain_openai import OpenAIEmbeddings #Generate Embeddings
from langchain_chroma import Chroma #Store the Vectors
from config import Config
from langchain_core.prompts import ChatPromptTemplate #Create the Prompt
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough #day 13
from pydantic import BaseModel, Field #day 15

class ProjectRiskResponse(BaseModel):
    answer: str = Field(
        description="Answer to the user's question based only on the provided context."
    )

    risk_level: str = Field(
        description="Risk level based on the provided project information."
    )

    impact: str = Field(
        description="Potential impact on the project."
    )

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

structured_llm = llm.with_structured_output(
    ProjectRiskResponse
)


prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

Do not use information that is not present in the context.

Context:
{context}

Question:
{question}

Return the answer using the required structured format.
""")

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )    

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | structured_llm
)

query = "How long is the payment gateway integration delayed?"
query_2="What impact could the payment gateway delay have?"
query_3="When is the production deployment planned?"

response = rag_chain.invoke(query)

#print("Response:", response.content)
print("Answer:", response.answer)
print("Risk Level:", response.risk_level)
print("Impact:", response.impact)

