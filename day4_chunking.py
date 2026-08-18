from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("project_status.txt")
documents = loader.load()
#print(len(documents))
documentContent=documents[0].page_content

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
    )
#chunkscontents=splitter.split_text(documentContent)
#print(chunkscontents)
chunksDocument=splitter.split_documents(documents)

for i, chunk in enumerate(chunksDocument):
    print(f"\n--- Chunk {i+1} ---")
    #print(chunk.page_content)
    print(chunk.page_content)


