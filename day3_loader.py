from langchain_community.document_loaders import TextLoader

#We're creating the loader.
loader = TextLoader("project_status.txt")

#Read the file and return the loaded document(s).
documents = loader.load()

#print(documents)
print(len(documents))
#print(documents[0].page_content)
#print(documents[0].metadata)