from openai import OpenAI
from config import Config

client = OpenAI() #Creates the API client.

# self.llm = ChatOpenAI(
#             model=Config.CHAT_MODEL,
#             api_key=Config.OPENAI_API_KEY
#         )

response = client.responses.create( 
    model="gpt-5.6-luna", #selects the model.
    #input="You are a Technical Project Management assistant. Explain project risk in 3 simple bullet points."    #is our prompt
    input="Explain project risk to a software developer in 3 simple bullet points."
)
#client.responses.create(...) -->"Send this request to the OpenAI API."

print(response.output_text)