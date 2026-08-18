from openai import OpenAI
from config import Config

client = OpenAI() #Creates the API client.

#Role - 
    # Act as a Technical Project Manager.
#context-  
    # A Vehicle Insurance project is 75% complete.
    # Development is almost finished, but the payment gateway integration is delayed by 4 days. 
    # Testing has started.
    # Two critical defects are open.
    # Production deployment is planned for 20 August
#Task -
    # Explain project risk to a software developer

#Output requirement
    # Explain in 3 simple bullet points.

prompt = """
    Act as a Technical Project Manager.

    Context:
    A Vehicle Insurance project is 75% complete.
    Development is almost finished, but the payment gateway integration is delayed by 4 days.
    Testing has started.
    Two critical defects are open.
    Production deployment is planned for 20 August.

    Task:
    Summarize the current project status for senior management..

    Output requirement:
    Explain in 3 simple bullet points.
  """
response = client.responses.create( 
    model="gpt-5.6-luna", #selects the model.    
    input=prompt
)


print(response.output_text)