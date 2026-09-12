from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import Config
from langchain_core.messages import ToolMessage

@tool
def get_claim_status(claim_number: str):
    """Get the current status of an insurance claim."""

    claims = {
        "CLM1001": "Under Review",
        "CLM1002": "Approved",
        "CLM1003": "Rejected"
    }

    return claims.get(claim_number, "Claim not found")

@tool
def get_policy_details(policy_number: str):
   # """Get the details of an insurance policy."""
    """Get the details and current status of an insurance policy using its policy number"""

    policies = {
        "POL1001": "Comprehensive → Active",
        "POL1002": "Third Party → Active",
        "POL1003": "Comprehensive → Expired"
    }

    return policies.get(policy_number, "Policy not found")

@tool
#def calculate_premium(age: int, vehicle_value: float):
def calculate_premium(vehicle_value: float):
    #"""Calculate the insurance premium based on age and vehicle value."""
    #"""Calculate the insurance premium based on driver's age and vehicle value."""
    """Calculate the insurance premium based on vehicle value."""

    # if age < 25:
    #     base_rate = 0.05
    # else:
    #     base_rate = 0.03

    #premium = vehicle_value * base_rate
    premium = vehicle_value * 0.05
    return f"Calculated premium: ${premium:.2f}"


result = get_claim_status.invoke({
    "claim_number": "CLM1001"
})

result1 = get_policy_details.invoke({
    "policy_number": "POL1001"
})

result2 = calculate_premium.invoke({
    "age": 30,
    "vehicle_value": 20000
})

# print(result)
# print(result1)
# print(result2)

llm = ChatOpenAI(
    model="gpt-5.6-luna",
    api_key=Config.OPENAI_API_KEY,
    reasoning_effort="none"

)
llm_with_tools = llm.bind_tools([
    get_claim_status,
    get_policy_details,
    calculate_premium
])
query1="What is the status of policy POL1001?"
query2="What is the status of claim CLM1002?"
query3="What is the premium for a vehicle worth 800000?"
query4="Check POL1001, tell me whether the policy is active, and calculate the premium for a vehicle worth ₹8 lakh"

response = llm_with_tools.invoke(
   query3
)

#print(response)
print(response.tool_calls)

#tool_call = response.tool_calls[0]
#print(f"Tool called: {tool_call}")

# tool_result = get_claim_status.invoke(
#     tool_call["args"]
# )

# print(tool_result)

# tool_message = ToolMessage(
#     content=str(tool_result),
#     tool_call_id=tool_call["id"]
# )
# final_response = llm_with_tools.invoke([
#     {
#         "role": "user",
#         "content": "What is the status of claim CLM1001?"
#     },
#     response,
#     tool_message
# ])

# print(final_response.content)