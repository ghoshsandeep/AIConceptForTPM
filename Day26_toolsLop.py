from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import Config


# --------------------------------------------------
# 1. Define Tools
# --------------------------------------------------

@tool
def get_policy_details(policy_number: str):
    """Get the details and current status of an insurance policy using its policy number."""

    policies = {
        "POL1001": {
            "policy_type": "Comprehensive",
            "status": "Active"
        },
        "POL1002": {
            "policy_type": "Third Party",
            "status": "Active"
        },
        "POL1003": {
            "policy_type": "Comprehensive",
            "status": "Expired"
        }
    }

    return policies.get(
        policy_number,
        "Policy not found"
    )


@tool
def get_claim_status(claim_number: str):
    """Get the current status of an insurance claim using its claim number."""

    claims = {
        "CLM1001": "Under Review",
        "CLM1002": "Approved",
        "CLM1003": "Rejected"
    }

    return claims.get(
        claim_number,
        "Claim not found"
    )


@tool
def calculate_premium(vehicle_value: float):
    """Calculate the insurance premium based on vehicle value."""

    premium = vehicle_value * 0.05

    return f"Calculated premium: ${premium:.2f}"


# --------------------------------------------------
# 2. Create LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-5.6-luna",
    api_key=Config.OPENAI_API_KEY,
    reasoning_effort="none"
)


# --------------------------------------------------
# 3. Bind Tools
# --------------------------------------------------

tools = [
    get_policy_details,
    get_claim_status,
    calculate_premium
]

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# 4. Create Tool Lookup
# --------------------------------------------------

tool_map = {
    tool.name: tool
    for tool in tools
}


# --------------------------------------------------
# 5. User Message
# --------------------------------------------------

messages = [
    HumanMessage(
        content="Check policy POL1001 and tell me whether it is active.Also calculate the premium for a vehicle worth 800000."
    )
]

# Loop through the conversation until the LLM indicates that it has completed its response
while True:
    # --------------------------------------------------
    # 6. LLM Call
    # --------------------------------------------------

    response = llm_with_tools.invoke(messages)

    #print("\n--- LLM RESPONSE ---")
    #print(response.content)

    # --------------------------------------------------
    # 7. Add AI response to conversation
    # --------------------------------------------------

    messages.append(response)

    # Check if the LLM has indicated that it has completed its response
    if not response.tool_calls:
        break

    # --------------------------------------------------
    # 8. Execute requested tools
    # --------------------------------------------------

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        #print("\nTool requested:", tool_name)
        #print("Tool arguments:", tool_args)

        # Find actual tool
        tool = tool_map[tool_name]

        # Execute tool
        tool_result = tool.invoke(tool_args)

        #print("Tool result:", tool_result)

        # Create ToolMessage
        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call_id
        )

        # Add result to conversation
        messages.append(tool_message)



print("\n\n--- FINAL RESPONSE ---")
print(response.content)