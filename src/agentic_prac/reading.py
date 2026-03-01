from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Literal


class WeatherSchema(BaseModel):
    condition: str = Field(description="Weather condition such as sunny, rainy, cloudy")
    temperature: int = Field(description="Temperature value")
    unit: str = Field(description="Temperature unit such as fahrenheit or celsius")


# Create an LLM instance
llm = ChatOpenAI(model="gpt-4.1-nano")  # or your preferred model

weather_llm = llm.bind_tools(tools=[WeatherSchema])
response = weather_llm.invoke("It's sunny and 75 degrees")
# Returns: {"condition": "sunny", "temperature": 75, "unit": "fahrenheit"}


'''
In a real-world example, we might ask a language model to book a flight, where the schema could include fields like destination, starting point, time, and date—structured data that would then be passed to an API. In this simplified example, we define a Pydantic schema (Add) to describe the expected input structure (two integers) for an addition task and use it as a tool for a language model (ChatOpenAI) to extract structured data from a natural language query. When a user says something like "add 1 and 10", the LLM interprets the request using the Add schema, extracts the numbers, and the code performs the actual addition and prints the result.
'''


# Define BaseModel class for addition
class Add(BaseModel):
    """Add two numbers together"""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

# Setup LLM and bind the Add tool
llm = ChatOpenAI(model="gpt-4.1-nano")
initial_chain = llm.bind_tools(tools=[Add])

# Ask LLM to add numbers
question = "add 1 and 10"
response = initial_chain.invoke([HumanMessage(content=question)])

# Extract and calculate from the LLM response
def extract_and_add(response):
    tool_call = response.tool_calls[0]
    a = tool_call["args"]['a']
    b = tool_call["args"]['b']
    return a + b

# Execute and print results
result = extract_and_add(response)
print(f"LLM extracted: a={response.tool_calls[0]['args']['a']}, b={response.tool_calls[0]['args']['b']}")
print(f"Result: {result}")

"""
DEFININIG RESUABLE MATH TOOLS WITH SCHEMAS 
"""


class TwoOperands(BaseModel):
    a: float
    b: float

class AddInput(TwoOperands):
    operation: Literal['add']

class SubtractInput(TwoOperands):
    operation: Literal['subtract']

class MathOutput(BaseModel):
    result: float

def add_tool(data: AddInput) -> MathOutput:
    return MathOutput(result=data.a + data.b)

def subtract_tool(data: SubtractInput) -> MathOutput:
    return MathOutput(result=data.a - data.b)


incoming_json = '{"a": 7, "b": 3, "operation": "subtract"}'

def dispatch_tool(json_payload: str) -> str:
    base = SubtractInput.model_validate(json_payload)
    if base.operation == "add":
        output = add_tool(AddInput.model_validate(json_payload))
    elif base.operation == "subtract":
        output = subtract_tool(SubtractInput.model_validate(json_payload))
    else:
        raise ValueError("Unsupported operation")
    return output.model_dump_json()

result_json = dispatch_tool(incoming_json)
print(result_json)  # {"result": 4.0}

"""
WHAT DOES LITERAL DO ?
"""
# Define a schema with Literal to restrict operation types
class CalculatorSchema(BaseModel):
    operation: Literal['add', 'subtract', 'multiply', 'divide'] = Field(
        description="The mathematical operation to perform"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    
calculator_llm = llm.bind_tools(tools=[CalculatorSchema])
# Test with valid operations
response1 = calculator_llm.invoke("Add 15 and 23")
print(response1.tool_calls[0]['args'])
# Output: {"operation": "add", "a": 15.0, "b": 23.0}
response2 = calculator_llm.invoke("Multiply 7 by 8")
print(response2.tool_calls[0]['args'])
# Output: {"operation": "multiply", "a": 7.0, "b": 8.0}
