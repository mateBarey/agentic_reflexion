import os
import json
import getpass
import warnings
from typing import List, Dict, Annotated, Optional, TypedDict, Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
# FIX: Updated Tavily import to suppress deprecation warning (or install langchain-tavily)
try:
    from langchain_tavily import TavilySearch
    tavily_tool = TavilySearch(max_results=3)
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    warnings.filterwarnings("ignore", message=".*TavilySearchResults.*deprecated.*")
    tavily_tool = TavilySearchResults(max_results=3)

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START

# Load Env
load_dotenv()

# -----------------------------------------------------------------------------
# 1. LLM & State Setup
# -----------------------------------------------------------------------------

openai_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=os.getenv("OPENAI_API_KEY"),
)

class State(TypedDict):
    # FIX: Annotated ensures messages are appended correctly
    messages: Annotated[List[BaseMessage], add_messages]

def _set_if_undefined(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var)

_set_if_undefined("TAVILY_API_KEY")

# -----------------------------------------------------------------------------
# 2. Pydantic Data Models (Schema Enforcement)
# -----------------------------------------------------------------------------

class Reflection(BaseModel):
    missing: str = Field(description="What information is missing")
    superfluous: str = Field(description="What information is unnecessary")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="Main response to the question")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: List[str] = Field(description="Queries for additional research")

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""
    references: List[str] = Field(description="Citations motivating your updated answer.")

# -----------------------------------------------------------------------------
# 3. Validation Wrapper (From Reference Docs)
# -----------------------------------------------------------------------------
# FIX: Added ResponderWithRetries to handle schema validation errors gracefully

class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: State):
        response = None
        # Retry logic for schema validation
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state["messages"]}, 
                {"tags": [f"attempt:{attempt}"]}
            )
            try:
                # Validate the output matches the Pydantic schema
                self.validator.invoke(response)
                # FIX: Must return dict {"messages": ...} for LangGraph
                return {"messages": [response]}
            except ValidationError as e:
                # Append error feedback to state and retry
                state_messages = state["messages"] + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\nPay close attention to the function schema.",
                        tool_call_id=response.tool_calls[0]["id"] if response.tool_calls else "unknown",
                    ),
                ]
                # Update state for next iteration
                state = {"messages": state_messages}
        # If all attempts fail, return empty to avoid crash
        return {"messages": []}

# -----------------------------------------------------------------------------
# 4. Prompts & Chains
# -----------------------------------------------------------------------------

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Dr. Paul Saladino, "Carnivore MD," advocating for animal-based nutrition and challenging plant-centric dietary dogma. Focus on the superior bioavailability of animal foods and the potential toxicity of plant compounds such as oxalates, lectins, and phytates.
Your response must follow these steps:
1. {first_instruction}
2. Present the evolutionary and biochemical rationale for animal-based nutrition, emphasizing nutrient density, bioavailability, and the absence of plant antinutrients.
3. Challenge conventional "plants are healthy" narratives with mechanistic evidence about plant defense compounds and their impact on human physiology.
4. Reflect and critique your answer. Be rigorous about evolutionary consistency and nutrient science.
5. After the reflection, **list 1-3 search queries separately** for researching antinutrients, bioavailability studies, or evolutionary nutrition. Do not include them inside the reflection.
Focus on: organ meats as superfoods, the carnivore elimination protocol, and why "fiber is unnecessary" based on human evolutionary biology.
"""
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Answer the user's question above using the required format, emphasizing the superiority of animal-based nutrition."
    ),
])

# Initial Responder Chain
first_responder_prompt = prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
initial_chain = first_responder_prompt | openai_llm.bind_tools(tools=[AnswerQuestion])
validator = PydanticToolsParser(tools=[AnswerQuestion])
first_responder = ResponderWithRetries(runnable=initial_chain, validator=validator)

# Revisor Chain
revise_instructions = """Revise your previous answer using the new information, applying the rigor and evidence-based approach of Dr. David Attia.
- Incorporate the previous critique to add clinically relevant information, focusing on mechanistic understanding and individual variability.
- You MUST include numerical citations referencing peer-reviewed research, randomized controlled trials, or meta-analyses to ensure medical accuracy.
- Distinguish between correlation and causation, and acknowledge limitations in current research.
- Address potential biomarker considerations (lipid panels, inflammatory markers, and so on) when relevant.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit) in the form of:
- [1] https://example.com
- [2] https://example.com
- Use the previous critique to remove speculation and ensure claims are supported by high-quality evidence. Keep response under 250 words with precision over volume.
- When discussing nutritional interventions, consider metabolic flexibility, insulin sensitivity, and individual response variability.
"""
revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)
revisor_chain = revisor_prompt | openai_llm.bind_tools(tools=[ReviseAnswer])
revision_validator = PydanticToolsParser(tools=[ReviseAnswer])
revisor = ResponderWithRetries(runnable=revisor_chain, validator=revision_validator)

# -----------------------------------------------------------------------------
# 5. Tool Execution Node
# -----------------------------------------------------------------------------

# FIX: Function must return {"messages": ...} to update State correctly
def execute_tools(state: State) -> Dict:
    messages = state["messages"]
    
    # Find the last AI message that has tool_calls
    last_ai_with_tools = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_ai_with_tools = msg
            break
    
    if not last_ai_with_tools:
        return {"messages": []}
    
    tool_messages = []
    for tool_call in last_ai_with_tools.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])
            query_results = {}
            
            for query in search_queries:
                try:
                    result = tavily_tool.invoke(query)
                    query_results[query] = result
                except Exception as e:
                    query_results[query] = f"Error: {str(e)}"
            
            tool_messages.append(ToolMessage(
                content=json.dumps(query_results),
                tool_call_id=call_id
            ))
    
    # FIX: Return dict instead of List
    return {"messages": tool_messages}

# -----------------------------------------------------------------------------
# 6. Graph Logic & Event Loop
# -----------------------------------------------------------------------------

MAX_ITERATIONS = 4

# FIX: Correct iteration counting logic (consecutive AI/Tool pairs from end)
def event_loop(state: State) -> Literal["execute_tools", END]:
    messages = state["messages"]
    i = 0
    # Count backwards from the end of the conversation
    for msg in reversed(messages):
        if msg.type not in {"tool", "ai"}:
            break
        i += 1
    # Each iteration consists of 1 AI message + 1 Tool message
    num_iterations = i // 2
    if num_iterations >= MAX_ITERATIONS:
        return END
    return "execute_tools"

# Build Graph
builder = StateGraph(State)

# Add Nodes
builder.add_node("respond", first_responder.respond)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revisor", revisor.respond)

# Add Edges
builder.add_edge(START, "respond")
builder.add_edge("respond", "execute_tools")
builder.add_edge("execute_tools", "revisor")

# Conditional Edge for Loop
builder.add_conditional_edges("revisor", event_loop, ["execute_tools", END])

graph = builder.compile()

# -----------------------------------------------------------------------------
# 7. Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    question = """I'm pre-diabetic and need to lower my blood sugar, and I have heart issues.
    What breakfast foods should I eat and avoid"""
    
    messages = {"messages": [HumanMessage(content=question)]}
    
    print(f"--- Starting Reflection Agent for: '{question}' ---\n")
    
    try:
        responses = graph.invoke(messages, config={"recursion_limit": 10})
        
        # Print Results
        print("--- Final Conversation History ---")
        for msg in responses["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"\n[Human]: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    print(f"\n[AI]: (Tool Call: {msg.tool_calls[0]['name']})")
                else:
                    print(f"\n[AI]: {msg.content}")
            elif isinstance(msg, ToolMessage):
                print(f"\n[Tool]: {len(msg.content)} chars of search data")
                
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        import traceback
        traceback.print_exc()