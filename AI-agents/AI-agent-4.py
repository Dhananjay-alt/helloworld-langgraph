#Draffter- Boss Orders

# Our company is not working efficiently! We spend way too much time
# drafting documents and this needs to be fixed! ⏱️
# For the company, you need to create an AI Agentic System that can
# speed up drafting documents, emails, etc.
# The AI Agentic System should have Human-AI Collaboration meaning
# the Human should be able to provide continuous feedback and the AI
# Agent should stop when the Human is happy with the draft.
# The system should be fast and have a great user interface.

import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Global varible to store document content
document_content =""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update the document with the provided content."""

    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n {document_content}"

@tool
def save(filename: str)-> str:
    """Save the current document to a text file and finish the process.

    Args: 
        filename: Name for the text file.
    """
    if not document_content.strip():
        return "Cannot save because the document is empty. Please update the document first."

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt" 

    try: 
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e: 
        return f"Error saving the document: {str(e)}"
    
tools = [update, save]

model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents
                                 
    -If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    -If the user wants to save and finish, you need to use the 'save' tool.
    -Make sure to always show the current documnent state after modifications.
    
    The current document content is:{document_content}
    """)

    all_messages = [system_prompt] + list(state["messages"])

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Route from tools node: continue drafting or end after save."""

    messages = state["messages"]

    if not messages:
        return "end"

    last_message = messages[-1]
    if not isinstance(last_message, ToolMessage):
        return "end"

    content = str(last_message.content).lower()
    if "saved" in content and "document" in content:
        return "end"

    return "continue"
    
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""

    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n⚒️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def is_saved(messages):
    if not messages:
        return False

    last_message = messages[-1]
    if isinstance(last_message, ToolMessage):
        content = str(last_message.content).lower()
        return "saved" in content and "document" in content

    return False

def run_document_agent():
    print("\n ==== DRAFTER ====")
    print("Type what to change in the document. Type 'exit' to quit.")

    state = {"messages": []}

    while True:
        user_input = input("\nWhat would you like to do with the document? ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        print(f"\nUSER: {user_input}")
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

        try:
            state = app.invoke(state)
            print_messages(state["messages"])
            if is_saved(state["messages"]):
                break
        except Exception as e:
            print(f"\nError while running Drafter: {e}")

    print("\n ==== DRAFTER FINISHED ====")

if __name__ == "__main__":
    run_document_agent()