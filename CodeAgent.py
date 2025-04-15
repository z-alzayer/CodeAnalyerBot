from typing import Annotated, Literal
from typing_extensions import TypedDict
from pathlib import Path
import os
from langgraph.graph.message import add_messages
from langchain_core.messages.ai import AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from collections.abc import Iterable
from langchain_core.messages.tool import ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, ToolException
from API_Key import GOOGLE_KEY

# Import the RAG tool
from CreateRag import code_analysis_rag  

os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY

class CodeState(TypedDict):
    """State representing the code analysis session"""
    messages: Annotated[list, add_messages]
    current_file: str
    files: list[str]
    analyzed_files: list[str]
    finished: bool

# Core Tool Implementation
@tool
def list_python_files(directory: Annotated[str, "Path to directory"]) -> str:
    """List all Python files in a directory and its subdirectories."""
    try:
        path = Path(directory)
        if not path.exists():
            raise ToolException(f"Directory '{directory}' does not exist")
            
        py_files = [str(file) for file in path.glob('**/*.py') if file.is_file()]
        
        if not py_files:
            return "No Python files found in directory"
            
        return "Python files found:\n- " + "\n- ".join(py_files)
        
    except PermissionError as e:
        raise ToolException(f"Permission denied accessing '{directory}'") from e
    except Exception as e:
        raise ToolException(f"File listing failed: {str(e)}") from e

@tool
def read_file_content(file_path: Annotated[str, "Path to the file to read"]) -> str:
    """Read and return the content of a file."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise ToolException(f"File '{file_path}' does not exist")
            
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        return f"Content of {file_path}:\n\n{content}"
        
    except PermissionError as e:
        raise ToolException(f"Permission denied accessing '{file_path}'") from e
    except UnicodeDecodeError:
        raise ToolException(f"File '{file_path}' could not be decoded as text")
    except Exception as e:
        raise ToolException(f"File reading failed: {str(e)}") from e

# LLM Configuration
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
auto_tools = [list_python_files, read_file_content, code_analysis_rag]
llm_with_tools = llm.bind_tools(auto_tools, tool_choice="auto")

# System Prompts
CODE_ANALYZER_SYSINT = (
    "system",
    "You are CodeAnalyzerBot, an interactive code review system. "
    "Analyze codebases file-by-file with user confirmation at each step.\n\n"
    "Workflow:\n"
    "1. List files with list_python_files\n"
    "2. Select a file to analyze\n"
    "3. Read file content with read_file_content\n"
    "4. Analyze the code and provide feedback\n"
    "5. For deeper analysis of the entire codebase, use code_analysis_rag\n"
    "6. Get user confirmation\n"
    "7. Proceed to next file or revise\n\n"
    "Always wait for explicit confirmation before proceeding.\n"
    "Highlight technical issues, style considerations, and potential improvements.\n"
    "When analyzing code, consider:\n"
    "- Code structure and organization\n"
    "- Potential bugs or edge cases\n"
    "- Performance considerations\n"
    "- Adherence to PEP 8 style guidelines\n"
    "- Documentation quality"
)

WELCOME_MSG = "Code analysis session started. Type 'q' to quit. Please provide the directory path to scan for Python files."

# Graph Nodes
def human_node(state: CodeState) -> CodeState:
    """Handle user interaction with tool output display"""
    last_msg = state["messages"][-1]
    
    if isinstance(last_msg, ToolMessage):
        print("\n[Tool Output]:", last_msg.content)
        
        # If we just listed files, update the files list in state
        if "Python files found:" in last_msg.content:
            files = [line.strip("- ") for line in last_msg.content.split("\n")[1:] if line.startswith("- ")]
            return state | {"messages": [("user", "I see the files. Let's analyze them one by one.")], "files": files}
        
        # If we just read a file, update the current_file in state
        if "Content of " in last_msg.content:
            file_path = last_msg.content.split("Content of ")[1].split(":")[0]
            analyzed_files = state.get("analyzed_files", []) + [file_path]
            return state | {"current_file": file_path, "analyzed_files": analyzed_files}
    else:
        print("\n[Analysis]:", last_msg.content)

    user_input = input("\nUser: ")
    
    if user_input.lower() in {"q", "quit", "exit"}:
        return state | {"finished": True}
    
    return state | {"messages": [("user", user_input)]}

def chatbot_node(state: CodeState) -> CodeState:
    """Generate analysis using LLM with tool support"""
    if not state["messages"]:
        return {"messages": [AIMessage(content=WELCOME_MSG)]}
    
    response = llm_with_tools.invoke([CODE_ANALYZER_SYSINT] + state["messages"])
    return state | {"messages": [response]}

# Routing Logic
def route_after_tools(state: CodeState) -> Literal["human", "__end__"]:
    """Route based on tool execution results"""
    if state.get("finished", False):
        return "__end__"
    return "human"

def route_based_on_last_message(state: CodeState) -> Literal["tools", "human"]:
    """Determine next step based on message content"""
    if not state["messages"]:
        return "human"
    
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "human"

# Graph Construction
graph_builder = StateGraph(CodeState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", ToolNode(auto_tools))

graph_builder.set_entry_point("chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    route_based_on_last_message,
    {"tools": "tools", "human": "human"}
)

graph_builder.add_conditional_edges(
    "human",
    lambda s: "__end__" if s.get("finished") else "chatbot"
)

graph_builder.add_conditional_edges(
    "tools",
    route_after_tools
)

# Final Compilation
analysis_agent = graph_builder.compile()

# Execution Example
if __name__ == "__main__":
    config = {"recursion_limit": 100}
    initial_state = {
        "messages": [], 
        "current_file": "", 
        "files": [], 
        "analyzed_files": [],
        "finished": False
    }
    analysis_agent.invoke(initial_state, config)
