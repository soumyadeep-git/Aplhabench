from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import explorer_node, coder_node, executor_node
from src.nodes.reporter import reporter_node

def should_retry(state: AgentState):
    """
    Conditional edge:
    - If success -> Go to Reporter
    - If failed -> Retry (max 3 times)
    - If max retries reached -> End (Fail)
    """
    if state.get("success"):
        return "reporter"
    
    current_iter = state.get("iteration", 0)
    if current_iter < 5:
        return "coder" # go back to fix the code
    
    return "reporter" # give up and report what happened

def create_graph():
    """
    Constructs the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)
    
    # 1. add Nodes
    workflow.add_node("explorer", explorer_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reporter", reporter_node)
    
    # 2. define Edges
    workflow.set_entry_point("explorer")
    
    workflow.add_edge("explorer", "coder")
    workflow.add_edge("coder", "executor")
    
    # conditional logic after execution
    workflow.add_conditional_edges(
        "executor",
        should_retry,
        {
            "coder": "coder",
            "reporter": "reporter"
        }
    )
    
    workflow.add_edge("reporter", END)
    
    # 3. compile
    return workflow.compile()