import os
import shutil
from langchain_core.messages import HumanMessage
from src.state import AgentState
from src.llm import get_llm
from src.utils.logger import log_event

def reporter_node(state: AgentState) -> dict:
    """
    Final node:
    1. Moves submission.csv to the submission/ folder.
    2. Writes the README.md.
    3. Saves the reasoning trace.
    """
    log_event("REPORTER", "Generating final report...")
    
    # 1. cleanup
    # move submission.csv from root to submission/ folder
    if os.path.exists("submission.csv"):
        shutil.move("submission.csv", "submission/submission.csv")
        log_event("REPORTER", "Moved submission.csv to submission/ folder.")
    
    # 2. generate readme
    llm = get_llm()
    
    prompt = f"""
    Write a README.md (strictly < 200 words) for this MLEbench submission.
    
    CONTEXT:
    - Task: {state.get('detected_modality', 'Unknown')}
    - Strategy: {state.get('plan', 'Unknown')}
    - Execution Status: {'Success' if state.get('success') else 'Failed'}
    
    REQUIREMENTS:
    1. Explain how you understood the task & modality.
    2. Explain why you chose that strategy.
    3. Self-Reflection: Identify one specific improvement (e.g., "Use 5-fold CV", "Use Optuna") you would make next time.
    
    Do not use markdown code blocks. Just raw text.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    with open("submission/README.md", "w") as f:
        f.write(response.content)
        
    # 3. save trace
    trace_log = "\n".join(state.get("reasoning_trace", []))
    with open("submission/reasoning_trace.txt", "w") as f:
        f.write(trace_log)
        
    log_event("REPORTER", "Submission files generated successfully.")
    
    return {"reasoning_trace": ["Final report generated."]}