from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import get_llm
from src.prompts import CODER_SYSTEM_PROMPT
from src.utils.logger import log_event

def coder_node(state: AgentState) -> dict:
    # check iteration count. If > 0, we are definitely retrying.
    iteration = state.get("iteration", 0)
    previous_error = state.get("execution_stderr", "")
    
    # we are retrying if we have an error OR if iteration > 0
    is_retry = iteration > 0 or bool(previous_error)
    
    log_event("CODER", f"Generating training script... (Retry: {is_retry}, Iteration: {iteration})")
    
    # boost creativity if we are scodtuck in a loop
    temp = 0.4 if is_retry else 0.1
    llm = get_llm(temperature=temp)
    
    system_prompt = CODER_SYSTEM_PROMPT.format(
        modality=state["detected_modality"],
        task=state["detected_task"],
        plan=state["plan"],
        dataset_dir=state["dataset_dir"]
    )
    
    user_content = f"""
    DATA SUMMARY:
    {state.get('sample_data', 'N/A')}
    
    METADATA:
    {state.get('metadata', {})}
    """
    
    if is_retry:
        user_content += f"""
        \n!!! CRITICAL: PREVIOUS CODE FAILED (Attempt {iteration}) !!!
        
        ERROR MESSAGE:
        {previous_error}
        
        INSTRUCTIONS:
        1. The previous code crashed. DO NOT generate the same code.
        2. Check imports carefully. (e.g., AdamW is in torch.optim, NOT transformers).
        3. Fix the specific error shown above.
        """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    response = llm.invoke(messages)
    code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {
        "python_code": code,
        "reasoning_trace": [f"Coder generated script (Retry={is_retry})"]
    }