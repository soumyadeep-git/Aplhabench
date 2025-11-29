from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.llm import get_llm
from src.prompts import CODER_SYSTEM_PROMPT
from src.utils.logger import log_event

def coder_node(state: AgentState) -> dict:
    """
    Generates Python code based on the analysis and plan.
    If there was an error in the previous run, it tries to fix it.
    """
    log_event("CODER", "Generating training script...")
    
    llm = get_llm(temperature=0.2) # Slight creativity for coding
    
    # check if we are retrying
    error_context = ""
    if state.get("execution_stderr"):
        error_context = f"""
        !!! PREVIOUS CODE FAILED !!!
        ERROR MESSAGE:
        {state['execution_stderr']}
        
        PREVIOUS CODE:
        {state.get('python_code', 'N/A')}
        
        Fix the error and output the corrected code.
        """
    
    # fill the prompt template
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
    
    {error_context}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    response = llm.invoke(messages)
    
    # clean code
    code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {
        "python_code": code,
        "reasoning_trace": ["Coder generated new script."]
    }