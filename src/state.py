import operator
from typing import Annotated, List, TypedDict, Optional, Any

class AgentState(TypedDict):
    """
    The Single Source of Truth for the Agent's execution.
    Passed between every node in the LangGraph.
    """
    
    #input
    dataset_dir: str          
    
    #perception
    file_structure: str       
    sample_data: str          
    readme_content: str       
    
    #analysis
    detected_modality: str    
    detected_task: str        
    target_column: str        
    
    #action
    plan: str                 
    python_code: str          
    
    #feedback aka the learning stage
    execution_log: str   
    execution_stdout: str
    execution_stderr: str     
    success: bool             
    iteration: Annotated[int, operator.add]          
    
    #logging
    reasoning_trace: Annotated[List[str], operator.add] 