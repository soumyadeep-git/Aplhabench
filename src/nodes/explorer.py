import json
import os
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.llm import get_llm
from src.tools.filesystem import list_directory_structure, inspect_csv, read_file_content
from src.utils.logger import log_event
from src.prompts import EXPLORER_SYSTEM_PROMPT

def explorer_node(state: AgentState) -> dict:
    """
    1. Scans folder structure.
    2. Reads CSV headers.
    3. Asks LLM to determine Modality & Task.
    """
    dataset_dir = state["dataset_dir"]
    log_event("EXPLORER", f"Analyzing dataset at: {dataset_dir}")
    
    # step 1: evidence
    
    # 1. Look at file structure
    structure = list_directory_structure(dataset_dir)
    
    # 2. Find the main CSV (usually train.csv or similar)
    csv_info = "No CSV found."
    potential_csvs = [
        f for f in os.listdir(dataset_dir) 
        if f.endswith(".csv") and "sample" not in f.lower()
    ]
    
    # Prioritize 'train.csv' if it exists
    main_csv = None
    if "train.csv" in potential_csvs:
        main_csv = os.path.join(dataset_dir, "train.csv")
    elif potential_csvs:
        main_csv = os.path.join(dataset_dir, potential_csvs[0])
        
    if main_csv:
        csv_info = inspect_csv(main_csv)
    
    # 3. Check for README
    readme_text = ""
    if os.path.exists(os.path.join(dataset_dir, "README.md")):
        readme_text = read_file_content(os.path.join(dataset_dir, "README.md"))

    # step 2: ask the brain what to do?
    
    llm = get_llm(temperature=0) # Low temp for factual extraction
    
    user_message = f"""
    DATASET PATH: {dataset_dir}
    
    FILE STRUCTURE:
    {structure}
    
    CSV HEADERS & DATA:
    {csv_info}
    
    README CONTEXT:
    {readme_text}
    """
    
    messages = [
        SystemMessage(content=EXPLORER_SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    
    # step 3: parse
    try:
        # Clean up Markdown code blocks if the LLM adds them
        content = response.content.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(content)
        
        modality = analysis.get("modality", "TABULAR")
        task = analysis.get("task_type", "CLASSIFICATION")
        target = analysis.get("target_column", "target")
        plan = analysis.get("strategy_hint", "Use standard training.")
        
        log_event("EXPLORER", f"Detected: {modality} | {task} | Target: {target}")
        log_event("EXPLORER", f"Strategy: {plan}")
        
        return {
            "file_structure": structure,
            "sample_data": csv_info,
            "readme_content": readme_text,
            "detected_modality": modality,
            "detected_task": task,
            "target_column": target,
            "plan": plan,
            "metadata": analysis, # Store the raw analysis for the coder
            "reasoning_trace": [f"Explorer analysis: Detected {modality} {task}. Plan: {plan}"]
        }
        
    except json.JSONDecodeError:
        log_event("EXPLORER", "Failed to parse LLM JSON output. Defaulting to TABULAR.")
        return {
            "detected_modality": "TABULAR",
            "detected_task": "CLASSIFICATION",
            "plan": "Fallback to Tabular XGBoost",
            "reasoning_trace": ["Explorer failed to parse JSON. Fallback to Tabular."]
        }