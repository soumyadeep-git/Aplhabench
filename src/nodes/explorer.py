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
    
    # --- STEP 1: GATHER EVIDENCE ---
    
    # 1. Look at file structure
    structure = list_directory_structure(dataset_dir)
    structure_lower = structure.lower()
    
    # 2. Find the main CSV
    csv_info = "No CSV found."
    potential_csvs = [
        f for f in os.listdir(dataset_dir) 
        if f.endswith(".csv") and "sample" not in f.lower()
    ]
    
    main_csv = None
    if "train.csv" in potential_csvs:
        main_csv = os.path.join(dataset_dir, "train.csv")
    elif potential_csvs:
        main_csv = os.path.join(dataset_dir, potential_csvs[0])
        
    if main_csv:
        csv_info = inspect_csv(main_csv)
    
    # 3. Check for README
    readme_text = ""
    readme_path = os.path.join(dataset_dir, "README.md")
    if os.path.exists(readme_path):
        readme_text = read_file_content(readme_path)

    # --- STEP 2: CONSULT THE LLM ---
    
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
    
    # --- STEP 3: PARSE AND REFINE ---
    try:
        # Clean up Markdown code blocks if the LLM adds them
        content = response.content.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(content)
        
        # Extract fields with safe defaults
        modality = analysis.get("modality", "TABULAR").upper()
        task = analysis.get("task_type", "CLASSIFICATION").upper()
        target = analysis.get("target_column", "target")
        plan = analysis.get("strategy_hint", "Use standard training fallback.")
        
        # --- HEURISTIC OVERRIDE (Safety Net) ---
        # If the LLM says TABULAR but we see clear Audio/Image signals, override it.
        # This helps with datasets like ICML Whale where CSVs look tabular but data is external.
        
        # Check for Audio
        if modality == "TABULAR" and any(x in structure_lower for x in ['.wav', '.mp3', '.aiff', '.flac']):
            modality = "AUDIO"
            plan = "Detected audio files. Switch to MelSpectrogram + ResNet18."
            log_event("EXPLORER", "Override: Switch TABULAR -> AUDIO based on file extensions.")
            
        # Check for Images
        elif modality == "TABULAR" and any(x in structure_lower for x in ['.jpg', '.png', '.jpeg', 'images/']):
            modality = "IMAGE"
            plan = "Detected images. Switch to ResNet18."
            log_event("EXPLORER", "Override: Switch TABULAR -> IMAGE based on file extensions.")

        log_event("EXPLORER", f"Final Detection: {modality} | {task} | Target: {target}")
        
        return {
            "file_structure": structure,
            "sample_data": csv_info,
            "readme_content": readme_text,
            "detected_modality": modality,  # Matches AgentState
            "detected_task": task,          # Matches AgentState
            "target_column": target,
            "plan": plan,
            "metadata": analysis,
            "reasoning_trace": [f"Explorer analysis: Detected {modality} {task}. Plan: {plan}"]
        }
        
    except json.JSONDecodeError:
        log_event("EXPLORER", "Failed to parse LLM JSON output. Applying emergency fallback.")
        
        # Emergency Fallback based on simple string matching
        fallback_modality = "TABULAR"
        if ".wav" in structure_lower or ".aiff" in structure_lower:
            fallback_modality = "AUDIO"
        elif ".jpg" in structure_lower or ".png" in structure_lower:
            fallback_modality = "IMAGE"
            
        return {
            "detected_modality": fallback_modality,
            "detected_task": "CLASSIFICATION",
            "target_column": "target", 
            "plan": "JSON Parse Failed. Using Emergency Fallback.",
            "reasoning_trace": ["Explorer failed to parse JSON. Hard fallback to safe defaults."]
        }