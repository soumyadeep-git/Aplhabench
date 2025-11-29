import os
import pandas as pd
from typing import Dict, Any

def list_directory_structure(path: str, max_depth: int = 2) -> str:
    """
    Recursively lists the file structure of a directory up to max_depth.
    Useful to see if there are folders like 'train/', 'test/', 'images/', etc.
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."

    structure = []
    path = os.path.abspath(path)
    base_depth = path.count(os.sep)

    for root, dirs, files in os.walk(path):
        current_depth = root.count(os.sep) - base_depth
        if current_depth >= max_depth:
            continue
            
        indent = "  " * current_depth
        folder_name = os.path.basename(root)
        structure.append(f"{indent}[DIR] {folder_name}/")
        
        # List first 5 files as samples (avoid flooding context with 10k image names)
        for i, file in enumerate(files):
            if i < 5:
                structure.append(f"{indent}  {file}")
            else:
                structure.append(f"{indent}  ... ({len(files) - 5} more files)")
                break
                
    return "\n".join(structure)

def inspect_csv(file_path: str, n_rows: int = 5) -> str:
    """
    Reads the first n rows of a CSV file to understand columns and data types.
    Handles encoding errors gracefully.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."

    try:
        #standard reading
        df = pd.read_csv(file_path, nrows=n_rows)
    except Exception:
        try:
            # Fallback for weird encodings often found in Kaggle datasets
            df = pd.read_csv(file_path, nrows=n_rows, encoding='latin1')
        except Exception as e:
            return f"Error reading CSV: {str(e)}"

    info = []
    info.append(f"File: {os.path.basename(file_path)}")
    info.append(f"Columns: {list(df.columns)}")
    info.append("Sample Data:")
    info.append(df.to_string(index=False))
    
    return "\n".join(info)

def read_file_content(file_path: str, max_chars: int = 2000) -> str:
    """Reads a text file (like README.md) to get context."""
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(max_chars)
    except Exception as e:
        return f"Error reading file: {str(e)}"