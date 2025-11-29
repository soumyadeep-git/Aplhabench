import sys
import subprocess
import os
from src.state import AgentState
from src.utils.logger import log_event

def executor_node(state: AgentState) -> dict:
    """
    Writes the generated code to a file and executes it.
    Captures stdout/stderr to feed back into the loop if needed.
    """
    log_event("EXECUTOR", "Executing training script...")
    
    code = state["python_code"]
    
    # writing code to file
    script_path = "temp_train.py"
    with open(script_path, "w") as f:
        f.write(code)
        
    # executing
    try:
        # run with a timeout of 24 hours, but practically 
        # we want it faster. Using sys.executable to ensure we use the same venv.
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600 * 24 
        )
        
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
        
        log_event("EXECUTOR", f"Finished with return code: {return_code}")
        
        if return_code == 0:
            # check if submission.csv was actually created
            if os.path.exists("submission.csv"):
                log_event("EXECUTOR", "Success! submission.csv found.")
                return {
                    "execution_stdout": stdout,
                    "execution_stderr": "",
                    "success": True,
                    "reasoning_trace": ["Execution successful. submission.csv generated."]
                }
            else:
                log_event("EXECUTOR", "Script ran but no submission.csv found.")
                return {
                    "execution_stdout": stdout,
                    "execution_stderr": "Script finished 0 but 'submission.csv' was not found. Ensure you save the file.",
                    "success": False,
                    "reasoning_trace": ["Execution ran, but no output file."]
                }
        else:
            log_event("EXECUTOR", "Execution failed.")
            return {
                "execution_stdout": stdout,
                "execution_stderr": stderr,
                "success": False,
                "iteration": state.get("iteration", 0) + 1,
                "reasoning_trace": [f"Execution failed. Error: {stderr[:200]}..."]
            }

    except subprocess.TimeoutExpired:
        return {
            "execution_stderr": "Execution timed out.",
            "success": False,
            "iteration": state.get("iteration", 0) + 1,
            "reasoning_trace": ["Execution timed out."]
        }
    except Exception as e:
        return {
            "execution_stderr": str(e),
            "success": False,
            "iteration": state.get("iteration", 0) + 1,
            "reasoning_trace": [f"Execution error: {str(e)}"]
        }