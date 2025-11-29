import sys
import subprocess
import os
from src.state import AgentState
from src.utils.logger import log_event

def executor_node(state: AgentState) -> dict:
    """
    Executes the code and streams output in real-time.
    """
    log_event("EXECUTOR", "Executing training script...")
    
    code = state["python_code"]
    script_path = "temp_train.py"
    
    # write code to file
    with open(script_path, "w") as f:
        f.write(code)
        
    # real time
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1, # Line buffered
        universal_newlines=True
    )

    stdout_buffer = []
    stderr_buffer = []

    # stream stdout to console
    print("\n--- SCRIPT OUTPUT START ---")
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"  {output.strip()}") # Print with indentation
            stdout_buffer.append(output)
            
    # capture any remaining stderr
    stderr_output = process.stderr.read()
    if stderr_output:
        print(f"--- SCRIPT ERROR ---\n{stderr_output}")
        stderr_buffer.append(stderr_output)
    print("--- SCRIPT OUTPUT END ---\n")

    return_code = process.poll()
    
    # combine buffers
    full_stdout = "".join(stdout_buffer)
    full_stderr = "".join(stderr_buffer)
    
    log_event("EXECUTOR", f"Finished with return code: {return_code}")
    
    # sucess check
    if return_code == 0:
        if os.path.exists("submission.csv"):
            log_event("EXECUTOR", "Success! submission.csv found.")
            return {
                "execution_stdout": full_stdout,
                "execution_stderr": full_stderr,
                "success": True,
                "reasoning_trace": ["Execution successful. submission.csv generated."]
            }
        else:
            return {
                "execution_stdout": full_stdout,
                "execution_stderr": "Script finished 0 but 'submission.csv' was not found.",
                "success": False,
                "reasoning_trace": ["Execution ran, but no output file."]
            }
    else:
        return {
            "execution_stdout": full_stdout,
            "execution_stderr": full_stderr,
            "success": False,
            "iteration": state.get("iteration", 0) + 1,
            "reasoning_trace": [f"Execution failed. Error: {full_stderr[:200]}..."]
        }