import argparse
import os
import sys
import shutil
from src.graph import create_graph

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="MLEbench Autonomous Agent")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Path to the dataset directory (containing train.csv, images/, etc.)"
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    
    # 2. Validation
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist.")
        sys.exit(1)

    print(f"🚀 Starting Autonomous Agent on: {dataset_path}")
    print("---------------------------------------------------")

    # 3. Setup Environment
    os.makedirs("submission", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    
    # delete old submission files so we don't get false positives
    # if the current run fails but an old file exists.
    if os.path.exists("submission/submission.csv"):
        print("🗑️  Cleaning up previous submission.csv...")
        os.remove("submission/submission.csv")
    
    # Also clean up previous temp training scripts
    if os.path.exists("temp_train.py"):
        os.remove("temp_train.py")
    

    # 4. Initialize State
    initial_state = {
        "dataset_dir": dataset_path,
        "iteration": 0,
        "success": False,
        "reasoning_trace": []
    }

    # 5. Build and Run Graph
    try:
        app = create_graph()
        final_state = app.invoke(initial_state)
        
        print("---------------------------------------------------")
        if final_state.get("success"):
            print("✅ Mission Complete!")
            print(f"📁 Submission: {os.path.abspath('submission/submission.csv')}")
            print(f"📄 README:     {os.path.abspath('submission/README.md')}")
        else:
            print("⚠️ Mission Failed (Max retries reached).")
            print("Check logs/ for details.")
            
    except Exception as e:
        print(f"❌ Critical System Error: {str(e)}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()