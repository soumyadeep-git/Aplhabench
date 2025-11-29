#!/usr/bin/env python3
import argparse
import os
import sys
from src.graph import create_graph

def main():
    # 1. parse arguments
    parser = argparse.ArgumentParser(description="MLEbench Autonomous Agent")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Path to the dataset directory (containing train.csv, images/, etc.)"
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    
    # 2. validation
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist.")
        sys.exit(1)

    print(f"🚀 Starting Autonomous Agent on: {dataset_path}")
    print("---------------------------------------------------")

    # 3. setup environment
    os.makedirs("submission", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 4. initialize state
    initial_state = {
        "dataset_dir": dataset_path,
        "iteration": 0,
        "success": False,
        "reasoning_trace": []
    }

    # 5. build and run graph
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
        sys.exit(1)

if __name__ == "__main__":
    main()