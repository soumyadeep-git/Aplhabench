# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
# All datasets are assumed to be located directly inside the local 'data' folder
DATA_ROOT="./data"
EVAL_RESULTS_ROOT="final_benchmark_results"
SEEDS=(10 20 30)

# Final list of verified, executable datasets (substitutions included)
COMPETITIONS=(
    "plant-pathology-2020-fgvc7"          # Substitutes siim-isic
    "spooky-author-identification"
    "tabular-playground-series-may-2022"
    "text-normalization-challenge-english-language"
    "mlsp-2013-birds"                     # Substitutes right-whale-redux
)

# --- Setup ---
mkdir -p $EVAL_RESULTS_ROOT
echo "Starting multi-seed evaluation on 5 verified MLEbench Lite tasks..."

# --- Main Evaluation Loop ---
for COMP_ID in "${COMPETITIONS[@]}"; do
    COMP_DIR="$DATA_ROOT/$COMP_ID"
    echo "========================================================="
    echo "🚀 Running Competition: $COMP_ID"
    echo "========================================================="
    
    # Check if data path exists (mandatory for agent input)
    if [ ! -d "$COMP_DIR" ]; then
        echo "❌ ERROR: Agent input path not found for $COMP_ID at $COMP_DIR. Skipping."
        continue
    fi
    
    COMP_RESULT_DIR="$EVAL_RESULTS_ROOT/$COMP_ID"
    mkdir -p $COMP_RESULT_DIR

    for SEED in "${SEEDS[@]}"; do
        RUN_DIR="$COMP_RESULT_DIR/seed_$SEED"
        mkdir -p $RUN_DIR

        echo "--- Running $COMP_ID with SEED $SEED ---"
        
        # CRITICAL: Set the environment variable for the agent's script to read
        export AGENT_SEED=$SEED 
        
        # Run the autonomous agent
        python main.py --dataset "$COMP_DIR" 
        
        # Check if the agent succeeded (submission.csv created in 'submission' folder)
        if [ -f "submission/submission.csv" ]; then
            # Move deliverables to unique location
            mv submission/submission.csv "$RUN_DIR/"
            mv submission/README.md "$RUN_DIR/"
            echo "✅ Run successful. Files saved to $RUN_DIR"
        else
            # Save the final error log for debugging if submission failed
            LATEST_LOG=$(ls -t logs/agent_run_*.log 2>/dev/null | head -n 1)
            if [ -n "$LATEST_LOG" ]; then
                cp "$LATEST_LOG" "$RUN_DIR/failure_log.log"
            fi
            echo "⚠️ Agent failed to produce submission.csv for seed $SEED. Log saved in $RUN_DIR"
        fi
        
        # Cleanup any final artifacts missed by main.py's cleanup
        rm -rf temp_train.py logs/agent_run_*
    done
done

echo "========================================================="
echo "✅ Benchmark run complete. Total results saved to $EVAL_RESULTS_ROOT"
echo "Next Step: 1. Grade these files using 'mlebench grade'. 2. Calculate Mean/SE."
echo "========================================================="
--- END OF run_benchmark.sh ---