import json
import statistics
import sys
from collections import defaultdict

def calculate_mean_se(report_file):
    """
    Reads the JSONL grading report and calculates the Mean and Standard Error (SE) 
    for the scores of each competition ID.
    """
    scores = defaultdict(list)
    
    try:
        with open(report_file, 'r') as f:
            for line in f:
                # Handle cases where the grading summary file might contain non-JSON output
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                comp_id = data.get('competition_id')
                score = data.get('score')
                metric_name = data.get('metric_name', 'Score')
                
                if comp_id and score is not None:
                    scores[comp_id].append(score)
            
    except FileNotFoundError:
        print(f"❌ Error: Grading report file '{report_file}' not found.")
        sys.exit(1)
        
    results = {}
    for comp_id, score_list in scores.items():
        
        N = len(score_list)
        
        # We need at least 3 successful submissions for the final calculation
        if N < 3:
            print(f"⚠️ Warning: Only {N}/3 submissions found for {comp_id}. Report may be incomplete.")
        
        # Calculate Mean
        mean = statistics.mean(score_list)
        
        # Calculate Standard Error (SE = Sample Standard Deviation / sqrt(N))
        if N > 1:
            try:
                std_dev = statistics.stdev(score_list)
                std_error = std_dev / (N ** 0.5)
            except statistics.StatisticsError:
                # Happens if all scores are identical (std_dev is zero)
                std_error = 0.0
        else:
            # Cannot calculate standard deviation/error with only one sample
            std_error = float('nan') 
            
        results[comp_id] = {
            "mean": mean,
            "se": std_error,
            "metric": metric_name,
            "runs": N
        }
    return results

if __name__ == "__main__":
    
    # We expect the grading output to be in grading_report.jsonl
    report_file_name = 'grading_report.jsonl'
    
    final_scores = calculate_mean_se(report_file_name)
    
    print("\n--- FINAL BENCHMARK REPORT (MEAN ± SE) ---")
    if not final_scores:
        print("No scores calculated. Ensure 'mlebench grade' was run successfully.")
    else:
        for comp, stats in final_scores.items():
            print(f"[{comp}] ({stats['runs']} runs)")
            print(f"  Metric: {stats['metric']}")
            # Use fixed precision for reporting
            se_str = f"{stats['se']:.4f}" if stats['se'] != float('nan') else "N/A"
            print(f"  Score: {stats['mean']:.4f} \u00B1 {se_str}")
    print("------------------------------------------")