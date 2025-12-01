# Alphabench ~inspired by DeepMind

This project implements a self-correcting autonomous agent designed to solve any MLEbench Lite dataset (Image, Text, Audio, Tabular) end-to-end via a single execution command. The agent uses a feedback loop to diagnose and fix its own code errors, maximizing robustness.

## 1. Architecture Overview

The agent is built using a **LangGraph state machine** featuring a critical self-correction loop, allowing for multiple retries based on execution errors (`Executor` feeds errors back to the `Coder`).

![Nodes](./nodes.png)

### Agent Flow Diagram

![Flow Diagram](./flow.jpeg)

### Conditional Branching

![Conditional Branching](./condional_branching.png)

---

## 2. Setup and Dependencies

This project requires Python 3.10+ and the packages listed in `requirements.txt`.

### A. Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone [your-repo-link]
    cd [repo-name]
    ```
2.  **Install Dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

### B. API Key Configuration

Create a file named **`.env`** in the project root directory and add your Google Gemini API key:

```env
# .env file content
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

## 3. Data Preparation

All MLEbench Lite datasets required for evaluation must be downloaded and placed directly inside a local folder named `data/`.

For example, your directory structure should look like this:

## 4. Execution

### 4.1. Run the Full Benchmark (Official Submission)

Use the provided shell script to run the agent across the target competitions, repeating the process three times with different random seeds (10, 20, 30).

**One-Liner Execution:**

```bash
bash run_benchmark.sh
```

Results will be saved in the final_benchmark_results/ directory.

4.2. Single Dataset Test (Development Mode)
To quickly test the agent on a specific dataset:

```bash
python main.py --dataset ./data/mlsp-2013-birds
```

Output files (submission.csv, README.md) will be placed in the root submission/ folder.
