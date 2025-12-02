import sys
import subprocess
import time
import matplotlib.pyplot as plt
import re
import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SOLVERS = [
    {"name": "Branch & Bound (C++)", "script": "./wcnf_bnb", "color": "#FF6B6B", "type": "binary"},
    {"name": "ILP (PuLP/CBC)", "script": "ilp.py", "color": "#4ECDC4", "type": "python"},
    {"name": "RC2 (PySAT)", "script": "rc2_pysat.py", "color": "#45B7D1", "type": "python"}
]

TIMEOUT = 60  # Seconds per solver per file
MAX_WORKERS = os.cpu_count() or 4  # Concurrency limit

def compile_cpp():
    print("--- Compiling C++ Solver ---")
    if not os.path.exists("wcnf_bnb.cpp"):
        print("Error: wcnf_bnb.cpp not found.")
        return False
        
    try:
        cmd = ["g++", "-O3", "-o", "wcnf_bnb", "wcnf_bnb.cpp"]
        subprocess.check_call(cmd)
        print("Compilation successful.\n")
        return True
    except subprocess.CalledProcessError:
        print("Compilation failed.\n")
        return False
    except FileNotFoundError:
        print("Error: g++ compiler not found.\n")
        return False

def parse_output(output, solver_name):
    patterns = [
        r"Best Cost: (\d+)",
        r"Optimal Cost.*: (\d+)",
        r"Total Relaxed Cost: (\d+)",
        r"Weight: (\d+)"
    ]
    
    cost = None
    for p in patterns:
        all_matches = re.findall(p, output)
        if all_matches:
            cost = int(all_matches[-1])
            
    return cost

def run_single_benchmark(input_file, solver):
    """
    Runs a single solver on a single file. Returns a dict of results.
    """
    script_path = solver['script']
    res = {
        "file": os.path.basename(input_file),
        "solver": solver['name'],
        "time": TIMEOUT,
        "cost": None,
        "status": "Failed"
    }

    if not os.path.exists(script_path):
        return res

    cmd = []
    if solver['type'] == "python":
        cmd = [sys.executable, script_path, input_file]
    else:
        cmd = [script_path, input_file, "--timeout", str(TIMEOUT)]

    start_time = time.time()
    try:
        # Buffer timeout to let internal logic print best found solution
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT + 2 
        )
        elapsed = time.time() - start_time
        output = result.stdout
        
        cost = parse_output(output, solver['name'])
        
        if cost is not None:
            is_internal_timeout = "Timeout reached" in output
            res["time"] = TIMEOUT if is_internal_timeout else elapsed
            res["cost"] = cost
            res["status"] = "Timeout (Partial)" if is_internal_timeout else "Solved"
        else:
            # Check for SAT with 0 cost (unweighted implied)
            if "SAT Found" in output:
                res["time"] = elapsed
                res["cost"] = 0
                res["status"] = "Solved"

    except subprocess.TimeoutExpired:
        res["status"] = "Hard Timeout"
        res["time"] = TIMEOUT
        
    except Exception as e:
        res["status"] = f"Error: {str(e)}"

    return res

def process_dataset(directory):
    files = glob.glob(os.path.join(directory, "*.wcnf"))
    if not files:
        print(f"No .wcnf files found in {directory}")
        return []

    print(f"Found {len(files)} files. Running with {MAX_WORKERS} concurrent threads...")
    
    tasks = []
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for f in files:
            for solver in SOLVERS:
                tasks.append(executor.submit(run_single_benchmark, f, solver))
        
        # Progress bar
        total = len(tasks)
        completed = 0
        for future in as_completed(tasks):
            res = future.result()
            results.append(res)
            completed += 1
            print(f"\rProgress: {completed}/{total} - {res['solver']} on {res['file']}: {res['status']}", end="")
    
    print("\nBenchmark Complete.")
    return pd.DataFrame(results)

def generate_graphs(df):
    if df.empty:
        return

    # 1. Success Rate Bar Chart
    plt.figure(figsize=(10, 6))
    
    # Count 'Solved' status per solver
    solved_counts = df[df['status'] == 'Solved'].groupby('solver').size()
    
    # Ensure all solvers are represented even if 0 count
    for s in SOLVERS:
        if s['name'] not in solved_counts:
            solved_counts[s['name']] = 0
            
    colors = [next(s['color'] for s in SOLVERS if s['name'] == name) for name in solved_counts.index]
    
    bars = solved_counts.plot(kind='bar', color=colors)
    plt.title('Success Rate (Instances Solved before Timeout)')
    plt.ylabel('Number of Files Solved')
    plt.xlabel('Solver')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(solved_counts):
        plt.text(i, v + 0.1, str(v), ha='center')
        
    plt.tight_layout()
    plt.savefig('maxsat_success_rate.png')
    print("Saved maxsat_success_rate.png")

    # 2. Cactus Plot (Time vs Solved Instances)
    plt.figure(figsize=(12, 8))
    
    for solver in SOLVERS:
        name = solver['name']
        color = solver['color']
        
        # Get times for successful solves only
        solver_data = df[(df['solver'] == name) & (df['status'] == 'Solved')].copy()
        
        # Sort times ascending
        times = sorted(solver_data['time'].tolist())
        
        # Cumulative count
        x = range(1, len(times) + 1)
        
        plt.plot(x, times, marker='o', label=name, color=color, linewidth=2)

    plt.title('Cactus Plot (Time vs Solved Instances)')
    plt.xlabel('Number of Solved Instances')
    plt.ylabel('Time (s)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('maxsat_cactus_plot.png')
    print("Saved maxsat_cactus_plot.png")

    # 3. CSV Export
    df.to_csv('benchmark_results.csv', index=False)
    print("Saved benchmark_results.csv")

def main():
    if len(sys.argv) < 2:
        print("Usage: python maxsat_analysis.py <directory_with_wcnf>")
        sys.exit(1)
    
    target = sys.argv[1]
    
    if not compile_cpp():
        print("Warning: C++ compilation failed.")

    if os.path.isfile(target):
        # Single file mode (backward compatibility)
        print("Single file detected. Processing...")
        # Wrap logic to reuse dataframe structure
        results = []
        for s in SOLVERS:
            results.append(run_single_benchmark(target, s))
        df = pd.DataFrame(results)
    else:
        # Directory mode
        df = process_dataset(target)

    generate_graphs(df)

if __name__ == "__main__":
    main()