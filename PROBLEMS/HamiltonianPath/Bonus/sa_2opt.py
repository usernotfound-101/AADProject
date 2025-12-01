import sys
import time
import re
import random
import math
import os
import glob
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_graph_and_tour(filepath):
    """
    Parses an .hcp file for:
    - n (dimension)
    - adj_set (a dict of sets for O(1) edge lookups, 0-indexed)
    - optimal_tour (a known solution, 0-indexed)
    """
    dimension = 0
    adj = defaultdict(set)
    tour = []
    
    in_edge_section = False
    in_tour_section = False
    dim_regex = re.compile(r"DIMENSION\s*:\s*(\d+)")

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not dimension:
                match = dim_regex.match(line)
                if match:
                    dimension = int(match.group(1))
                    # Initialize for 0-indexed nodes (0 to n-1)
                    adj = {i: set() for i in range(dimension)}

            if line == "EDGE_DATA_SECTION":
                in_edge_section = True
                continue
            
            if "TOUR_SECTION" in line:
                in_edge_section = False
                in_tour_section = True
                continue
                
            if line == "EOF" or line == "-1":
                in_edge_section = False
                in_tour_section = False
                break

            if in_edge_section:
                try:
                    # Convert 1-indexed (1-66) to 0-indexed (0-65)
                    parts = list(map(int, line.split()))
                    # Some files might have multiple edges on one line or just u v
                    # The standard format usually has one edge per line or list of neighbors?
                    # HCP format usually: "1 2" means edge (1,2).
                    # But sometimes it is an adjacency list? 
                    # Let's assume standard TSPLIB HCP format which is usually edge list.
                    # Wait, the previous code assumed "u v".
                    if len(parts) >= 2:
                        u = parts[0]
                        v = parts[1]
                        adj[u-1].add(v-1)
                        adj[v-1].add(u-1)
                except ValueError:
                    continue
            
            if in_tour_section:
                try:
                    # Convert 1-indexed to 0-indexed
                    val = int(line)
                    if val != -1:
                        tour.append(val - 1)
                except ValueError:
                    continue
                    
    if dimension == 0:
        raise ValueError("Could not find 'DIMENSION' in graph file.")
        
    return adj, dimension, tour

def calculate_path_cost(path, adj_set, n):
    """
    Calculates the 'cost' of a path (a full permutation of n nodes).
    The cost is the number of missing edges for it to be a cycle.
    Cost = 0 means a perfect Hamiltonian Cycle.
    """
    cost = 0
    for i in range(n - 1):
        u = path[i]
        v = path[i+1]
        # Check if edge (u, v) exists
        if v not in adj_set[u]:
            cost += 1
            
    # Check the "wrap-around" edge to make it a cycle
    last_node = path[n-1]
    first_node = path[0]
    if first_node not in adj_set[last_node]:
        cost += 1
        
    return cost

def get_longest_path_in_cycle(path, adj_set, n):
    """
    Finds the length of the longest contiguous path in the given cycle permutation.
    Returns the number of edges in that path.
    """
    # We treat the path as a cycle, so we concatenate it with itself to handle wrap-around easily
    extended_path = path + path
    max_len = 0
    current_len = 0
    
    for i in range(len(extended_path) - 1):
        u = extended_path[i]
        v = extended_path[i+1]
        if v in adj_set[u]:
            current_len += 1
        else:
            max_len = max(max_len, current_len)
            current_len = 0
            
        # Optimization: if we found a path of length n-1, that's the max possible
        if current_len >= n - 1:
            return n - 1
            
    max_len = max(max_len, current_len)
    # The max length cannot exceed n-1 (Hamiltonian Path) or n (Hamiltonian Cycle)
    # But for "Path Ratio", we usually care about edges. A Hamiltonian Path has n-1 edges.
    return min(max_len, n)

def perform_2_opt_swap(path, i, j):
    """
    Performs a 2-opt swap on a path by reversing the segment
    from index i to index j.
    """
    # Ensure i is less than j
    if i > j:
        i, j = j, i
        
    # Create the new path:
    # 1. The part before the swap
    # 2. The reversed segment
    # 3. The part after the swap
    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
    return new_path

def solve_local_search_2_opt(initial_path, adj_set, n):
    """
    Runs the 2-opt local search algorithm.
    It repeatedly finds the *best* swap and applies it,
    until no more improvements can be found (local optimum).
    """
    current_path = list(initial_path)
    current_cost = calculate_path_cost(current_path, adj_set, n)
    best_cost = current_cost
    
    improvement_found = True
    
    while improvement_found:
        improvement_found = False
        
        # This is a "steepest descent" search:
        # We check all (n^2) swaps and pick the BEST one.
        best_new_path = current_path
        best_new_cost = current_cost
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_path = perform_2_opt_swap(current_path, i, j)
                new_cost = calculate_path_cost(new_path, adj_set, n)
                
                if new_cost < best_new_cost:
                    best_new_path = new_path
                    best_new_cost = new_cost
        
        # After checking all swaps, see if we found an improvement
        if best_new_cost < current_cost:
            current_path = best_new_path
            current_cost = best_new_cost
            improvement_found = True
        
        if current_cost < best_cost:
            best_cost = current_cost
            
    return current_path, best_cost

def solve_simulated_annealing(initial_path, adj_set, n,
                              initial_temp=100.0,
                              cooling_rate=0.9995,
                              max_iterations=500000):
    """
    Runs the Simulated Annealing metaheuristic.
    """
    current_path = list(initial_path)
    current_cost = calculate_path_cost(current_path, adj_set, n)
    
    best_path = current_path
    best_cost = current_cost
    
    temp = initial_temp
    
    for i in range(max_iterations):
        if best_cost == 0:
            print(f"   (SA found optimal solution at iteration {i})")
            break
            
        if temp <= 0.0001:
            break
            
        # 1. Generate a "neighbor" solution using a *random* 2-opt move
        i, j = sorted(random.sample(range(n), 2))
        new_path = perform_2_opt_swap(current_path, i, j)
        new_cost = calculate_path_cost(new_path, adj_set, n)
        
        # 2. Decide to accept
        cost_delta = new_cost - current_cost
        
        if cost_delta < 0:
            # Better solution: always accept
            current_path = new_path
            current_cost = new_cost
        else:
            # Worse solution: accept based on probability
            acceptance_prob = math.exp(-cost_delta / temp)
            if random.random() < acceptance_prob:
                current_path = new_path
                current_cost = new_cost
                
        # 3. Update best-ever found solution
        if current_cost < best_cost:
            best_path = current_path
            best_cost = current_cost
            
        # 4. Cool down
        temp *= cooling_rate
        
    return best_path, best_cost


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "../../../../Datasets_Material/FHCPCS/")
    # Get all .hcp files and sort them to ensure consistent order
    files = sorted(glob.glob(os.path.join(dataset_path, "*.hcp")))
    
    # Take the first 100 graphs
    files_to_process = files[:1000]
    
    if not files_to_process:
        print(f"No .hcp files found in {dataset_path}")
        sys.exit(1)
        
    print(f"Found {len(files)} files. Processing the first {len(files_to_process)}...")
    
    results = []
    
    for idx, filepath in enumerate(files_to_process):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(files_to_process)}] Processing {filename}...")
        
        try:
            adj, n, _ = parse_graph_and_tour(filepath)
        except Exception as e:
            print(f"  Error parsing {filename}: {e}")
            continue
            
        # Calculate number of edges m
        # adj is u -> {v1, v2...}. Since it's undirected, each edge is counted twice.
        m = sum(len(neighbors) for neighbors in adj.values()) // 2
        
        # Create a random starting point
        random_path = list(range(n))
        random.shuffle(random_path)
        
        # Run Simulated Annealing
        start_time = time.perf_counter()
        # Using fewer iterations for speed if needed, but 500k is default. 
        # Let's stick to default or maybe slightly less if 100 graphs take too long.
        # 500,000 iterations is quite a lot for 100 graphs. 
        # Let's reduce to 100,000 for this batch run to be reasonable.
        sa_path, sa_cost = solve_simulated_annealing(random_path, adj, n, max_iterations=100000)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        
        # Calculate metrics
        longest_path_edges = get_longest_path_in_cycle(sa_path, adj, n)
        # Path ratio: length of found path / max possible length (n-1)
        path_ratio = longest_path_edges / (n - 1) if n > 1 else 1.0
        
        is_hamiltonian = (sa_cost == 0)
        
        results.append({
            "filename": filename,
            "n": n,
            "m": m,
            "time": elapsed_time,
            "path_ratio": path_ratio,
            "is_hamiltonian": is_hamiltonian,
            "cost": sa_cost
        })
        
        print(f"  n={n}, m={m}, time={elapsed_time:.4f}s, ratio={path_ratio:.4f}, cost={sa_cost}")

    # Save results to JSON
    output_json = "sa_metrics.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nMetrics saved to {output_json}")
    
    # --- Plotting ---
    if not results:
        print("No results to plot.")
        return

    ns = [r['n'] for r in results]
    ms = [r['m'] for r in results]
    times = [r['time'] for r in results]
    ratios = [r['path_ratio'] for r in results]
    
    # 1. Edges vs Time
    plt.figure(figsize=(10, 6))
    plt.scatter(ms, times, alpha=0.7, c='blue')
    plt.title("Edges vs Time (Simulated Annealing)")
    plt.xlabel("Number of Edges (m)")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig("sa_edges_vs_time.png")
    plt.close()
    
    # 2. Path Ratio vs n
    plt.figure(figsize=(10, 6))
    plt.scatter(ns, ratios, alpha=0.7, c='green')
    plt.title("Path Ratio vs Number of Vertices (n)")
    plt.xlabel("Number of Vertices (n)")
    plt.ylabel("Path Ratio (Found Length / (n-1))")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("sa_path_ratio_vs_n.png")
    plt.close()
    
    # 3. Time vs n
    plt.figure(figsize=(10, 6))
    plt.scatter(ns, times, alpha=0.7, c='red')
    plt.title("Time vs Number of Vertices (n)")
    plt.xlabel("Number of Vertices (n)")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig("sa_time_vs_n.png")
    plt.close()
    
    print("Plots saved: sa_edges_vs_time.png, sa_path_ratio_vs_n.png, sa_time_vs_n.png")

if __name__ == "__main__":
    main()