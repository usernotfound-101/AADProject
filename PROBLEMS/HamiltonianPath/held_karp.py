import sys
import time
import re
from collections import defaultdict

def parse_graph(filepath):
    """
    Parses an .hcp file to extract the graph dimension (n)
    and an adjacency list.
    
    NOTE: This algorithm requires 0-indexed nodes for bitmasking.
    This parser will convert the 1-indexed graph to 0-indexed.
    """
    dimension = 0
    adj = defaultdict(set)
    in_edge_section = False
    dim_regex = re.compile(r"DIMENSION\s*:\s*(\d+)")

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not dimension:
                match = dim_regex.match(line)
                if match:
                    dimension = int(match.group(1))
                    # Initialize adj list for 0-indexed nodes
                    adj = {i: set() for i in range(dimension)}

            if line == "EDGE_DATA_SECTION":
                in_edge_section = True
                continue
            
            if line == "EOF" or "TOUR_SECTION" in line:
                in_edge_section = False
                break

            if in_edge_section:
                try:
                    # Convert 1-indexed (e.g., 1-66) to 0-indexed (e.g., 0-65)
                    u, v = map(int, line.split())
                    adj[u-1].add(v-1)
                    adj[v-1].add(u-1)
                except ValueError:
                    continue
                    
    if dimension == 0:
        raise ValueError("Could not find 'DIMENSION' in graph file.")
        
    return adj, dimension

def solve_held_karp(adj, n):
    """
    Solves the Hamiltonian Path problem using Dynamic Programming (Held-Karp).
    
    State: dp[mask][u] = True if a path exists visiting all nodes in 'mask'
                         and ending at node 'u'.
    """
    
    # 1. Check if n is too large to even attempt.
    if n > 20:
        print(f"Error: n={n} is too large for Held-Karp (O(n^2 * 2^n)).")
        print("   This algorithm is only feasible for n <= ~20.")
        print("   Aborting to prevent system crash.")
        return None, 0.0

    # 2. Initialize DP table
    # dp[mask][last_node]
    # We need 2^n masks (0 to 2^n - 1)
    # We need n last_nodes (0 to n-1)
    try:
        dp = [[False for _ in range(n)] for _ in range(1 << n)]
    except MemoryError:
        print(f"Error: Failed to allocate DP table of size (2^{n}) x {n}.")
        print("   This demonstrates the O(n * 2^n) space complexity.")
        return None, 0.0

    # 3. Base Cases
    # A path of length 1, starting and ending at node 'i'
    # The mask for a set {i} is (1 << i)
    for i in range(n):
        dp[1 << i][i] = True

    start_time = time.perf_counter()

    # 4. Fill the DP table
    # Iterate through all masks, from 1 to 2^n - 1
    for mask in range(1, 1 << n):
        # For each mask, check all possible last nodes 'u'
        for u in range(n):
            # 'u' must be in the mask. (mask >> u) & 1 checks if u's bit is set.
            if (mask >> u) & 1:
                # If the sub-problem dp[mask][u] is possible...
                if dp[mask][u]:
                    # ...then try to extend this path to a new node 'v'
                    for v in adj[u]:
                        # 'v' must NOT already be in the mask
                        if not (mask >> v) & 1:
                            # New mask is 'mask' with 'v's bit turned on
                            new_mask = mask | (1 << v)
                            # We can now reach state (new_mask, v)
                            dp[new_mask][v] = True
                            
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # 5. Check for a solution
    # The final mask is (1 << n) - 1, which has all n bits set
    all_nodes_mask = (1 << n) - 1
    
    for last_node in range(n):
        if dp[all_nodes_mask][last_node]:
            return f"Path found (ended at node {last_node + 1})", duration

    return None, duration

def main():
    filepath = "../../../Datasets_Material/FHCPCS/graph1.hcp"
    
    try:
        adj, n = parse_graph(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    print(f"--- Held-Karp Dynamic Programming Solver ---")
    print(f"Loaded graph '{filepath}' with {n} vertices.")
    print("\nAttempting to find path using Held-Karp (DP)...")
    print("="*40)
    print("WARNING: This algorithm has a time complexity of O(n^2 * 2^n)")
    print("   and a space complexity of O(n * 2^n).")
    print(f"   For n={n}, this is computationally infeasible.")
    print("="*40)
    
    path, duration = solve_held_karp(adj, n)
    
    if path:
        print(f"\n✅ {path}")
        print(f"   (Search completed in {duration:.6f} seconds)")
    else:
        print(f"\n❌ No Hamiltonian Path found.")
        print(f"   (Search completed in {duration:.6f} seconds)")


if __name__ == "__main__":
    main() 