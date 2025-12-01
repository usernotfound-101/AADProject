""" This is based on partial paths mentioned in wikipedia (Still O(N!) in worst case) but due to pruning rules it performs better"""

import sys
import time
import re
from collections import defaultdict

def parse_graph(filepath):
    """
    Parses an .hcp file to extract the graph dimension (n)
    and an adjacency list.
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
                    adj = {i: set() for i in range(1, dimension + 1)}

            if line == "EDGE_DATA_SECTION":
                in_edge_section = True
                continue
            
            if line == "EOF" or "TOUR_SECTION" in line:
                in_edge_section = False
                break

            if in_edge_section:
                try:
                    u, v = map(int, line.split())
                    adj[u].add(v)
                    adj[v].add(u)
                except ValueError:
                    continue
                    
    if dimension == 0:
        raise ValueError("Could not find 'DIMENSION' in graph file.")
        
    return adj, dimension

def preprocess_prune_for_cycle(adj, n):
    """
    Applies the simplest pruning rule for Hamiltonian Cycles.
    If any node has degree < 2, a cycle is impossible.
    """
    # Rule: Check degree of all nodes.
    for node, neighbors in adj.items():
        if len(neighbors) < 2:
            # This node makes a cycle impossible.
            return False, f"Node {node} has degree {len(neighbors)}. (Rule: degree >= 2)"
    
    # All nodes passed the test.
    return True, "All nodes have degree >= 2."


def solve_hamiltonian_cycle(adj, n):
    start_node = 1
    path = [start_node]
    visited = {start_node}
    
    if backtrack(start_node, path, visited, adj, n, start_node):
        path.append(start_node)
        return path

    return None

def backtrack(u, path, visited, adj, n, start_node):
    """
    (Identical to backtracking_cycle.py)
    """
    if len(path) == n:
        if start_node in adj[u]:
            return True
        else:
            return False

    # This is where Rubin's algorithm would add *dynamic pruning* rules
    # We are omitting them for simplicity.
    
    for v in adj[u]:
        if v not in visited:
            visited.add(v)
            path.append(v)

            if backtrack(v, path, visited, adj, n, start_node):
                return True
            
            path.pop()
            visited.remove(v)
            
    return False

def main():
    filepath = "../../../Datasets_Material/FHCPCS/graph2.hcp"
    
    try:
        adj, n = parse_graph(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    print(f"--- Pruning-Based Hamiltonian Cycle Solver ---")
    print(f"Loaded graph '{filepath}' with {n} vertices.")
    
    # --- NEW PRUNING STEP ---
    print("\n[Step 1: Running Pre-processing Pruning Rules]")
    is_possible, reason = preprocess_prune_for_cycle(adj, n)
    print(f"   Result: {reason}")
    
    if not is_possible:
        print("\n[Result Graph is provably impossible. Halting.")
        sys.exit(0)
    # --------------------------

    try:
        start_time = time.perf_counter()
        cycle = solve_hamiltonian_cycle(adj, n)
        end_time = time.perf_counter()
        
        duration = end_time - start_time

        if cycle:
            print(f"\n✅ Found Hamiltonian Cycle in {duration:.6f} seconds.")
        else:
            print(f"\n❌ No Hamiltonian Cycle found.")
            print(f"   (Search completed in {duration:.6f} seconds)")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Ran for {duration:.6f} seconds before stopping.")


if __name__ == "__main__":
    main()