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
                    # Handle empty lines or malformed data
                    continue
    if dimension == 0:
        raise ValueError("Could not find 'DIMENSION' in graph file.")
    return adj, dimension

def solve_hamiltonian_path(adj, n):
    for start_node in range(1, n + 1):
        path = [start_node]
        visited = {start_node}
        
        # Try to find a path starting from this node
        if backtrack(start_node, path, visited, adj, n):
            return path  # Found a path

    return None  # No path found from any starting node

def backtrack(u, path, visited, adj, n):
    # 1. Goal: If path length is n, we've visited every node.
    if len(path) == n:
        return True

    # 2. Choices: Explore all neighbors 'v' of the current node 'u'
    for v in adj[u]:
        # 3. Constraint: Only visit 'v' if it hasn't been visited
        if v not in visited:
            
            # Make move
            visited.add(v)
            path.append(v)

            # Recurse
            if backtrack(v, path, visited, adj, n):
                return True
            
            # Backtrack (undo the move)
            path.pop()
            visited.remove(v)
            
    # If no neighbor leads to a solution, return False
    return False

def main():
    # Use the hardcoded path from your prompt
    filepath = "../../../Datasets_Material/FHCPCS/graph1.hcp"
    
    try:
        adj, n = parse_graph(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please check the relative path from where you are running the script.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    print(f"--- Brute-Force Hamiltonian Path Solver ---")
    print(f"Loaded graph '{filepath}' with {n} vertices.")

    try:
        start_time = time.perf_counter()
        path = solve_hamiltonian_path(adj, n)
        end_time = time.perf_counter()
        
        duration = end_time - start_time

        if path:
            print(f"\nFound Hamiltonian Path in {duration:.6f} seconds:")
            # Truncate output if path is too long
            if len(path) > 20:
                print(" -> ".join(map(str, path[:10])) + " ... " + " -> ".join(map(str, path[-10:])))
            else:
                print(" -> ".join(map(str, path)))
        else:
            print(f"\n No Hamiltonian Path found.")
            print(f"   (Search completed in {duration:.6f} seconds)")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Ran for {duration:.6f} seconds before stopping.")


if __name__ == "__main__":
    main()