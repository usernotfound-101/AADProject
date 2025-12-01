import sys
import time
import re
from collections import defaultdict
import tempfile

def parse_graph_and_tour(filepath):
    """
    Parses an .hcp file to extract the graph dimension (n),
    an adjacency list, and the optimal tour (if one is provided).
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
                    adj = {i: set() for i in range(1, dimension + 1)}

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
                    u, v = map(int, line.split())
                    adj[u].add(v)
                    adj[v].add(u)
                except ValueError:
                    continue
            
            if in_tour_section:
                try:
                    tour.append(int(line))
                except ValueError:
                    continue
                    
    if dimension == 0:
        raise ValueError("Could not find 'DIMENSION' in graph file.")
        
    return adj, dimension, tour

def count_unvisited_neighbors(node, adj, visited):
    """
    Helper function: counts neighbors of 'node' that are NOT in 'visited'.
    """
    count = 0
    for neighbor in adj[node]:
        if neighbor not in visited:
            count += 1
    return count

def solve_greedy_heuristic(adj, n):
    """
    Attempts to find a long path using the "Fewest Neighbors First"
    greedy heuristic. It tries all possible starting nodes.
    """
    best_path = []

    for start_node in range(1, n + 1):
        path = [start_node]
        visited = {start_node}
        current_node = start_node

        while True:
            # 1. Find all unvisited neighbors
            unvisited_neighbors = []
            for neighbor in adj[current_node]:
                if neighbor not in visited:
                    unvisited_neighbors.append(neighbor)
            
            # 2. If no unvisited neighbors, this path is stuck.
            if not unvisited_neighbors:
                break  # End of this path
            
            # 3. Find the "best" neighbor (greedy choice)
            # This is the one with the fewest unvisited neighbors OF ITS OWN.
            best_next_node = None
            min_neighbor_count = float('inf')

            for neighbor in unvisited_neighbors:
                count = count_unvisited_neighbors(neighbor, adj, visited)
                
                if count < min_neighbor_count:
                    min_neighbor_count = count
                    best_next_node = neighbor
            
            # 4. Make the greedy move
            path.append(best_next_node)
            visited.add(best_next_node)
            current_node = best_next_node
        
        # After a path is stuck, check if it's the longest one found
        if len(path) > len(best_path):
            best_path = path
            
            # Early exit: If we find a full path, we're done.
            if len(best_path) == n:
                break
                
    return best_path

def main():
    filepath = "../../../Datasets_Material/FHCPCS/graph5.hcp"
    
    try:
        adj, n, optimal_tour = parse_graph_and_tour(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    print(f"--- Greedy Heuristic Path Solver (Fewest Neighbors First) ---")
    print(f"Loaded graph '{filepath}' with {n} vertices.")
    
    if optimal_tour:
        print(f"   (Optimal tour of length {len(optimal_tour)} found in file)")
    
    print("\nAttempting to find a long path using greedy heuristic...")
    
    start_time = time.perf_counter()
    path = solve_greedy_heuristic(adj, n)
    end_time = time.perf_counter()
    
    duration = end_time - start_time

    print(f"\n--- Results ---")
    print(f"✅ Search complete in {duration:.6f} seconds.")
    print(f"   Heuristic Path Length: {len(path)}")
    print(f"   Graph Dimension (n):   {n}")
    
    if len(path) == n:
        print("\nSUCCESS: The heuristic found a full Hamiltonian Path!")
    else:
        print("\nNOTE: The heuristic found a partial path (this is expected).")
        
    # Truncate output if path is too long
    if len(path) > 20:
        print("\nPath Found:")
        print(" -> ".join(map(str, path[:10])) + " ... " + " -> ".join(map(str, path[-10:])))
    else:
        print("\nPath Found:")
        print(" -> ".join(map(str, path)))
    
    output_file = "path.txt"

    with open(output_file, "w") as f:
        f.write("--- Full Heuristic Path Output ---\n")
        f.write(f"Path length: {len(path)}\n\n")

        for node in path:
            f.write(f"{node}\n")  # each node on its own line

    print(f"\n📄 Full path saved to: {output_file}")

if __name__ == "__main__":
    main()