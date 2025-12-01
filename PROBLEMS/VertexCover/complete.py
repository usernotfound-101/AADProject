import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys

# ==========================================
# 1. ALGORITHM IMPLEMENTATIONS
# ==========================================

def exact_vertex_cover_recursive(G):
    """
    Exact Vertex Cover using Recursive Branching (Branch & Bound).
    Logic: Pick an edge (u, v). The optimal cover MUST contain u or v.
    Branch 1: Take u, remove it, solve for remaining graph.
    Branch 2: Take v, remove it, solve for remaining graph.
    Return the smaller of the two sets.
    
    Time Complexity: O(2^k * n) where k is the size of the cover.
    """
    # Base Case: If no edges remain, no vertices are needed.
    if G.number_of_edges() == 0:
        return set()
    
    # Pick an arbitrary edge (u, v)
    # Converting to list is necessary to access an element, 
    # but we only take the first one.
    u, v = list(G.edges())[0]
    
    # Branch 1: Select u (remove u and its edges)
    G_without_u = G.copy()
    G_without_u.remove_node(u)
    cover_u = {u} | exact_vertex_cover_recursive(G_without_u)
    
    # Optimization: If cover_u is small enough (e.g. <= current best), 
    # we might prune, but for this project, we explore both to be safe.
    
    # Branch 2: Select v (remove v and its edges)
    G_without_v = G.copy()
    G_without_v.remove_node(v)
    cover_v = {v} | exact_vertex_cover_recursive(G_without_v)
    
    # Return the smaller cover
    if len(cover_u) <= len(cover_v):
        return cover_u
    else:
        return cover_v

def greedy_degree_vertex_cover(G):
    """
    Greedy Heuristic: Always pick the vertex with the highest current degree.
    Logic: Remove that vertex and its edges, repeat.
    
    Time Complexity: O(V^2) in this simple implementation.
    Approximation: O(log n) factor approximation (can be bad on specific graphs).
    """
    cover = set()
    # Work on a copy to avoid modifying the original graph
    G_curr = G.copy()
    
    while G_curr.number_of_edges() > 0:
        # Find vertex with maximum degree
        # G_curr.degree returns (node, degree) pairs
        degrees = dict(G_curr.degree())
        if not degrees: break
        
        # Pick node with max degree
        best_node = max(degrees, key=degrees.get)
        
        cover.add(best_node)
        G_curr.remove_node(best_node)
        
    return cover

def approx_matching_vertex_cover(G):
    """
    2-Approximation Algorithm based on Maximal Matching.
    Logic: Pick an arbitrary edge (u, v). Add BOTH u and v to the cover.
    Remove both, repeat.
    
    Time Complexity: O(E) (Linear).
    Approximation: Guaranteed to be <= 2 * Optimal Size.
    """
    cover = set()
    G_curr = G.copy()
    
    while G_curr.number_of_edges() > 0:
        # Pick an arbitrary edge
        u, v = list(G_curr.edges())[0]
        
        # Add both to cover
        cover.add(u)
        cover.add(v)
        
        # Remove both endpoints from graph (removes incident edges too)
        G_curr.remove_node(u)
        G_curr.remove_node(v)
        
    return cover

# ==========================================
# 2. DATASET GENERATION & BENCHMARKING
# ==========================================

def generate_random_graph(n, p=0.5):
    """Generates an Erdos-Renyi random graph."""
    return nx.gnp_random_graph(n, p)

def run_benchmark():
    print("--- Starting Vertex Cover Project Benchmark ---")
    print("Comparison: Exact vs Greedy-Degree vs Approx-Matching")
    
    results = []
    
    # Define graph sizes to test.
    # Note: Exact algo is exponential. Keep N small (<25) for the exact test.
    # We will skip exact for larger graphs.
    sizes = [5, 10, 12, 15, 18, 20, 30, 50, 100]
    
    for N in sizes:
        # Generate a random graph
        # p=0.3 ensures graph isn't too dense (which makes Exact slower)
        G = generate_random_graph(N, p=0.3) 
        num_edges = G.number_of_edges()
        
        row = {'N': N, 'Edges': num_edges}
        
        print(f"\nTesting Graph N={N}, Edges={num_edges}...")
        
        # 1. Run Exact (Only if N is small enough)
        if N <= 22:
            start = time.time()
            exact_sol = exact_vertex_cover_recursive(G)
            duration = time.time() - start
            row['Exact_Size'] = len(exact_sol)
            row['Exact_Time'] = duration
            print(f"  [Exact]   Size: {len(exact_sol)}, Time: {duration:.4f}s")
        else:
            row['Exact_Size'] = None
            row['Exact_Time'] = None
            print(f"  [Exact]   Skipped (Too large for Brute Force)")
            
        # 2. Run Greedy Degree
        start = time.time()
        greedy_sol = greedy_degree_vertex_cover(G)
        duration = time.time() - start
        row['Greedy_Size'] = len(greedy_sol)
        row['Greedy_Time'] = duration
        print(f"  [Greedy]  Size: {len(greedy_sol)}, Time: {duration:.4f}s")
        
        # 3. Run Approx Matching
        start = time.time()
        approx_sol = approx_matching_vertex_cover(G)
        duration = time.time() - start
        row['Approx_Size'] = len(approx_sol)
        row['Approx_Time'] = duration
        print(f"  [Approx]  Size: {len(approx_sol)}, Time: {duration:.4f}s")
        
        # Calculate Approx Ratios (if Exact was run)
        if row['Exact_Size'] is not None and row['Exact_Size'] > 0:
            row['Ratio_Greedy'] = row['Greedy_Size'] / row['Exact_Size']
            row['Ratio_Approx'] = row['Approx_Size'] / row['Exact_Size']
        else:
            row['Ratio_Greedy'] = None
            row['Ratio_Approx'] = None
            
        results.append(row)

    return pd.DataFrame(results)

# ==========================================
# 3. VISUALIZATION
# ==========================================

def plot_results(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Solution Size Comparison
    # We filter out the very large graphs for the bar chart comparison 
    # where exact is missing, or we just plot lines.
    
    df_small = df.dropna(subset=['Exact_Size'])
    
    ax1.plot(df_small['N'], df_small['Exact_Size'], 'o-', label='Exact (Optimal)', color='green')
    ax1.plot(df['N'], df['Greedy_Size'], 's--', label='Greedy Heuristic', color='orange')
    ax1.plot(df['N'], df['Approx_Size'], '^--', label='2-Approx Matching', color='blue')
    
    ax1.set_title('Solution Size Comparison (Lower is Better)')
    ax1.set_xlabel('Number of Vertices (N)')
    ax1.set_ylabel('Vertex Cover Size')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Execution Time (Log Scale)
    ax2.plot(df_small['N'], df_small['Exact_Time'], 'o-', label='Exact (Exponential)', color='green')
    ax2.plot(df['N'], df['Greedy_Time'], 's-', label='Greedy (Poly)', color='orange')
    ax2.plot(df['N'], df['Approx_Time'], '^-', label='2-Approx (Linear)', color='blue')
    
    ax2.set_title('Execution Time (Log Scale)')
    ax2.set_xlabel('Number of Vertices (N)')
    ax2.set_ylabel('Time (Seconds)')
    ax2.set_yscale('log') # Critical for showing exponential growth
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Approximation Ratios
    # Only for sizes where Exact was computed
    if not df_small.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(df_small['N'], df_small['Ratio_Greedy'], 's-', label='Greedy Ratio', color='orange')
        plt.plot(df_small['N'], df_small['Ratio_Approx'], '^-', label='Approx Ratio', color='blue')
        plt.axhline(y=1.0, color='g', linestyle='-', alpha=0.3, label='Optimal (1.0)')
        plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.3, label='Theoretical Bound (2.0)')
        
        plt.title('Approximation Ratio (Closer to 1.0 is better)')
        plt.xlabel('Number of Vertices (N)')
        plt.ylabel('Ratio (Alg / Optimal)')
        plt.legend()
        plt.grid(True)
        plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Run benchmarks
    df_results = run_benchmark()
    
    # Display text table
    print("\n--- Final Results Summary ---")
    print(df_results[['N', 'Exact_Size', 'Greedy_Size', 'Approx_Size', 'Exact_Time']])
    
    # Plot graphs
    plot_results(df_results)