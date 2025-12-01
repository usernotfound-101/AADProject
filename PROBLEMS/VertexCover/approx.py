import copy

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        # Using a dictionary for adjacency list to handle edge removal easily
        self.graph = {i: [] for i in range(vertices)}
        self.edges = []

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.edges.append((u, v))

    def print_result(self, algorithm_name, cover):
        print(f"--- {algorithm_name} ---")
        print(f"Vertex Cover Size: {len(cover)}")
        print(f"Vertices Selected: {sorted(cover)}")
        print("-" * 30)

def get_degree(adj, vertex):
    return len(adj[vertex])

# ---------------------------------------------------------
# 1. Greedy Heuristic (Max Degree)
# Strategy: Pick vertex with max degree, remove incident edges, repeat.
# Approximation Ratio: O(log n)
# ---------------------------------------------------------
def greedy_heuristic_vertex_cover(g_input):
    # Create a deep copy to avoid modifying original graph structure
    adj = copy.deepcopy(g_input.graph)
    cover = []
    
    # While there are still edges in the graph
    while any(len(neighbors) > 0 for neighbors in adj.values()):
        # 1. Find the vertex with the maximum current degree
        # We filter keys to ensure we only look at valid vertices
        best_vertex = max(adj.keys(), key=lambda v: get_degree(adj, v))
        
        # If the graph has edges left but max degree is 0 (disconnected components), break
        if get_degree(adj, best_vertex) == 0:
            break

        # 2. Add to cover
        cover.append(best_vertex)
        
        # 3. Remove all edges incident to this vertex
        neighbors = adj[best_vertex]
        
        # We must remove the 'best_vertex' from the neighbor lists of its neighbors
        for neighbor in neighbors:
            if best_vertex in adj[neighbor]:
                adj[neighbor].remove(best_vertex)
        
        # Effectively remove the vertex (set its neighbors to empty)
        adj[best_vertex] = []
        
    return cover

# ---------------------------------------------------------
# 2. Approx-2 Algorithm (Maximal Matching)
# Strategy: Pick an edge (u,v), take BOTH u and v, remove all incident edges.
# Approximation Ratio: Exactly 2
# ---------------------------------------------------------
def approx2_vertex_cover(g_input):
    # We work with the edge list directly
    edges_remaining = copy.deepcopy(g_input.edges)
    cover = []
    
    while len(edges_remaining) > 0:
        # 1. Pick an arbitrary edge (u, v)
        u, v = edges_remaining.pop(0)
        
        # 2. Add BOTH u and v to the cover
        cover.append(u)
        cover.append(v)
        
        # 3. Remove all edges incident to u OR v from the remaining set
        # We iterate backwards or filter to safely remove while looping
        edges_remaining = [
            edge for edge in edges_remaining 
            if edge[0] != u and edge[0] != v and edge[1] != u and edge[1] != v
        ]
            
    return cover

# --- Driver Code ---
if __name__ == "__main__":
    # Creating a sample graph
    # This graph is often used to show where Greedy fails compared to Optimal,
    # though Approx-2 is a "loose" bound.
    g = Graph(7)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(1, 4)
    g.add_edge(2, 5)
    g.add_edge(3, 6)

    print("Graph Structure: 0 connects to 1, 2, 3. 1-4, 2-5, 3-6 are connected.")
    
    # Run Greedy
    greedy_solution = greedy_heuristic_vertex_cover(g)
    g.print_result("Greedy Heuristic (Max Degree)", greedy_solution)

    # Run Approx-2
    approx2_solution = approx2_vertex_cover(g)
    g.print_result("Approx-2 Algorithm (Maximal Matching)", approx2_solution)