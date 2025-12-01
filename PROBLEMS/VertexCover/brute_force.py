import networkx as nx

def exact_vertex_cover(graph):
    """
    Finds the minimum vertex cover using a recursive branching algorithm.
    Args:
        graph (nx.Graph): The input graph.
    Returns:
        set: The set of vertices in the minimum vertex cover.
    """
    # Base Case: If there are no edges left, the cover is empty.
    if graph.number_of_edges() == 0:
        return set()

    # 1. Pick an arbitrary edge (u, v)
    # We use an iterator to get just one edge efficiently
    u, v = next(iter(graph.edges()))

    # --- BRANCH 1: Pick 'u' ---
    # Create a copy of the graph and remove u (and its edges)
    g_without_u = graph.copy()
    g_without_u.remove_node(u)
    
    # Recursive call: Find cover for the rest + add u to current set
    cover_u = exact_vertex_cover(g_without_u)
    cover_u.add(u)

    # --- BRANCH 2: Pick 'v' ---
    # Create a copy of the graph and remove v (and its edges)
    g_without_v = graph.copy()
    g_without_v.remove_node(v)
    
    # Recursive call: Find cover for the rest + add v to current set
    cover_v = exact_vertex_cover(g_without_v)
    cover_v.add(v)

    # --- COMPARE ---
    # Return the smaller of the two valid covers
    if len(cover_u) <= len(cover_v):
        return cover_u
    else:
        return cover_v

# --- TEST DRIVER ---
if __name__ == "__main__":
    # Create a sample graph (A triangle + one extra edge)
    # 0 -- 1
    # |  / 
    # 2 -- 3
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])

    print("Calculating exact cover...")
    min_cover = exact_vertex_cover(G)
    
    print(f"Minimum Vertex Cover Size: {len(min_cover)}")
    print(f"Vertices: {min_cover}")