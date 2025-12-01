import numpy as np

data_type = [('u','i4'),('v','i4'),('w','f8')]

def get_graph():
    numcities = int(input("Enter number of cities (0-indexed): "))
    numroads = int(input("Enter number of roads: "))

    adj = np.zeros((numcities, numcities), dtype=float)
    edges = np.array([], dtype=data_type)

    print("Enter roads in u v w format (u and v are cities, w is weight):")
    for _ in range(numroads):
        u, v, w = input().split()
        u, v, w = int(u), int(v), float(w)

        if adj[u, v] == 0:
            adj[u, v] = w
            adj[v, u] = w
        else:
            adj[u, v] = adj[v, u] = min(adj[u, v], w)

        edges = np.append(edges, np.array([(u, v, w)], dtype=data_type))
    return adj, edges, numcities, numroads

def find_mst(adj, n):
    tree = np.zeros((n, n), dtype=float)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True

    for _ in range(n-1):
        min_edge = None
        min_weight = float('inf')
        for u in range(n):
            if visited[u]:
                for v in range(n):
                    if not visited[v] and adj[u, v] != 0 and adj[u, v] < min_weight:
                        min_edge = (u, v)
                        min_weight = adj[u, v]
        u, v = min_edge
        tree[u, v] = tree[v, u] = min_weight
        visited[v] = True
    return tree

def find_odd_nodes(tree, n):
    odd = []
    for i in range(n):
        if np.sum(tree[i] != 0) % 2 != 0:
            odd.append(i)
    return odd

def minimum_matching(adj, odd):
    matched = set()
    matching = []

    edges = []
    for i in range(len(odd)):
        u = odd[i]
        for j in range(i+1, len(odd)):
            v = odd[j]
            edges.append((u, v, adj[u, v]))
    edges = sorted(edges, key=lambda x: x[2])

    for u, v, w in edges:
        if u not in matched and v not in matched:
            matching.append((u, v, w))
            matched.add(u)
            matched.add(v)
    return matching

def euler_tour(graph, u, visited, tour):
    visited[u] = True
    tour.append(u)
    for v in range(len(graph)):
        if graph[u, v] != 0 and not visited[v]:
            euler_tour(graph, v, visited, tour)

def christofides_tsp(adj, n):
    mst = find_mst(adj, n)
    odd_nodes = find_odd_nodes(mst, n)
    matching = minimum_matching(adj, odd_nodes)
    multigraph = np.array(mst)
    for u, v, w in matching:
        multigraph[u, v] = multigraph[v, u] = w

    tour = []
    visited = np.zeros(n, dtype=bool)
    euler_tour(multigraph, 0, visited, tour)
    tour.append(0)
    cost = 0
    for i in range(len(tour)-1):
        cost += adj[tour[i], tour[i+1]]

    return tour, cost

if __name__ == "__main__":
    adj, edges, n, m = get_graph()
    tour, cost = christofides_tsp(adj, n)
    print("Approximate TSP tour:", tour)
    print("Tour cost:", cost)
