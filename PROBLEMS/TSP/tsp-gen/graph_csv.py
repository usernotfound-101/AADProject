import os
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

data_type = [('u','i4'),('v','i4'),('w','f8')]


########################################################
# PARSER — Supports all normal TSPLIB coordinate formats
########################################################
def parse_tsp_file(filename):
    coords = []
    with open(filename) as f:
        reading = False
        for line in f:
            line = line.strip()

            if line == "NODE_COORD_SECTION":
                reading = True
                continue
            if line == "EOF":
                break
            if not reading:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            # id x y
            _, x, y = parts[:3]
            coords.append((float(x), float(y)))

    n = len(coords)

    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            adj[i, j] = d
            adj[j, i] = d

    # Edge list
    edge_list = [(i, j, adj[i, j]) for i in range(n) for j in range(i + 1, n)]
    edges = np.array(edge_list, dtype=data_type)
    m = len(edges)

    return coords, adj, edges, n, m


########################################################
# 2-APPROX MST
########################################################
def mstapprox(adj, edges, n):
    parent = np.arange(n)
    size = np.ones(n)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            parent[ra] = rb
            size[rb] += size[ra]
        else:
            parent[rb] = ra
            size[ra] += size[rb]
        return True

    # Kruskal
    T = np.zeros((n, n))
    for u, v, w in sorted(edges, key=lambda x: x[2]):
        if union(u, v):
            T[u, v] = T[v, u] = w

    # DFS preorder → Tour
    visited = np.zeros(n, dtype=bool)
    tour = []

    def dfs(u):
        visited[u] = True
        tour.append(u)
        for v in range(n):
            if T[u, v] != 0 and not visited[v]:
                dfs(v)

    dfs(0)
    tour.append(0)

    cost = sum(adj[tour[i], tour[i+1]] for i in range(len(tour)-1))
    return tour, cost


########################################################
# CHRISTOFIDES
########################################################
def find_mst(adj, n):
    T = np.zeros((n, n))
    visited = np.zeros(n, dtype=bool)
    visited[0] = True

    for _ in range(n - 1):
        best = (None, None, float("inf"))
        for i in range(n):
            if visited[i]:
                for j in range(n):
                    if not visited[j] and 0 < adj[i, j] < best[2]:
                        best = (i, j, adj[i, j])
        u, v, w = best
        T[u, v] = T[v, u] = w
        visited[v] = True

    return T


def find_odd_nodes(T):
    return [i for i in range(len(T)) if np.sum(T[i] != 0) % 2 == 1]


def minimum_matching(adj, odd):
    used = set()
    pairs = []
    distlist = []

    for i in range(len(odd)):
        for j in range(i + 1, len(odd)):
            u, v = odd[i], odd[j]
            distlist.append((adj[u, v], u, v))

    distlist.sort()

    for w, u, v in distlist:
        if u not in used and v not in used:
            used.add(u)
            used.add(v)
            pairs.append((u, v, w))

    return pairs


def christofides_tsp(adj, n):
    mst = find_mst(adj, n)
    odd = find_odd_nodes(mst)
    match = minimum_matching(adj, odd)

    multigraph = np.copy(mst)
    for u, v, w in match:
        multigraph[u, v] = multigraph[v, u] = w

    # DFS gives us an Eulerian walk -> simple TSP tour
    visited = np.zeros(n, dtype=bool)
    tour = []

    def dfs(u):
        visited[u] = True
        tour.append(u)
        for v in range(n):
            if multigraph[u, v] != 0 and not visited[v]:
                dfs(v)

    dfs(0)
    tour.append(0)

    cost = sum(adj[tour[i], tour[i+1]] for i in range(len(tour)-1))
    return tour, cost


########################################################
# PLOTTING
########################################################
def save_tour_image(coords, tour, title, outfile):
    xs = [coords[i][0] for i in tour]
    ys = [coords[i][1] for i in tour]

    plt.figure(figsize=(7, 7))
    plt.plot(xs, ys, marker='o')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


########################################################
# RUN ALL FILES
########################################################
def run_all_tsp_files():
    input_dir = os.path.join(os.getcwd(), "dataset-tsp")
    out_base = os.path.join(os.getcwd(), "generated")
    out_mst = os.path.join(out_base, "mst-approx")
    out_ch = os.path.join(out_base, "christofides")
    out_rt = os.path.join(out_base, "runtime")

    # create directories if missing
    os.makedirs(out_mst, exist_ok=True)
    os.makedirs(out_ch, exist_ok=True)
    os.makedirs(out_rt, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.tsp')]

    # runtime CSV
    runtime_csv = os.path.join(out_rt, "runtime.csv")
    write_header = not os.path.exists(runtime_csv)

    with open(runtime_csv, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["filename", "n_nodes", "mst_cost", "mst_time_s", "christofides_cost", "christofides_time_s", "total_time_s"])

        for fname in files:
            print("\n==========================================")
            print(f"Processing {fname}")
            print("==========================================")

            name = fname.replace(".tsp", "")

            filepath = os.path.join(input_dir, fname)
            coords, adj, edges, n, m = parse_tsp_file(filepath)

            # 2-approx MST
            t0 = time.perf_counter()
            mst_tour, mst_cost = mstapprox(adj, edges, n)
            t1 = time.perf_counter()
            mst_time = t1 - t0
            print(f"2-Approx MST Cost: {mst_cost} (time: {mst_time:.6f}s)")
            mst_img = os.path.join(out_mst, f"{name}_mst.png")
            save_tour_image(coords, mst_tour, f"{name} - MST Tour", mst_img)

            # Christofides
            t2 = time.perf_counter()
            ch_tour, ch_cost = christofides_tsp(adj, n)
            t3 = time.perf_counter()
            ch_time = t3 - t2
            print(f"Christofides Cost: {ch_cost} (time: {ch_time:.6f}s)")
            ch_img = os.path.join(out_ch, f"{name}_christofides.png")
            save_tour_image(coords, ch_tour, f"{name} - Christofides Tour", ch_img)

            total_time = (t3 - t0)
            writer.writerow([name, n, mst_cost, f"{mst_time:.6f}", ch_cost, f"{ch_time:.6f}", f"{total_time:.6f}"])

            print(f"Saved images: {mst_img}, {ch_img}")
            print(f"Runtime written to: {runtime_csv}")


if __name__ == "__main__":
    run_all_tsp_files()
