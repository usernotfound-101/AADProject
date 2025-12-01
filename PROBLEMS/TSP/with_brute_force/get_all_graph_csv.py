import os
import json
import csv
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# ensure generated dirs (next to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "small_tsp_dataset.csv")
OUT_BASE = os.path.join(BASE_DIR, "generated")
OUT_MST = os.path.join(OUT_BASE, "mst-approx")
OUT_CH = os.path.join(OUT_BASE, "christofides")
OUT_BR = os.path.join(OUT_BASE, "brute")
OUT_DP = os.path.join(OUT_BASE, "dp")
OUT_RT = os.path.join(OUT_BASE, "runtime")
os.makedirs(OUT_MST, exist_ok=True)
os.makedirs(OUT_CH, exist_ok=True)
os.makedirs(OUT_BR, exist_ok=True)
os.makedirs(OUT_DP, exist_ok=True)
os.makedirs(OUT_RT, exist_ok=True)

RUNTIME_CSV = os.path.join(OUT_RT, "runtime.csv")
CSV_FIELDS = [
    "filename", "n_nodes",
    "mst_cost", "mst_time_s",
    "christofides_cost", "christofides_time_s",
    "brute_cost", "brute_time_s",
    "dp_cost", "dp_time_s",
    "total_time_s"
]

def _parse_dataset(csv_path=DATASET_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    instances = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            name = str(row.get("instance_id") or row.get("id") or f"inst_{idx}")
            dm = row.get("distance_matrix") or row.get("distance") or ""
            coords_s = row.get("city_coordinates") or row.get("city_coords") or ""
            if not dm:
                continue
            try:
                mat = json.loads(dm)
                adj = np.array(mat, dtype=float)
            except Exception:
                continue
            coords = None
            if coords_s:
                try:
                    cj = json.loads(coords_s)
                    coords = [(float(x), float(y)) for x, y in cj]
                except Exception:
                    coords = None
            if coords is None:
                n0 = adj.shape[0]
                coords = [(math.cos(2*math.pi*i/n0), math.sin(2*math.pi*i/n0)) for i in range(n0)]
            instances.append((name, adj, coords))
    return instances

def _edges_from_adj(adj):
    n = adj.shape[0]
    data_type = [('u','i4'),('v','i4'),('w','f8')]
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            edge_list.append((i,j,float(adj[i,j])))
    edges = np.array(edge_list, dtype=data_type)
    return edges, len(edge_list)

def _preorder_from_tree(tree):
    n = tree.shape[0]
    vis = [False]*n
    tour = []
    def dfs(u):
        vis[u] = True
        tour.append(u)
        for v in range(n):
            if tree[u,v] != 0 and not vis[v]:
                dfs(v)
    dfs(0)
    tour.append(0)
    return tour

def _plot_tour(coords, tour, outpath, title):
    coords = np.array(coords)
    plt.figure(figsize=(4,4))
    plt.scatter(coords[:,0], coords[:,1], c='k', s=12)
    for i,(x,y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=8, color='blue')
    if tour:
        for a,b in zip(tour[:-1], tour[1:]):
            x1,y1 = coords[int(a)]
            x2,y2 = coords[int(b)]
            plt.plot([x1,x2],[y1,y2], 'r-', linewidth=0.9)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def _brute_force_tour(adj):
    n = adj.shape[0]
    if n > 10:
        return None, None
    best = float('inf'); best_t = None
    for perm in permutations(range(1,n)):
        tour = [0] + list(perm) + [0]
        c = 0.0; valid = True
        for u,v in zip(tour[:-1], tour[1:]):
            w = float(adj[u,v])
            if w == 0 and u!=v:
                valid = False; break
            c += w
        if valid and c < best:
            best = c; best_t = tour
    if best == float('inf'):
        return None, None
    return best_t, best

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

def mstapprox(adj, edges, n, m):
    def kruskal():
        tree=np.zeros((n,n),dtype=float)
        comps=np.ones(n,dtype=int)
        p=np.arange(n,dtype=int)

        def getparent(node):
            if node==p[node]:
                return node
            else:
                p[node]=getparent(p[node])
                return p[node]

        def join(u, v):
            pu=getparent(u)
            pv=getparent(v)
            if pu==pv:
                return 0
            if comps[pv]>comps[pu]:
                comps[pv]+=comps[pu]
                p[pu]=pv
            else:
                comps[pu]+=comps[pv]
                p[pv]=pu

            return 1

        sorted=np.sort(edges,order='w')

        for i in range(m):
            u=sorted[i]['u']
            v=sorted[i]['v']
            w=sorted[i]['w']
            if join(u,v):
                tree[u,v]=w
                tree[v,u]=w

        treeroot=getparent(0)

        return tree,treeroot

    tree,root=kruskal()
    tr=np.array([],dtype=int)
    vis=np.zeros(n,dtype=int)
    def preorder(root,tr):
        tr=np.append(tr,root)
        vis[root]=1
        for i in range(n):
            if tree[root,i]!=0 and vis[i]==0:
                tr = preorder(i,tr)
        return tr

    tr=preorder(root,tr)
    tr=np.append(tr,root)

    print(tr)

    curcost=float(0)

    for i in range(n):
        u=tr[i]
        v=tr[i+1]
        curcost+=adj[u,v]

    return curcost

def brute_force(adj, n):
    min_cost=[float('inf')]

    start_node=0

    all_intermediate_paths=permutations(range(1,n))

    for path in all_intermediate_paths:
        current_tour=[start_node]+list(path)+[start_node]
        current_cost=0.0
        isvalid=True

        for i in range(n):
            u=current_tour[i]
            v=current_tour[i+1]

            w=adj[u,v]

            if w==0 and u!=v:
                isvalid=False
                break

            current_cost+=w

        if isvalid:
            min_cost[0]=min(min_cost[0],current_cost)

    
    if min_cost[0]==float('inf'):
        return -1.0

    return min_cost[0]

def get_graph():
    numcities=int(input("Enter number of cities (cities will be 0 indexed) : "))
    numroads=int(input("Enter number of roads: "))

    adj=np.zeros((numcities,numcities), dtype=float)

    print("Type in roads in u,v,w format, where u and v are cities and w is the wieght we want to give the bidirectional road between them: ")
    for i in range(numroads):
        input_line=input();
        u_str,v_str,w_str=input_line.split()
        u=int(u_str)
        v=int(v_str)
        w=float(w_str)
        if adj[u,v]==0:
            adj[u,v]=w
            adj[v,u]=w
        else:
            adj[u,v]=np.min(adj[u,v],w)
            adj[v,u]=adj[u,v]

    return adj, numcities

def dp_approach(adj,n):

    memo=[[-1]*(1<<n) for x in range(n)]

    def totalcost(mask,curr):
        if mask==(1<<n)-1:
            return adj[curr][0]

        if memo[curr][mask] != -1:
            return memo[curr][mask]

        ans=float('inf')

        for i in range(n):
            if((mask&(1<<i))==0):
                ans=min(ans,adj[curr][i]+totalcost(mask|(1<<i),i))

        memo[curr][mask]=ans
        return ans

    return totalcost(1,0)

def main():
    instances = _parse_dataset()
    write_header = not os.path.exists(RUNTIME_CSV)
    with open(RUNTIME_CSV, "a", newline='') as outf:
        writer = csv.writer(outf)
        if write_header:
            writer.writerow(CSV_FIELDS)
        for name, adj, coords in instances:
            n = adj.shape[0]
            print(f"Processing {name} (n={n})")
            run_t0 = time.perf_counter()

            # prepare edges for mstapprox
            edges, m = _edges_from_adj(adj)

            t0 = time.perf_counter()
            try:
                mst_cost = mstapprox(adj, edges, n, m)
            except Exception:
                mst_cost = None
            t1 = time.perf_counter()
            mst_time = t1 - t0

            # get a tour for plotting from MST (use find_mst if available)
            try:
                tree = find_mst(adj, n)
                mst_tour = _preorder_from_tree(tree)
            except Exception:
                mst_tour = None

            t2 = time.perf_counter()
            try:
                ch_tour, ch_cost = christofides_tsp(adj, n)
            except Exception:
                ch_tour, ch_cost = None, None
            t3 = time.perf_counter()
            ch_time = t3 - t2

            # brute force (tour + cost)
            bf_t0 = time.perf_counter()
            bf_tour, bf_cost = _brute_force_tour(adj)
            bf_t1 = time.perf_counter()
            bf_time = bf_t1 - bf_t0

            # dp cost (uses dp_approach if present)
            dp_t0 = time.perf_counter()
            try:
                dp_cost = dp_approach(adj, n)
            except Exception:
                dp_cost = None
            dp_t1 = time.perf_counter()
            dp_time = dp_t1 - dp_t0

            total_time = time.perf_counter() - run_t0

            # save plots
            if mst_tour:
                _plot_tour(coords, mst_tour, os.path.join(OUT_MST, f"{name}_mst.png"), f"{name} MST-approx")
            if ch_tour:
                _plot_tour(coords, ch_tour, os.path.join(OUT_CH, f"{name}_christofides.png"), f"{name} Christofides")
            if bf_tour:
                _plot_tour(coords, bf_tour, os.path.join(OUT_BR, f"{name}_brute.png"), f"{name} Brute-force")
            # for dp use brute tour for visualization if dp doesn't provide tour
            if bf_tour:
                _plot_tour(coords, bf_tour, os.path.join(OUT_DP, f"{name}_dp.png"), f"{name} DP (visualized by brute)")

            writer.writerow([
                name, n,
                "" if mst_cost is None else f"{mst_cost:.6f}", f"{mst_time:.6f}",
                "" if ch_cost is None else f"{ch_cost:.6f}", f"{ch_time:.6f}",
                "" if bf_cost is None else f"{bf_cost:.6f}", f"{bf_time:.6f}",
                "" if dp_cost is None else f"{dp_cost:.6f}", f"{dp_time:.6f}",
                f"{total_time:.6f}"
            ])
            print(f"done {name}")

if __name__ == "__main__":
    main()
