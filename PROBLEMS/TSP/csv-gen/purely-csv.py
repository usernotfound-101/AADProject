import os
import time
import csv
import math
import numpy as np

# run from this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _coords_from_string(s):
    import json, ast, re
    if not s:
        return []
    s = s.strip()
    # try JSON
    try:
        obj = json.loads(s)
        return [(float(p[0]), float(p[1])) for p in obj]
    except Exception:
        pass
    # try python literal
    try:
        obj = ast.literal_eval(s)
        return [(float(p[0]), float(p[1])) for p in obj]
    except Exception:
        pass
    # fallback: extract numbers
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', s)
    if len(nums) % 2 != 0:
        return []
    return [(float(nums[i]), float(nums[i+1])) for i in range(0, len(nums), 2)]

def build_adj_edges(coords):
    n = len(coords)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            adj[i, j] = d
            adj[j, i] = d
    edge_list = [(i, j, adj[i, j]) for i in range(n) for j in range(i + 1, n)]
    edges = np.array(edge_list, dtype=[('u','i4'),('v','i4'),('w','f8')])
    return adj, edges, n, len(edge_list)

def parse_tsp_file(filename):
    """
    If filename is a .tsp -> return a single instance: (name, coords, adj, edges, n, m)
    If filename is a .csv with columns like instance_id,num_cities,city_coordinates,... ->
      yields multiple instances as (name, coords, adj, edges, n, m) for each row.
    """
    if filename.lower().endswith(".tsp"):
        coords = []
        with open(filename) as fh:
            reading = False
            for line in fh:
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
                _, x, y = parts[:3]
                try:
                    coords.append((float(x), float(y)))
                except Exception:
                    continue
        adj, edges, n, m = build_adj_edges(coords)
        name = os.path.splitext(os.path.basename(filename))[0]
        return [(name, coords, adj, edges, n, m)]

    # CSV handling (may contain multiple instances)
    import csv as _csv, sys
    try:
        _csv.field_size_limit(sys.maxsize)
    except OverflowError:
        _csv.field_size_limit(10**9)

    instances = []
    with open(filename, newline='') as fh:
        reader = _csv.DictReader(fh)
        # normalize fieldnames
        fns = [fn.strip().lower() for fn in (reader.fieldnames or [])]

        # preferred column that contains coordinates
        coord_key = None
        for key in reader.fieldnames or []:
            lk = key.strip().lower()
            if lk in ("city_coordinates", "citycoordinates", "coordinates", "city_coords", "citycoords"):
                coord_key = key
                break

        # if there's a per-row coordinates column, process each row as one instance
        if coord_key:
            for row in reader:
                s = row.get(coord_key)
                coords = _coords_from_string(s)
                if not coords:
                    continue
                # choose name: instance_id if present else use file + row index
                name = None
                for possible in ("instance_id", "id", "name"):
                    for k in reader.fieldnames:
                        if k.strip().lower() == possible:
                            name = str(row.get(k)).strip()
                            break
                    if name:
                        break
                if not name:
                    name = f"{os.path.splitext(os.path.basename(filename))[0]}_{len(instances)}"
                adj, edges, n, m = build_adj_edges(coords)
                instances.append((name, coords, adj, edges, n, m))
            return instances

        # otherwise try to interpret CSV as simple coordinate list (one point per row)
        # fall back: read numeric pairs from rows until end -> single instance
        coords = []
        fh.seek(0)
        reader2 = _csv.reader(fh)
        for row in reader2:
            if not row:
                continue
            fields = [c for c in row if c != "" and c is not None]
            if len(fields) < 2:
                continue
            try:
                x = float(fields[-2]); y = float(fields[-1])
                coords.append((x, y))
            except Exception:
                continue
        if coords:
            name = os.path.splitext(os.path.basename(filename))[0]
            adj, edges, n, m = build_adj_edges(coords)
            return [(name, coords, adj, edges, n, m)]

    # if nothing parsed, return empty
    return []

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
    T = np.zeros((n, n))
    for u, v, w in sorted(edges, key=lambda x: x[2]):
        if union(u, v):
            T[u, v] = T[v, u] = w
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

def main():
    input_dir = BASE_DIR
    out_base = os.path.join(BASE_DIR, "generated")
    out_per_instance = os.path.join(out_base, "csv")
    out_runtime = os.path.join(out_base, "runtime")

    os.makedirs(out_per_instance, exist_ok=True)
    os.makedirs(out_runtime, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tsp", ".csv"))]
    summary_path = os.path.join(out_runtime, "summary.csv")
    fields = ["filename", "n_nodes", "mst_cost", "mst_time_s", "christofides_cost", "christofides_time_s", "total_time_s"]

    summary_rows = []

    for fname in files:
        filepath = os.path.join(input_dir, fname)
        instances = parse_tsp_file(filepath)
        if not instances:
            print(f"Skipping {fname}: no instances parsed")
            continue

        for (name, coords, adj, edges, n, m) in instances:
            t0 = time.perf_counter()
            mst_tour, mst_cost = mstapprox(adj, edges, n)
            t1 = time.perf_counter()
            mst_time = t1 - t0

            t2 = time.perf_counter()
            ch_tour, ch_cost = christofides_tsp(adj, n)
            t3 = time.perf_counter()
            ch_time = t3 - t2

            total_time = t3 - t0

            row = {
                "filename": name,
                "n_nodes": n,
                "mst_cost": mst_cost,
                "mst_time_s": f"{mst_time:.6f}",
                "christofides_cost": ch_cost,
                "christofides_time_s": f"{ch_time:.6f}",
                "total_time_s": f"{total_time:.6f}",
            }
            summary_rows.append(row)

            per_path = os.path.join(out_per_instance, f"{name}.csv")
            with open(per_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote per-instance CSVs to: {out_per_instance}")
    print(f"Wrote summary CSV to: {summary_path}")

if __name__ == "__main__":
    main()