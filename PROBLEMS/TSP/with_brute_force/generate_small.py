import os
import json
import csv
import random
import argparse

"""
Generate small TSP instances (<= 10 cities). Outputs a CSV with columns:
instance_id,num_cities,city_coordinates,distance_matrix

- city_coordinates: JSON list of [x,y] (useful if Euclidean)
- distance_matrix: JSON square matrix (symmetric, zero diagonal). Can be Euclidean-derived
  or fully random (non-Euclidean). Default mixes both.
"""
random.seed(42)
def make_euclidean_coords(n, scale=100.0):
    return [[random.uniform(0, scale), random.uniform(0, scale)] for _ in range(n)]

def euclidean_matrix(coords):
    n = len(coords)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i+1, n):
            xj, yj = coords[j]
            d = ((xi-xj)**2 + (yi-yj)**2)**0.5
            mat[i][j] = mat[j][i] = d
    return mat

def random_symmetric_matrix(n, low=1.0, high=100.0, integer=False):
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            v = random.randint(int(low), int(high)) if integer else random.uniform(low, high)
            mat[i][j] = mat[j][i] = v
    return mat

def mix_matrix(euc_mat, prob_replace=0.3, low=1.0, high=100.0):
    n = len(euc_mat)
    mat = [row[:] for row in euc_mat]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < prob_replace:
                mat[i][j] = mat[j][i] = random.uniform(low, high)
    return mat

def generate_instances(count=50, max_cities=10, out_path="small_tsp_dataset.csv",
                       mode="mix", mix_replace_prob=0.3, integer_dist=False, seed=None):
    if seed is not None:
        random.seed(seed)
    rows = []
    for idx in range(count):
        n = random.randint(2, max(2, max_cities))
        coords = make_euclidean_coords(n)
        if mode == "euclidean":
            mat = euclidean_matrix(coords)
        elif mode == "random":
            mat = random_symmetric_matrix(n, integer=integer_dist)
        else:  # mix
            euc = euclidean_matrix(coords)
            mat = mix_matrix(euc, prob_replace=mix_replace_prob)
            if integer_dist:
                # optionally round/make integer
                mat = [[int(round(v)) if i != j else 0 for j, v in enumerate(row)] for i, row in enumerate(mat)]
        rows.append({
            "instance_id": idx,
            "num_cities": n,
            "city_coordinates": json.dumps(coords),
            "distance_matrix": json.dumps(mat)
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=["instance_id", "num_cities", "city_coordinates", "distance_matrix"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_path

def main():
    p = argparse.ArgumentParser(description="Generate small TSP instances (<=10 cities).")
    p.add_argument("--count", type=int, default=50, help="number of instances to generate")
    p.add_argument("--max-cities", type=int, default=10, help="maximum cities per instance (>=2)")
    p.add_argument("--out", default=None, help="output CSV path (default: datasets/small_tsp_dataset.csv or ./small_tsp_dataset.csv)")
    p.add_argument("--mode", choices=("euclidean","random","mix"), default="mix",
                   help="euclidean: use coords-derived distances; random: fully random symmetric matrix; mix: mostly euclidean with some replaced")
    p.add_argument("--mix-replace-prob", type=float, default=0.3, help="for mix mode, probability to replace a distance with random value")
    p.add_argument("--integer-dist", action="store_true", help="round distances to integers")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    args = p.parse_args()

    # decide default output path: prefer ../datasets if it exists in repo layout
    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    if not os.path.isdir(default_dir):
        default_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(default_dir, "small_tsp_dataset.csv")

    path = generate_instances(count=args.count,
                              max_cities=min(10, max(2, args.max_cities)),
                              out_path=out_path,
                              mode=args.mode,
                              mix_replace_prob=args.mix_replace_prob,
                              integer_dist=args.integer_dist,
                              seed=args.seed)
    print(f"Wrote {args.count} instances to: {path}")

if __name__ == "__main__":
    main()