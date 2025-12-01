import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_CSV = os.path.join(BASE_DIR, "generated", "runtime", "runtime.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "generated", "runtime", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(RUNTIME_CSV):
    print(f"runtime CSV not found: {RUNTIME_CSV}")
    sys.exit(1)

df = pd.read_csv(RUNTIME_CSV)
df.columns = [c.strip() for c in df.columns]

# candidate algorithms and their (time_col, cost_col)
alg_map = {
    "mst": ("mst_time_s", "mst_cost"),
    "christofides": ("christofides_time_s", "christofides_cost"),
    "brute": ("brute_time_s", "brute_cost"),
    "dp": ("dp_time_s", "dp_cost"),
    "total": ("total_time_s", None),
}

# coerce numeric for all columns except filename
for col in df.columns:
    if col != "filename":
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "n_nodes" not in df.columns:
    print("n_nodes column missing in runtime CSV; aborting.")
    sys.exit(1)

# determine available algorithms (must have at least the time column)
available_algs = []
for alg, (tcol, ccol) in alg_map.items():
    if tcol in df.columns or (ccol and ccol in df.columns):
        available_algs.append(alg)

# aggregate mean by n_nodes — only numeric columns
# use numeric_only=True to avoid trying to average string columns like filename
grouped = df.groupby("n_nodes", as_index=False).mean(numeric_only=True).sort_values("n_nodes")
x_nodes = grouped["n_nodes"].values
palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# 1) Time vs n_nodes (one line per available algorithm that has time column)
time_plotted = False
plt.figure(figsize=(8, 4))
for i, alg in enumerate(available_algs):
    tcol = alg_map[alg][0]
    if tcol and tcol in grouped.columns:
        plt.plot(x_nodes, grouped[tcol].values, marker="o", linestyle="-", label=alg, color=palette[i % len(palette)])
        time_plotted = True
if time_plotted:
    plt.xlabel("n_nodes")
    plt.ylabel("Mean time (s)")
    plt.title("Mean runtime vs n_nodes")
    plt.grid(alpha=0.3)
    plt.legend()
    out = os.path.join(PLOTS_DIR, "time_vs_n_nodes_lines.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("Wrote:", out)
else:
    print("No time columns available to plot time vs n_nodes.")

# 2) Cost vs n_nodes (one line per available algorithm that has cost column)
cost_plotted = False
plt.figure(figsize=(8, 4))
for i, alg in enumerate(available_algs):
    ccol = alg_map[alg][1]
    if ccol and ccol in grouped.columns:
        plt.plot(x_nodes, grouped[ccol].values, marker="o", linestyle="-", label=alg, color=palette[i % len(palette)])
        cost_plotted = True
if cost_plotted:
    plt.xlabel("n_nodes")
    plt.ylabel("Mean tour cost")
    plt.title("Mean tour cost vs n_nodes")
    plt.grid(alpha=0.3)
    plt.legend()
    out = os.path.join(PLOTS_DIR, "cost_vs_n_nodes_lines.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("Wrote:", out)
else:
    print("No cost columns available to plot cost vs n_nodes.")

# 3) Mean (cost / n_nodes) vs Mean time:
# for each algorithm that has both time and cost, plot mean(cost/n_nodes) (y) vs mean(time) (x)
pairs = []
for alg in available_algs:
    tcol, ccol = alg_map[alg]
    if tcol in grouped.columns and ccol in grouped.columns:
        x_vals = grouped[tcol].values
        # avoid division by zero
        y_vals = grouped[ccol].values / np.where(grouped["n_nodes"].values == 0, np.nan, grouped["n_nodes"].values)
        # order by x for plotting
        order = np.argsort(np.nan_to_num(x_vals, nan=np.inf))
        pairs.append((alg, x_vals[order], y_vals[order]))

if pairs:
    plt.figure(figsize=(8, 5))
    for i, (alg, xs, ys) in enumerate(pairs):
        plt.plot(xs, ys, marker="o", linestyle="-", label=alg, color=palette[i % len(palette)])
    plt.xlabel("Mean time (s)")
    plt.ylabel("Mean cost per node")
    plt.title("Mean (cost / n_nodes) vs Mean time")
    plt.grid(alpha=0.3)
    plt.legend()
    out = os.path.join(PLOTS_DIR, "avg_cost_per_node_vs_time.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("Wrote:", out)
else:
    print("No algorithm has both time and cost columns for avg(cost/n_nodes) vs time plot.")