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

# algorithm columns mapping: (time_col, cost_col)
alg_map = {
    "mst": ("mst_time_s", "mst_cost"),
    "christofides": ("christofides_time_s", "christofides_cost"),
    "brute": ("brute_time_s", "brute_cost"),
    "dp": ("dp_time_s", "dp_cost"),
}

# collect available columns
available_algs = []
for alg, (tcol, ccol) in alg_map.items():
    if (tcol in df.columns) or (ccol in df.columns):
        available_algs.append(alg)

if not available_algs:
    print("No algorithm columns found in runtime CSV.")
    sys.exit(1)

# coerce numeric where present
for _, (tcol, ccol) in alg_map.items():
    if tcol in df.columns:
        df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    if ccol in df.columns:
        df[ccol] = pd.to_numeric(df[ccol], errors="coerce")

# need n_nodes
if "n_nodes" not in df.columns:
    print("n_nodes column missing in runtime CSV.")
    sys.exit(1)
df["n_nodes"] = pd.to_numeric(df["n_nodes"], errors="coerce")
df = df.dropna(subset=["n_nodes"])
df["n_nodes"] = df["n_nodes"].astype(int)

# aggregate mean per n_nodes
grouped = df.groupby("n_nodes").mean().reset_index().sort_values("n_nodes")

x = grouped["n_nodes"].values
palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# 1) Time vs n_nodes (one line per algorithm)
plt.figure(figsize=(7,4))
for i, alg in enumerate(available_algs):
    tcol = alg_map[alg][0]
    if tcol in grouped.columns:
        y = grouped[tcol].values
        plt.plot(x, y, marker='o', linestyle='-', label=alg, color=palette[i % len(palette)])
plt.xlabel("n_nodes")
plt.ylabel("Mean time (s)")
plt.title("Mean runtime vs n_nodes")
plt.grid(alpha=0.3)
plt.legend()
out1 = os.path.join(PLOTS_DIR, "time_vs_n_nodes_lines.png")
plt.tight_layout()
plt.savefig(out1, dpi=150)
plt.close()
print("Wrote:", out1)

# 2) Cost vs n_nodes (one line per algorithm)
plt.figure(figsize=(7,4))
any_cost = False
for i, alg in enumerate(available_algs):
    ccol = alg_map[alg][1]
    if ccol in grouped.columns:
        any_cost = True
        y = grouped[ccol].values
        plt.plot(x, y, marker='o', linestyle='-', label=alg, color=palette[i % len(palette)])
if not any_cost:
    print("No cost columns available to plot.")
else:
    plt.xlabel("n_nodes")
    plt.ylabel("Mean tour cost")
    plt.title("Mean tour cost vs n_nodes")
    plt.grid(alpha=0.3)
    plt.legend()
    out2 = os.path.join(PLOTS_DIR, "cost_vs_n_nodes_lines.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Wrote:", out2)

# optional: write aggregated CSV of means for inspection
agg_cols = []
for alg in available_algs:
    tcol, ccol = alg_map[alg]
    if tcol in grouped.columns:
        agg_cols.append(tcol)
    if ccol in grouped.columns:
        agg_cols.append(ccol)
summary_path = os.path.join(PLOTS_DIR, "runtime_cost_means_by_n_nodes.csv")
grouped[["n_nodes"] + [c for c in agg_cols if c in grouped.columns]].to_csv(summary_path, index=False)
print("Wrote summary CSV:", summary_path)