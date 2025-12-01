import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "generated", "runtime", "summary.csv")
OUT_DIR = os.path.join(BASE_DIR, "generated", "runtime", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"CSV not found: {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# Ensure numeric coercion for relevant columns
for c in ["n_nodes", "mst_time_s", "christofides_time_s", "mst_cost", "christofides_cost"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "n_nodes" not in df.columns:
    print("n_nodes column missing in CSV")
    sys.exit(1)

# Aggregate by n_nodes (mean) to ensure one value per n_nodes
grouped = df.groupby("n_nodes", as_index=False).mean(numeric_only=True).sort_values("n_nodes")
x = grouped["n_nodes"].values
palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# 1) Time vs n_nodes (mst and christofides)
time_cols = []
if "mst_time_s" in grouped.columns:
    time_cols.append(("mst", "mst_time_s"))
if "christofides_time_s" in grouped.columns:
    time_cols.append(("christofides", "christofides_time_s"))

if time_cols:
    plt.figure(figsize=(8,4))
    for i, (label, col) in enumerate(time_cols):
        y = grouped[col].values
        plt.plot(x, y, marker="o", linestyle="-", label=label, color=palette[i % len(palette)])
    plt.xlabel("n_nodes")
    plt.ylabel("Mean time (s)")
    plt.title("Mean runtime vs n_nodes")
    plt.grid(alpha=0.3)
    plt.legend()
    out1 = os.path.join(OUT_DIR, "time_vs_n_nodes.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()
    print("Wrote:", out1)
else:
    print("No time columns (mst_time_s or christofides_time_s) present to plot.")

# 2) Score (cost) vs n_nodes (mst and christofides)
cost_cols = []
if "mst_cost" in grouped.columns:
    cost_cols.append(("mst", "mst_cost"))
if "christofides_cost" in grouped.columns:
    cost_cols.append(("christofides", "christofides_cost"))

if cost_cols:
    plt.figure(figsize=(8,4))
    for i, (label, col) in enumerate(cost_cols):
        y = grouped[col].values
        plt.plot(x, y, marker="o", linestyle="-", label=label, color=palette[i % len(palette)])
    plt.xlabel("n_nodes")
    plt.ylabel("Mean tour cost")
    plt.title("Mean tour cost vs n_nodes")
    plt.grid(alpha=0.3)
    plt.legend()
    out2 = os.path.join(OUT_DIR, "cost_vs_n_nodes.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Wrote:", out2)
else:
    print("No cost columns (mst_cost or christofides_cost) present to plot.")

# Optional: also save a small CSV with the grouped values used for plotting
summary_out = os.path.join(OUT_DIR, "grouped_by_n_nodes.csv")
grouped.to_csv(summary_out, index=False)
print("Wrote grouped CSV:", summary_out)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "generated", "runtime", "summary.csv")
OUT_DIR = os.path.join(BASE_DIR, "generated", "runtime", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"CSV not found: {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# Ensure numeric coercion for relevant columns
for c in ["n_nodes", "mst_time_s", "christofides_time_s", "mst_cost", "christofides_cost"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "n_nodes" not in df.columns:
    print("n_nodes column missing in CSV")
    sys.exit(1)

# Aggregate by n_nodes (mean) to ensure one value per n_nodes
grouped = df.groupby("n_nodes", as_index=False).mean(numeric_only=True).sort_values("n_nodes")
x = grouped["n_nodes"].values
palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# 1) Time vs n_nodes (mst and christofides)
time_cols = []
if "mst_time_s" in grouped.columns:
    time_cols.append(("mst", "mst_time_s"))
if "christofides_time_s" in grouped.columns:
    time_cols.append(("christofides", "christofides_time_s"))

if time_cols:
    plt.figure(figsize=(8,4))
    for i, (label, col) in enumerate(time_cols):
        y = grouped[col].values
        plt.plot(x, y, marker="o", linestyle="-", label=label, color=palette[i % len(palette)])
    plt.xlabel("n_nodes")
    plt.ylabel("Mean time (s)")
    plt.title("Mean runtime vs n_nodes")
    plt.grid(alpha=0.3)
    plt.legend()
    out1 = os.path.join(OUT_DIR, "time_vs_n_nodes.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()
    print("Wrote:", out1)
else:
    print("No time columns (mst_time_s or christofides_time_s) present to plot.")

# 2) Score (cost) vs n_nodes (mst and christofides)
cost_cols = []
if "mst_cost" in grouped.columns:
    cost_cols.append(("mst", "mst_cost"))
if "christofides_cost" in grouped.columns:
    cost_cols.append(("christofides", "christofides_cost"))

if cost_cols:
    plt.figure(figsize=(8,4))
    for i, (label, col) in enumerate(cost_cols):
        y = grouped[col].values
        plt.plot(x, y, marker="o", linestyle="-", label=label, color=palette[i % len(palette)])
    plt.xlabel("n_nodes")
    plt.ylabel("Mean tour cost")
    plt.title("Mean tour cost vs n_nodes")
    plt.grid(alpha=0.3)
    plt.legend()
    out2 = os.path.join(OUT_DIR, "cost_vs_n_nodes.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Wrote:", out2)
else:
    print("No cost columns (mst_cost or christofides_cost) present to plot.")

# Optional: also save a small CSV with the grouped values used for plotting
summary_out = os.path.join(OUT_DIR, "grouped_by_n_nodes.csv")
grouped.to_csv(summary_out, index=False)
print("Wrote grouped CSV:", summary_out)