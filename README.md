# 📘 **P vs NP - AAD Project Repository**

This repository contains the full implementation, experiments, and documentation for our course project exploring the relationships among classical NP-complete problems. The project demonstrates the computational hardness of these problems through:

* **Exact solvers** (brute force, dynamic programming, branch-and-bound)
* **Approximation and heuristic algorithms**
* **Polynomial-time reductions between NP-complete problems**
* **Visualization tools and empirical performance analysis**

The central theme of the project is to show *how NP-completeness manifests in practice* and why efficient solutions are unlikely to exist for general instances.

---

# 🧩 **Implemented NP-Complete Problems**

We implement and analyze the following problems:

### ✔ 3-SAT & Max-SAT

* Brute-force solver
* ILP formulation
* RC2 core-guided Max-SAT solver
* Branch & Bound
* Full theoretical + empirical analysis

### ✔ Vertex Cover / Set Cover

* Brute force
* Greedy 2-approximation (Set Cover)
* Reduction from 3-SAT
* Solution quality analysis

### ✔ Hamiltonian Path (Directed & Undirected)

* Brute-force
* Held–Karp dynamic programming
* Greedy extension heuristic
* Partial-path pruning
* Simulated annealing
* Empirical comparisons

### ✔ Subset Sum

* Brute force enumeration
* Dynamic programming (pseudo-polynomial)
* Greedy approximation
* Fast PTAS-style approximation using list-trimming
* Runtime and approximation studies

### ✔ Metric TSP & General TSP

* Brute-force
* Held–Karp DP (exact)
* MST 2-approximation
* Christofides 1.5-approximation
* Discussion of why *general TSP has no polynomial-time approximation*

### ⭐ **Bonus Highlight: 3-SAT → Minesweeper Reduction**

A central novelty of our project.
We encode Boolean formulas into Minesweeper grids, showing how a familiar puzzle hides NP-complete structure.

---

# 🔁 **Reductions Implemented**

We implement several classical NP-complete reductions, including:

* **3-SAT → Vertex Cover**
* **3-SAT → Hamiltonian Path**
* **Vertex Cover → Subset Sum**
* **3-SAT → Minesweeper (highlight reduction)**

Each reduction includes:

* Mathematical correctness explanation
* Step-by-step transformation
* Visualizations of constructed gadgets
* Code to generate instances programmatically

---

# 📊 **Experiments & Benchmarking**

Every solver and approximation algorithm is benchmarked on systematically generated instances.
We evaluate:

* Runtime growth
* Memory usage
* Approximation ratios
* Success rate of heuristics
* Comparison between exact and approximate methods
* Observed vs. theoretical complexity

---

---

# 🚀 **Setup & Installation**

### **1. Clone the repository**
### **2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

Dependencies include:

* `pysat` (for RC2)
* `networkx` (for graph algorithms & TSP)
* `numpy`
* `matplotlib`
* `scipy` (optional for matching routines)

---

# ▶️ **Running Solvers**

Each problem subdirectory provides runnable scripts.


# 🙌 **Team & Acknowledgements**

Thanks to instructors, peers, and open-source contributors to PySAT, NetworkX, and BlossomV implementations.

---

# 📜 **License**

MIT License — see `LICENSE` file for details.


