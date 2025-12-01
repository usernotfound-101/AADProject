import sys
import argparse
import time

import pulp

class ILPSolver:
    def __init__(self, filename):
        self.filename = filename
        self.num_vars = 0
        self.hard_clauses = []

        self.soft_clauses = []

        self.parse_wcnf()

    def parse_wcnf(self):
        max_var_seen = 0
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('c'):
                        continue

                    parts = line.split()

                    if line.startswith('p'):
                        if len(parts) >= 3:

                            pass
                        continue

                    try:
                        lits = []
                        weight = 1
                        is_hard = False

                        if parts[0] == 'h':
                            is_hard = True

                            lits = [int(x) for x in parts[1:] if x != '0']
                        else:

                            weight = int(parts[0])
                            lits = [int(x) for x in parts[1:] if x != '0']

                        for x in lits:
                            max_var_seen = max(max_var_seen, abs(x))

                        if is_hard:
                            self.hard_clauses.append(lits)
                        else:
                            self.soft_clauses.append({'lits': lits, 'weight': weight})

                    except ValueError:
                        continue

            self.num_vars = max_var_seen

        except FileNotFoundError:
            print(f"Error: File {self.filename} not found.")
            sys.exit(1)

    def solve(self):
        print(f"Building ILP Model for {self.filename}...")
        print(f"Variables: {self.num_vars}")
        print(f"Hard Clauses: {len(self.hard_clauses)}")
        print(f"Soft Clauses: {len(self.soft_clauses)}")

        start_time = time.time()

        prob = pulp.LpProblem("MaxSAT_ILP", pulp.LpMinimize)

        sat_vars = {}
        for i in range(1, self.num_vars + 1):
            sat_vars[i] = pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary)

        relax_vars = []
        for i, clause in enumerate(self.soft_clauses):
            u = pulp.LpVariable(f"u_{i}", cat=pulp.LpBinary)
            relax_vars.append(u)

        def get_lit_expr(lit):
            v_idx = abs(lit)
            if lit > 0:
                return sat_vars[v_idx]
            else:
                return 1 - sat_vars[v_idx]

        for i, lits in enumerate(self.hard_clauses):
            expr_items = [get_lit_expr(lit) for lit in lits]
            prob += pulp.lpSum(expr_items) >= 1, f"Hard_{i}"

        total_cost_expr = []

        for i, clause in enumerate(self.soft_clauses):
            lits = clause['lits']
            weight = clause['weight']
            u_var = relax_vars[i]

            expr_items = [get_lit_expr(lit) for lit in lits]

            prob += pulp.lpSum(expr_items) + u_var >= 1, f"Soft_{i}"

            total_cost_expr.append(weight * u_var)

        prob += pulp.lpSum(total_cost_expr), "Total_Weight_Unsatisfied"

        print("Model built. Sending to solver (CBC)...")

        solver_engine = pulp.PULP_CBC_CMD(msg=True)

        prob.solve(solver_engine)

        solve_time = time.time() - start_time
        print(f"\nSolver Status: {pulp.LpStatus[prob.status]}")
        print(f"Time: {solve_time:.2f}s")

        if pulp.LpStatus[prob.status] == 'Optimal':
            obj_val = pulp.value(prob.objective)
            print(f"Optimal Cost (Weight of Unsatisfied Clauses): {obj_val}")

            parts = []
            for i in range(1, self.num_vars + 1):
                val = pulp.value(sat_vars[i])

                if val is not None and val > 0.5:
                    parts.append(str(i))
                else:
                    parts.append(str(-i))

            print("v " + " ".join(parts) + " 0")
        else:
            print("No optimal solution found (Model might be infeasible if hard clauses conflict).")

def main():
    parser = argparse.ArgumentParser(description="ILP MaxSAT Solver using PuLP")
    parser.add_argument("file", help="Path to .wcnf file")
    args = parser.parse_args()

    solver = ILPSolver(args.file)
    solver.solve()

if __name__ == "__main__":
    main()