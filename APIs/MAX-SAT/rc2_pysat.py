import sys
import argparse
import time

try:
    from pysat.solvers import Solver
    from pysat.formula import IDPool
except ImportError:
    print("Error: This script requires the 'python-sat' library.")
    print("Please install it using: pip install python-sat")
    sys.exit(1)

class Totalizer:
    """
    A simple implementation of the Totalizer encoding for cardinality constraints.
    Used to count how many relaxation variables are set to True.
    """
    def __init__(self, solver, inputs, vpool):
        self.solver = solver
        self.inputs = inputs
        self.vpool = vpool
        self.outputs = []

        self.outputs = self._build(inputs)

    def _build(self, inputs):
        n = len(inputs)
        if n == 0:
            return []
        if n == 1:
            return inputs

        mid = n // 2
        left = self._build(inputs[:mid])
        right = self._build(inputs[mid:])

        return self._merge(left, right)

    def _merge(self, left, right):
        len_a = len(left)
        len_b = len(right)
        len_out = len_a + len_b

        outputs = [self.vpool.id() for _ in range(len_out)]

        for i in range(len_a):
            self.solver.add_clause([-left[i], outputs[i]])

        for j in range(len_b):
            self.solver.add_clause([-right[j], outputs[j]])

        for i in range(len_a):
            for j in range(len_b):
                self.solver.add_clause([-left[i], -right[j], outputs[i + j + 1]])

        return outputs

class RC2Solver:
    def __init__(self, filename):
        self.filename = filename

        self.sat = Solver(name='g3')
        self.vpool = IDPool()

        self.soft_clauses = []

        self.soft_weights = []

        self.max_file_var = 0

        self.parse_wcnf()

        if self.max_file_var > 0:

            for i in range(1, self.max_file_var + 1):
                self.vpool.id(i)

    def parse_wcnf(self):
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('c'): continue
                    parts = line.split()

                    if line.startswith('p'): continue

                    try:

                        lits = []
                        weight = 1

                        if parts[0] == 'h':

                            lits = [int(x) for x in parts[1:] if x != '0']
                            self.sat.add_clause(lits)
                        else:

                            weight = int(parts[0])
                            lits = [int(x) for x in parts[1:] if x != '0']
                            self.soft_clauses.append(lits)
                            self.soft_weights.append(weight)

                        for x in lits:

                            self.max_file_var = max(self.max_file_var, abs(x))

                    except ValueError: continue

        except FileNotFoundError:
            print(f"Error: File {self.filename} not found.")
            sys.exit(1)

    def solve(self):
        print(f"Starting RC2 with {len(self.soft_clauses)} soft clauses...")
        start_time = time.time()

        assumptions = []

        for i, lits in enumerate(self.soft_clauses):
            sel_var = self.vpool.id()

            self.sat.add_clause(lits + [sel_var])

            assumptions.append(-sel_var)

        cost = 0

        print(f"Solver initialized. Variables: {self.vpool.top}")

        while True:

            is_sat = self.sat.solve(assumptions=assumptions)

            if is_sat:
                print(f"\nSAT Found!")
                print(f"Optimal Cost: {cost}")
                print(f"Time: {time.time() - start_time:.2f}s")

                model = self.sat.get_model()

                final_model = [l for l in model if abs(l) <= self.max_file_var]

                output_parts = [str(l) for l in final_model]
                print("v " + " ".join(output_parts) + " 0")
                return

            core = self.sat.get_core()

            if not core:
                print("\nProblem is Hard-UNSAT (Constraints cannot be satisfied).")
                return

            print(f"Core found: size {len(core)}")

            increment = 1

            cost += increment

            core_set = set(core)
            assumptions = [l for l in assumptions if l not in core_set]

            relax_vars = [-l for l in core]

            t = Totalizer(self.sat, relax_vars, self.vpool)

            for i in range(1, len(t.outputs)):
                assumptions.append(-t.outputs[i])

            if cost > 1000:
                print("Limit reached.")
                break

def main():
    parser = argparse.ArgumentParser(description="RC2 Solver using PySAT")
    parser.add_argument("file", help="Path to .wcnf file")
    args = parser.parse_args()

    solver = RC2Solver(args.file)
    solver.solve()

if __name__ == "__main__":
    main()