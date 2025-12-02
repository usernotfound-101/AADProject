from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

Clause = List[int]
Formula = List[Clause]


def parse_dimacs(path: str) -> Tuple[int, Formula]:
	clauses: Formula = []
	num_vars = 0
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("c"):
				continue
			if line.startswith("p"):
				# p cnf VARS CLAUSES
				parts = line.split()
				if len(parts) >= 3 and parts[1] == "cnf":
					try:
						num_vars = int(parts[2])
					except ValueError:
						pass
				continue
			# parse clause ints, terminated by 0
			parts = line.split()
			if parts:
				ints = [int(x) for x in parts]
				clause: Clause = []
				for lit in ints:
					if lit == 0:
						break
					clause.append(lit)
				if clause:
					clauses.append(clause)
	return num_vars, clauses


def evaluate_clause(clause: Clause, assignment: Dict[int, bool]) -> bool:
	"""Return True if the clause is satisfied under the given assignment."""
	for lit in clause:
		var = abs(lit)
		val = assignment.get(var, False)
		if lit < 0:
			val = not val
		if val:
			return True
	return False


def solve_bruteforce(num_vars: int, clauses: Formula) -> Optional[Dict[int, bool]]:
	if num_vars < 0:
		num_vars = 0
	if not clauses:
		# vacuously satisfiable with any assignment
		return {i: False for i in range(1, num_vars + 1)}

	total = 1 << num_vars
	for mask in range(total):
		assignment = {var: bool((mask >> (var - 1)) & 1) for var in range(1, num_vars + 1)}
		satisfied = True
		for clause in clauses:
			if not evaluate_clause(clause, assignment):
				satisfied = False
				break
		if satisfied:
			return assignment
	return None


def pretty_assignment(assignment: Dict[int, bool]) -> str:
	return " ".join([f"x{v}={'T' if val else 'F'}" for v, val in sorted(assignment.items())])


def _parse_formula_string(formula_str: str) -> Tuple[int, Formula]:
	formula_str = formula_str.replace("(", " (").replace(")", " ) ")
	# split clauses by "and"
	groups = [g.strip() for g in formula_str.split("and") if g.strip()]
	clauses: Formula = []
	max_var = 0
	for group in groups:
		# strip enclosing parentheses
		g = group.strip()
		if g.startswith("(") and g.endswith(")"):
			g = g[1:-1]
		parts = [p.strip() for p in g.split("or") if p.strip()]
		clause: Clause = []
		for p in parts:
			p = p.strip()
			neg = False
			if p.startswith("not "):
				neg = True
				p = p[4:]
			if p.startswith("x"):
				try:
					vid = int(p[1:])
					max_var = max(max_var, vid)
					lit = -vid if neg else vid
					clause.append(lit)
				except ValueError:
					pass
		if clause:
			clauses.append(clause)
	return max_var, clauses


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description="Brute-force 3-SAT solver")
	parser.add_argument("input", nargs="?", help="DIMACS CNF file or nothing for examples")
	parser.add_argument(
		"--formula",
		help='Simple inline formula string (e.g. "(x1 or not x2 or x3) and (x2 or x3 or x4)")',
	)

	args = parser.parse_args(argv)
	if args.formula:
		num_vars, clauses = _parse_formula_string(args.formula)
	elif args.input:
		num_vars, clauses = parse_dimacs(args.input)
	else:
		# small default example: (x1 or x2 or x3) and (not x1 or not x2 or x3)
		num_vars = 3
		clauses = [[1, 2, 3], [-1, -2, 3]]

	print(f"Loaded {len(clauses)} clauses with {num_vars} variables")
	assignment = solve_bruteforce(num_vars, clauses)
	if assignment is None:
		print("UNSATISFIABLE")
		return 10
	else:
		print("SATISFIABLE")
		print(pretty_assignment(assignment))
		return 0


if __name__ == "__main__":
	raise SystemExit(main())

