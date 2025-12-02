from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import argparse
import importlib.util

# Local import: reuse DIMACS parser from brute_force
spec = importlib.util.spec_from_file_location("bf", str(Path(__file__).parent / "brute_force.py"))
if spec is None or spec.loader is None:
    raise ImportError("Could not import brute_force parser module")
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

Clause = List[int]
Formula = List[Clause]


class AssignmentEntry:
    def __init__(self, value: bool, level: int, reason: Optional[Clause]):
        self.value = value
        self.level = level
        self.reason = reason


def _is_clause_satisfied(clause: Clause, assign: Dict[int, AssignmentEntry]) -> bool:
    for lit in clause:
        a = assign.get(abs(lit))
        if a is None:
            continue
        val = a.value
        if lit < 0:
            val = not val
        if val:
            return True
    return False


def _count_unassigned_and_last(clause: Clause, assign: Dict[int, AssignmentEntry]) -> Tuple[int, Optional[int], Optional[int]]:
    unassigned = 0
    last_lit = None
    last_level = None
    for lit in clause:
        a = assign.get(abs(lit))
        if a is None:
            unassigned += 1
            last_lit = lit
        else:
            # if it's satisfied, count as satisfied -> signal via last_level
            val = a.value
            if lit < 0:
                val = not val
            if val:
                last_level = a.level
    return unassigned, last_lit, last_level


def unit_propagate(formula: Formula, assign: Dict[int, AssignmentEntry], trail: List[int], decision_level: int) -> Optional[Clause]:
    changed = True
    while changed:
        changed = False
        for clause in formula:
            if _is_clause_satisfied(clause, assign):
                continue
            unassigned, last_lit, _ = _count_unassigned_and_last(clause, assign)
            if unassigned == 0:
                # conflict
                return clause
            if unassigned == 1 and last_lit is not None:
                var = abs(last_lit)
                val = last_lit > 0
                if var in assign:
                    # Inconsistent assignment for same var
                    if assign[var].value != val:
                        return clause
                    continue
                assign[var] = AssignmentEntry(value=val, level=decision_level, reason=clause)
                trail.append(var)
                changed = True
                # Continue outer loop – since we changed the assignment.
    return None


def _all_vars_assigned(num_vars: int, assign: Dict[int, AssignmentEntry]) -> bool:
    return all(v in assign for v in range(1, num_vars + 1))


def _current_level_vars(assign: Dict[int, AssignmentEntry], level: int) -> Set[int]:
    return {v for v, e in assign.items() if e.level == level}


def _literal_level(assign: Dict[int, AssignmentEntry], lit: int) -> Optional[int]:
    entry = assign.get(abs(lit))
    if entry is None:
        return None
    return entry.level


def conflict_analysis(conflict_clause: Clause, assign: Dict[int, AssignmentEntry], decision_level: int, trail: List[int]) -> Tuple[Clause, int]:
    learned = set(conflict_clause)
    while True:
        # count literals from current decision level
        cur_level_lits = [lit for lit in learned if _literal_level(assign, lit) == decision_level]
        if len(cur_level_lits) <= 1:
            break
        # pick one to resolve (heuristic: last assigned on that level): scan trail
        pick_lit = None
        for v in reversed(trail):
            if any(abs(lit) == v for lit in cur_level_lits):
                # choose the literal whose var is v
                for lit in cur_level_lits:
                    if abs(lit) == v:
                        pick_lit = lit
                        break
                if pick_lit is not None:
                    break
        if pick_lit is None:
            break
        var = abs(pick_lit)
        reason = assign[var].reason
        if reason is None:
            # decision var; can't resolve
            break
        # resolve learned with reason on var: (A v x) resolve (B v ~x) -> A v B
        new_learned = set()
        for l in learned:
            if abs(l) == var:
                continue
            new_learned.add(l)
        for l in reason:
            if abs(l) == var:
                continue
            new_learned.add(l)
        learned = new_learned
    # compute backjump level: max level among all literals in learned except max one
    levels = [assign[abs(lit)].level for lit in learned if assign.get(abs(lit)) is not None]
    if not levels:
        backjump = 0
    else:
        backjump = 0
        if len(levels) > 1:
            # backtrack to second highest
            sorted_levels = sorted(levels)
            backjump = sorted_levels[-2]
        else:
            backjump = 0
    learned_clause = sorted(list(learned), key=lambda x: abs(x))
    return learned_clause, backjump


def backtrack(assign: Dict[int, AssignmentEntry], trail: List[int], level: int) -> None:
    # unassign variables with level > target level; the trail is in chronological order
    while trail:
        v = trail[-1]
        if assign[v].level > level:
            del assign[v]
            trail.pop()
        else:
            break


def choose_branch_var(num_vars: int, assign: Dict[int, AssignmentEntry]) -> Optional[int]:
    for v in range(1, num_vars + 1):
        if v not in assign:
            return v
    return None


def solve_cdcl(num_vars: int, formula: Formula, max_iterations: int = 100000) -> Optional[Dict[int, bool]]:
    """Run a simple CDCL solver; returns a satisfying assignment dict or None."""
    assign: Dict[int, AssignmentEntry] = {}
    trail: List[int] = []  # assignment order
    decision_level = 0
    iterations = 0
    learned_clauses: List[Clause] = []

    while True:
        iterations += 1
        if iterations > max_iterations:
            return None
        # unit propagate across original + learned
        conflict = unit_propagate(formula + learned_clauses, assign, trail, decision_level)
        if conflict is not None:
            if decision_level == 0:
                return None  # unsatisfiable
            # analyze conflict -> learn and backjump
            learned_clause, backjump_level = conflict_analysis(conflict, assign, decision_level, trail)
            learned_clauses.append(learned_clause)
            # backtrack
            backtrack(assign, trail, backjump_level)
            decision_level = backjump_level
            continue
        if _all_vars_assigned(num_vars, assign):
            # success
            return {v: assign[v].value for v in assign}
        # choose branching variable
        var = choose_branch_var(num_vars, assign)
        if var is None:
            # all variables assigned
            return {v: assign[v].value for v in range(1, num_vars + 1)}
        decision_level += 1
        # choose value True as default
        assign[var] = AssignmentEntry(value=True, level=decision_level, reason=None)
        trail.append(var)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CDCL 3-SAT solver (simple).")
    parser.add_argument("input", nargs="?", help="DIMACS CNF file path")
    parser.add_argument("--formula", help="Inline formula string (see brute_force) for quick tests")
    args = parser.parse_args(argv)
    if args.formula:
        num_vars, formula = bf._parse_formula_string(args.formula)
    elif args.input:
        num_vars, formula = bf.parse_dimacs(args.input)
    else:
        # default small sample
        num_vars, formula = bf._parse_formula_string("(x1 or x2 or x3) and (not x1 or not x2 or x3)")
    print(f"Loaded {len(formula)} clauses, {num_vars} variables")
    result = solve_cdcl(num_vars, formula)
    if result is None:
        print("UNSATISFIABLE")
        return 10
    else:
        print("SATISFIABLE")
        print(" ".join([f"x{v}={'T' if val else 'F'}" for v, val in sorted(result.items())]))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
