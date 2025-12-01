import sys
import argparse
import time
from collections import defaultdict, deque

class MiniSATSolver:
    def __init__(self):
        self.n_vars = 0
        self.clauses = []
        self.watches = defaultdict(list)
        self.assignment = {} 
        self.reason = {}     
        self.level = {}      
        self.trail = []      
        self.decision_level = 0
        self.q_prop = deque()
        
        
        self.assumption_set = set()
        self.conflict_clause = None

    def new_var(self):
        self.n_vars += 1
        return self.n_vars

    def add_clause(self, lits):
        if not lits:
            return False 
        clause_idx = len(self.clauses)
        self.clauses.append(lits)
        
        if len(lits) == 1:
            self.watches[lits[0]].append(clause_idx)
            
            if not self.propagate_unit(lits[0]):
                return False
        else:
            self.watches[lits[0]].append(clause_idx)
            self.watches[lits[1]].append(clause_idx)
            
        return True

    def propagate_unit(self, lit, reason_idx=None):
        var = abs(lit)
        val = (lit > 0)
        
        if var in self.assignment:
            if self.assignment[var] != val:
                return False 
            return True
        
        self.assignment[var] = val
        self.level[var] = self.decision_level
        self.reason[var] = reason_idx
        self.trail.append(var)
        self.q_prop.append(var)
        return True

    def propagate(self):
        while self.q_prop:
            var = self.q_prop.popleft()
            
            
            p = var if self.assignment[var] else -var
            false_lit = -p
            
            
            notify_list = self.watches[false_lit]
            self.watches[false_lit] = []
            
            for c_idx in notify_list:
                clause = self.clauses[c_idx]
                
                
                if clause[0] == false_lit:
                    clause[0], clause[1] = clause[1], clause[0]
                first = clause[0]
                val_first = self.value(first)
                if val_first is True:
                    self.watches[false_lit].append(c_idx)
                    continue
                
                
                found_new_watch = False
                for i in range(2, len(clause)):
                    other = clause[i]
                    if self.value(other) is not False:
                        clause[1], clause[i] = clause[i], clause[1]
                        self.watches[other].append(c_idx)
                        found_new_watch = True
                        break
                
                if found_new_watch:
                    continue
                
                
                self.watches[false_lit].append(c_idx)
                
                if val_first is False:
                    
                    self.conflict_clause = c_idx
                    
                    self.q_prop.clear()
                    return False
                else:
                    
                    if not self.propagate_unit(first, c_idx):
                        self.conflict_clause = c_idx
                        self.q_prop.clear()
                        return False
        return True

    def value(self, lit):
        v = abs(lit)
        if v not in self.assignment:
            return None
        return self.assignment[v] if lit > 0 else not self.assignment[v]

    def backtrack(self, target_level):
        while len(self.trail) > 0:
            var = self.trail[-1]
            if self.level[var] <= target_level:
                break
            self.trail.pop()
            del self.assignment[var]
            del self.level[var]
            del self.reason[var]
        self.decision_level = target_level

    def analyze_conflict(self):
        if self.conflict_clause is None:
            return []

        core = set()
        seen = set()
        q = deque()
        
        
        if self.conflict_clause == -1:
            
            pass
        else:
            for lit in self.clauses[self.conflict_clause]:
                v = abs(lit)
                if v not in seen:
                    seen.add(v)
                    q.append(v)

        while q:
            var = q.popleft()
            
            if var in self.assumption_set:
                
                
                assumed_lit = var if self.assignment[var] else -var
                if assumed_lit in self.assumption_set:
                    core.add(assumed_lit)
                elif -assumed_lit in self.assumption_set:
                    core.add(-assumed_lit)
            
            
            r_idx = self.reason.get(var)
            if r_idx is not None:
                for lit in self.clauses[r_idx]:
                    v_res = abs(lit)
                    if v_res != var and v_res not in seen:
                        seen.add(v_res)
                        q.append(v_res)
                        
        return list(core)

    def solve(self, assumptions=[]):
        """
        Returns (True, None) if SAT.
        Returns (False, core) if UNSAT.
        """
        self.assumption_set = set(assumptions)
        self.backtrack(0)
        self.conflict_clause = None
        
        
        if not self.propagate():
            return False, [] 

        
        
        
        
        final_core = []
        
        for lit in assumptions:
            self.decision_level += 1
            if not self.propagate_unit(lit, reason_idx=None): 
                
                
                final_core = self.analyze_conflict()
                
                if not final_core: 
                    final_core = [lit]
                
                self.backtrack(0)
                return False, final_core
            
            if not self.propagate():
                final_core = self.analyze_conflict()
                self.backtrack(0)
                return False, final_core

        
        
        
        
        to_assign = []
        for i in range(1, self.n_vars + 1):
            if i not in self.assignment:
                to_assign.append(i)
        queue = [0] 
        
        saved_trail_len = len(self.trail)
        saved_level = self.decision_level
        
        return True, None





class Totalizer:
    def __init__(self, solver, inputs):
        self.solver = solver
        self.inputs = inputs
        self.outputs = [] 
        self.n_vars = len(inputs)
        
        
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
        
        outputs = [self.solver.new_var() for _ in range(len_out)]
        
        
        
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
        self.sat = MiniSATSolver()
        self.soft_clauses = []
        self.soft_weights = []
        self.core_selectors = {}
        
        self.top_id = 0
        self.parse_wcnf()

    def parse_wcnf(self):
        max_var = 0
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('c'): continue
                    parts = line.split()
                    
                    if line.startswith('p'): continue 
                    
                    try:
                        weight = 1
                        lits = []
                        if parts[0] == 'h':
                            lits = [int(x) for x in parts[1:] if x != '0']
                            self.sat.add_clause(lits) 
                        else:
                            weight = int(parts[0])
                            lits = [int(x) for x in parts[1:] if x != '0']
                            self.soft_clauses.append(lits)
                            self.soft_weights.append(weight)
                        
                        for x in lits: max_var = max(max_var, abs(x))
                    except ValueError: continue
            
            self.sat.n_vars = max_var
            
        except FileNotFoundError:
            print(f"Error: File {self.filename} not found.")
            sys.exit(1)

    def solve(self):
        print(f"Starting RC2 with {len(self.soft_clauses)} soft clauses...")
        start_time = time.time()
        assumptions = []
        soft_to_lit_map = {}
        
        for i, clause in enumerate(self.soft_clauses):
            sel_var = self.sat.new_var()
            self.sat.add_clause(clause + [sel_var])
            assumptions.append(-sel_var)
            soft_to_lit_map[i] = -sel_var

        cost = 0
        
        
        while True:
            
            is_sat, core = self.sat.solve(assumptions)
            
            if is_sat:
                print(f"SAT Found! Optimization Terminated.")
                print(f"Total Relaxed Cost: {cost}")
                return
            
            
            if not core:
                print("Problem is Hard-UNSAT (Constraints cannot be satisfied).")
                return

            print(f"Core found with {len(core)} literals.")
            
            core_indices = []
            min_w = float('inf')
            new_assumptions = [l for l in assumptions if l not in core]
            
            
            core_vars = [-lit for lit in core]
            
            cost += 1 
            
            
            t = Totalizer(self.sat, core_vars)
            assumptions = new_assumptions
            
            for i in range(1, len(t.outputs)):
                assumptions.append(-t.outputs[i])
            
            
            print(f"Relaxed core. Current Cost: {cost}")
            
            if cost > 100: 
                print("Breaking early (safety limit).")
                break

def main():
    parser = argparse.ArgumentParser(description="Simplified RC2 Solver")
    parser.add_argument("file", help="Path to .wcnf file")
    args = parser.parse_args()
    
    solver = RC2Solver(args.file)
    solver.solve()

if __name__ == "__main__":
    main()