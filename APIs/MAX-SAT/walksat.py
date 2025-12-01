import sys
import random
import time
import argparse

class WCNFSolver:
    def __init__(self, filename):
        self.filename = filename
        self.num_vars = 0
        self.num_clauses = 0
        self.top_weight = 0
        
        # Internal storage
        self.clauses = [] # List of sets for O(1) lookup
        self.weights = [] # Parallel list to clauses
        
        # Parse the file immediately upon initialization
        self.parse_wcnf()

    def parse_wcnf(self):
        """
        Parses DIMACS WCNF format.
        Handles missing headers and 'h' hard clause indicators.
        """
        max_var_seen = 0
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('c'):
                        continue
                    
                    parts = line.split()
                    
                    # Handle header if it exists
                    if line.startswith('p'):
                        if len(parts) >= 5 and parts[1] == 'wcnf':
                            self.num_vars = int(parts[2])
                            self.num_clauses = int(parts[3])
                            self.top_weight = int(parts[4])
                        continue
                        
                    # Parse clause line
                    # standard: weight lit1 lit2 ... 0
                    # special: h lit1 lit2 ... 0
                    try:
                        lits_parts = []
                        weight = 1 # Default for unweighted logic
                        
                        if parts[0] == 'h':
                            # Hard clause marked with 'h'
                            # In unweighted mode, we just treat it as a clause to satisfy
                            lits_parts = parts[1:]
                        else:
                            # Standard format: first token is weight
                            # We parse it, though current mode ignores specific weights
                            weight = int(parts[0])
                            lits_parts = parts[1:]
                        
                        lits = [int(x) for x in lits_parts if x != '0'] # Remove trailing 0
                        
                        # Track max variable index to infer num_vars if header is missing
                        for lit in lits:
                            abs_lit = abs(lit)
                            if abs_lit > max_var_seen:
                                max_var_seen = abs_lit

                        self.clauses.append(lits)
                        self.weights.append(weight)
                            
                    except ValueError:
                        continue
            
            # If header was missing or incorrect, use the max variable found
            if self.num_vars == 0:
                self.num_vars = max_var_seen

        except FileNotFoundError:
            print(f"Error: File {self.filename} not found.")
            sys.exit(1)

    def evaluate_clause(self, clause, assignment):
        """Returns True if the clause is satisfied by the assignment."""
        for lit in clause:
            # Assignment is 0-indexed, lit is 1-based index
            var_idx = abs(lit) - 1
            if var_idx >= len(assignment):
                # Safety check against malformed input or logic errors
                continue
                
            is_true = assignment[var_idx] if lit > 0 else not assignment[var_idx]
            if is_true:
                return True
        return False

    def calculate_total_score(self, assignment):
        score = 0
        for i, clause in enumerate(self.clauses):
            if self.evaluate_clause(clause, assignment):
                score += 1  # Unweighted: count 1 for every satisfied clause
        return score

    def solve(self, max_flips=100000, max_tries=10, noise_prob=0.5):
        """
        Executes WalkSAT.
        
        Strategy (Unweighted):
        1. Pick an Unsatisfied Clause randomly.
        2. Pick a variable in that clause to flip:
           - With prob `noise_prob`: Pick random variable.
           - Else: Pick variable that maximizes number of satisfied clauses.
        """
        if self.num_vars == 0:
            print("Error: No variables found in problem definition.")
            return None, 0

        best_overall_assignment = None
        best_overall_score = -1

        start_time = time.time()
        
        print(f"Solving {self.filename}...")
        print(f"Vars: {self.num_vars}, Total Clauses: {len(self.clauses)}")

        for try_num in range(max_tries):
            # Random initialization
            assignment = [random.choice([True, False]) for _ in range(self.num_vars)]
            
            # Initial evaluation
            current_score = self.calculate_total_score(assignment)
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_assignment = list(assignment)

            for flip in range(max_flips):
                
                # Identify unsatisfied clauses
                unsat_indices = []
                for i, c in enumerate(self.clauses):
                    if not self.evaluate_clause(c, assignment):
                        unsat_indices.append(i)

                if not unsat_indices:
                    # All satisfied
                    print(f"  Perfect Solution Found (Try {try_num+1}, Flip {flip})")
                    return best_overall_assignment, best_overall_score

                # Progress logging
                if flip % 1000 == 0 and flip > 0:
                     if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_overall_assignment = list(assignment)
                        print(f"  New Best Score: {best_overall_score}/{len(self.clauses)} (Try {try_num+1}, Flip {flip})")

                target_idx = random.choice(unsat_indices)
                target_clause = self.clauses[target_idx]

                # Select Variable to Flip
                var_to_flip = None

                if random.random() < noise_prob:
                    # Random Walk
                    lit_to_flip = random.choice(target_clause)
                    var_to_flip = abs(lit_to_flip) - 1
                else:
                    # Greedy move
                    # Find flip that maximizes TOTAL satisfied clauses
                    
                    best_score_gain = float('-inf')
                    candidates = []

                    for lit in target_clause:
                        v_idx = abs(lit) - 1
                        
                        # Temporarily flip
                        assignment[v_idx] = not assignment[v_idx]
                        
                        # Calculate new score
                        new_score = self.calculate_total_score(assignment)
                        
                        if new_score > best_score_gain:
                            best_score_gain = new_score
                            candidates = [v_idx]
                        elif new_score == best_score_gain:
                            candidates.append(v_idx)

                        # Flip back
                        assignment[v_idx] = not assignment[v_idx]
                    
                    if candidates:
                        var_to_flip = random.choice(candidates)
                    else:
                        lit_to_flip = random.choice(target_clause)
                        var_to_flip = abs(lit_to_flip) - 1

                # Perform the flip
                assignment[var_to_flip] = not assignment[var_to_flip]
                
                # Update score efficiently (recalc is slow but safe for this version)
                current_score = self.calculate_total_score(assignment)
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_assignment = list(assignment)

        elapsed = time.time() - start_time
        print(f"\nFinished in {elapsed:.2f}s")
        return best_overall_assignment, best_overall_score

def main():
    parser = argparse.ArgumentParser(description="WalkSAT Solver for WCNF (MaxSAT)")
    parser.add_argument("file", help="Path to .wcnf file")
    parser.add_argument("--flips", type=int, default=1000, help="Max flips per try")
    parser.add_argument("--tries", type=int, default=5, help="Number of restarts")
    parser.add_argument("--noise", type=float, default=0.5, help="Probability of random walk (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    solver = WCNFSolver(args.file)
    assignment, score = solver.solve(max_flips=args.flips, max_tries=args.tries, noise_prob=args.noise)
    
    if assignment:
        print(f"\nBest Solution Found (Satisfied Clauses: {score}/{len(solver.clauses)})")
        # Print in DIMACS format: v 1 -2 3 ... 0
        output_parts = []
        for i, val in enumerate(assignment):
            # i is 0-based index, variable is i+1
            lit = (i + 1) if val else -(i + 1)
            output_parts.append(str(lit))
        print("v " + " ".join(output_parts))
    else:
        print("\nNo feasible solution found.")

if __name__ == "__main__":
    main()