#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>

using namespace std;

// --- Data Structures ---

struct SoftClause {
    vector<int> lits;
    long long weight;
    int id;
};

// 0: Unassigned, 1: True, -1: False
// We use 1-based indexing for variables, so assignment[0] is unused.
vector<int> assignment; 
int num_vars = 0;

vector<vector<int>> hard_clauses;
vector<SoftClause> soft_clauses;

// Adjacency list: var -> list of hard clause indices where var appears
// Used for Unit Propagation optimization (simplified)
vector<vector<int>> hard_watch; 

long long best_cost = numeric_limits<long long>::max();
vector<int> best_assignment;
long long nodes_explored = 0;
auto start_time = chrono::high_resolution_clock::now();
double time_limit = -1.0; // Seconds, -1 means no limit

// --- Parsing ---

void parse_wcnf(const string& filename) {
    ifstream infile(filename);
    if (!infile.good()) {
        cerr << "Error: File not found." << endl;
        exit(1);
    }

    string line;
    int max_var = 0;
    int soft_id = 0;

    while (getline(infile, line)) {
        if (line.empty() || line[0] == 'c') continue;

        if (line[0] == 'p') {
            // Header parsing: p wcnf <vars> <clauses> <top>
            stringstream ss(line);
            string tmp;
            ss >> tmp >> tmp >> num_vars;
            continue;
        }

        stringstream ss(line);
        string first_token;
        ss >> first_token;

        if (first_token.empty()) continue;

        vector<int> lits;
        bool is_hard = false;
        long long weight = 1;

        if (first_token == "h") {
            is_hard = true;
        } else {
            try {
                weight = stoll(first_token);
            } catch (...) {
                continue; 
            }
        }

        int lit;
        while (ss >> lit && lit != 0) {
            lits.push_back(lit);
            if (abs(lit) > max_var) max_var = abs(lit);
        }

        if (is_hard) {
            hard_clauses.push_back(lits);
        } else {
            soft_clauses.push_back({lits, weight, soft_id++});
        }
    }

    if (num_vars == 0) num_vars = max_var;
    assignment.resize(num_vars + 1, 0);
    best_assignment.resize(num_vars + 1, 0);
}

// --- Logic ---

// Evaluate a literal under current assignment
// Returns: 1 (True), -1 (False), 0 (Unassigned)
inline int eval_lit(int lit) {
    int var = abs(lit);
    int val = assignment[var];
    if (val == 0) return 0;
    if (lit > 0) return val;       // lit=5, val=1 -> True (1)
    return -val;                   // lit=-5, val=1 -> False (-1)
}

// Perform Unit Propagation on Hard Clauses
// Returns true if no conflict, false if conflict
bool unit_propagate(vector<int>& trail) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& clause : hard_clauses) {
            int unassigned_lit = 0;
            int unassigned_count = 0;
            bool satisfied = false;

            for (int lit : clause) {
                int status = eval_lit(lit);
                if (status == 1) {
                    satisfied = true;
                    break;
                }
                if (status == 0) {
                    unassigned_lit = lit;
                    unassigned_count++;
                }
            }

            if (satisfied) continue;

            if (unassigned_count == 0) {
                return false; // Conflict: all lits false
            }

            if (unassigned_count == 1) {
                // Unit Clause found
                int var = abs(unassigned_lit);
                int val = (unassigned_lit > 0) ? 1 : -1;
                
                // If previously assigned to opposite (should be caught by unassigned_count==0 check mostly)
                if (assignment[var] != 0 && assignment[var] != val) return false;

                if (assignment[var] == 0) {
                    assignment[var] = val;
                    trail.push_back(var);
                    changed = true;
                }
            }
        }
    }
    return true;
}

// Recursive Branch and Bound
void solve_recursive(long long current_cost) {
    nodes_explored++;
    
    // Check timeout every 10k nodes to avoid overhead
    if (nodes_explored % 10000 == 0) {
        if (time_limit > 0) {
            auto now = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = now - start_time;
            if (elapsed.count() > time_limit) {
                cout << "Timeout reached (" << time_limit << "s)." << endl;
                cout << "Time: " << elapsed.count() << "s" << endl;
                cout << "Optimal Cost: " << best_cost << endl;
                
                // Print best found solution if valid
                if (best_assignment[1] != 0) { // Simple check if we found at least one full assignment
                    cout << "v ";
                    for (int i = 1; i <= num_vars; ++i) {
                        if (best_assignment[i] == 1) cout << i << " ";
                        else cout << -i << " ";
                    }
                    cout << "0" << endl;
                } else {
                    cout << "No feasible solution found before timeout." << endl;
                }
                exit(0);
            }
        }
    }

    // 1. Pruning
    if (current_cost >= best_cost) return;

    // 2. Propagation
    vector<int> trail; // Store vars assigned at this level to backtrack later
    if (!unit_propagate(trail)) {
        // Conflict in hard clauses
        // Backtrack local changes
        for (int var : trail) assignment[var] = 0;
        return;
    }

    // 3. Estimate Lower Bound (Lookahead)
    // Calculate cost of soft clauses that are currently FALSE
    long long lb = 0;
    for (const auto& sc : soft_clauses) {
        bool possible = false;
        bool satisfied = false;
        for (int lit : sc.lits) {
            int status = eval_lit(lit);
            if (status == 1) { satisfied = true; break; }
            if (status == 0) { possible = true; } // Can still be satisfied
        }
        
        if (!satisfied && !possible) {
            lb += sc.weight;
        }
    }

    // Add current incurred cost (passed in arg)? 
    // Actually, 'current_cost' is usually accumulated. 
    // In this simple recursive structure, let's recalculate total cost from scratch or incremental.
    // Let's use 'lb' as the total cost of definitely unsatisfied clauses.
    // (If we passed current_cost incrementally, we wouldn't need to loop all soft clauses every time, 
    // but iteration is safer for correctness in a simple implementation).
    
    if (lb >= best_cost) {
        // Backtrack
        for (int var : trail) assignment[var] = 0;
        return;
    }

    // 4. Branching
    // Pick unassigned variable
    int branch_var = -1;
    // Heuristic: MOM or VSIDS is better, but simple ordered is okay for FPT small cases
    for (int i = 1; i <= num_vars; ++i) {
        if (assignment[i] == 0) {
            branch_var = i;
            break;
        }
    }

    if (branch_var == -1) {
        // Leaf Node - Solution Found
        best_cost = lb;
        best_assignment = assignment;
        cout << "New Best Cost: " << best_cost << endl;
        // Backtrack
        for (int var : trail) assignment[var] = 0;
        return;
    }

    // Branch True
    assignment[branch_var] = 1;
    solve_recursive(lb); // Cost calculation is done inside next call

    // Quick check if we found optimal (0)
    if (best_cost == 0) {
        for (int var : trail) assignment[var] = 0; // cleanup
        assignment[branch_var] = 0; // cleanup branching
        return; 
    }

    // Branch False
    assignment[branch_var] = -1;
    solve_recursive(lb);

    // Backtrack local assignments
    assignment[branch_var] = 0;
    for (int var : trail) assignment[var] = 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./wcnf_bnb <file.wcnf> [--timeout <sec>]" << endl;
        return 1;
    }

    string filename;
    
    // Simple arg parsing
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--timeout") {
            if (i + 1 < argc) {
                time_limit = stod(argv[++i]);
            }
        } else {
            filename = arg;
        }
    }

    if (filename.empty()) {
        cout << "Error: No input file specified." << endl;
        return 1;
    }

    parse_wcnf(filename);

    // Calculate initial upper bound (sum of all soft weights)
    long long total_soft_weight = 0;
    for(const auto& sc : soft_clauses) total_soft_weight += sc.weight;
    best_cost = total_soft_weight + 1;

    // Pre-sort soft clauses by weight descending?
    sort(soft_clauses.begin(), soft_clauses.end(), [](const SoftClause& a, const SoftClause& b) {
        return a.weight > b.weight;
    });

    cout << "Starting C++ Branch and Bound..." << endl;
    cout << "Vars: " << num_vars << ", Hard: " << hard_clauses.size() << ", Soft: " << soft_clauses.size() << endl;
    if (time_limit > 0) cout << "Timeout: " << time_limit << "s" << endl;

    solve_recursive(0);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Search Finished." << endl;
    cout << "Time: " << elapsed.count() << "s" << endl;
    cout << "Optimal Cost: " << best_cost << endl;

    // Print assignment in DIMACS format
    cout << "v ";
    for (int i = 1; i <= num_vars; ++i) {
        if (best_assignment[i] == 1) cout << i << " ";
        else cout << -i << " ";
    }
    cout << "0" << endl;

    return 0;
}