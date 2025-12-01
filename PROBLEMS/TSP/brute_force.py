#THIS IS THE BRUTE FORCE APPROACH TO THE TSP PROBLEM

import numpy as np
from itertools import permutations

def get_graph():
    numcities=int(input("Enter number of cities (cities will be 0 indexed) : "))
    numroads=int(input("Enter number of roads: "))

    adj=np.zeros((numcities,numcities), dtype=float)

    print("Type in roads in u,v,w format, where u and v are cities and w is the wieght we want to give the bidirectional road between them: ")
    for i in range(numroads):
        input_line=input();
        u_str,v_str,w_str=input_line.split()
        u=int(u_str)
        v=int(v_str)
        w=float(w_str)
        if adj[u,v]==0:
            adj[u,v]=w
            adj[v,u]=w
        else:
            adj[u,v]=np.min(adj[u,v],w)
            adj[v,u]=adj[u,v]

    return adj, numcities


def brute_force(adj, n):
    min_cost=[float('inf')]

    start_node=0

    all_intermediate_paths=permutations(range(1,n))

    for path in all_intermediate_paths:
        current_tour=[start_node]+list(path)+[start_node]
        current_cost=0.0
        isvalid=True

        for i in range(n):
            u=current_tour[i]
            v=current_tour[i+1]

            w=adj[u,v]

            if w==0 and u!=v:
                isvalid=False
                break

            current_cost+=w

        if isvalid:
            min_cost[0]=min(min_cost[0],current_cost)

    
    if min_cost[0]==float('inf'):
        return -1.0

    return min_cost[0]

adj, n=get_graph()
print("Through brute-force approach, the answer to the TSP described here is: ")
print(brute_force(adj,n))


