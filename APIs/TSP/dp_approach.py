#THIS IS THE DP APPROACH TO THE TSP PROBLEM

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

def dp_approach(adj,n):

    memo=[[-1]*(1<<n) for x in range(n)]

    def totalcost(mask,curr):
        if mask==(1<<n)-1:
            return adj[curr][0]

        if memo[curr][mask] != -1:
            return memo[curr][mask]

        ans=float('inf')

        for i in range(n):
            if((mask&(1<<i))==0):
                ans=min(ans,adj[curr][i]+totalcost(mask|(1<<i),i))

        memo[curr][mask]=ans
        return ans

    return totalcost(1,0)


adj, n=get_graph()
print("The dp approach to the TSP problem yields: ")
print(dp_approach(adj,n))


