#THIS IS THE DP APPROACH TO THE TSP PROBLEM

import numpy as np
from itertools import permutations

data_type=[('u','i4'),('v','i4'),('w','f8')]
def get_graph():
    numcities=int(input("Enter number of cities (cities will be 0 indexed) : "))
    numroads=int(input("Enter number of roads: "))

    adj=np.zeros((numcities,numcities), dtype=float)
    edges=np.array([],dtype=data_type)

    print("Type in roads in u,v,w format, where u and v are cities and w is the wieght we want to give the bidirectional road between them: ")
    for i in range(numroads):
        input_line=input();
        u_str,v_str,w_str=input_line.split()
        u=int(u_str)
        v=int(v_str)
        w=float(w_str)

        new_entry=[(u,v,w)]
        entry_array=np.array(new_entry,dtype=data_type)
        if adj[u,v]==0:
            adj[u,v]=w
            adj[v,u]=w
        else:
            adj[u,v]=np.min(adj[u,v],w)
            adj[v,u]=adj[u,v]

        edges=np.append(edges,entry_array)

    return adj, edges, numcities, numroads

def mstapprox(adj, edges, n, m):
    def kruskal():
        tree=np.zeros((n,n),dtype=float)
        comps=np.ones(n,dtype=int)
        p=np.arange(n,dtype=int)

        def getparent(node):
            if node==p[node]:
                return node
            else:
                p[node]=getparent(p[node])
                return p[node]

        def join(u, v):
            pu=getparent(u)
            pv=getparent(v)
            if pu==pv:
                return 0
            if comps[pv]>comps[pu]:
                comps[pv]+=comps[pu]
                p[pu]=pv
            else:
                comps[pu]+=comps[pv]
                p[pv]=pu

            return 1

        sorted=np.sort(edges,order='w')

        for i in range(m):
            u=sorted[i]['u']
            v=sorted[i]['v']
            w=sorted[i]['w']
            if join(u,v):
                tree[u,v]=w
                tree[v,u]=w

        treeroot=getparent(0)

        return tree,treeroot

    tree,root=kruskal()
    tr=np.array([],dtype=int)
    vis=np.zeros(n,dtype=int)
    def preorder(root,tr):
        tr=np.append(tr,root)
        vis[root]=1
        for i in range(n):
            if tree[root,i]!=0 and vis[i]==0:
                tr = preorder(i,tr)
        return tr

    tr=preorder(root,tr)
    tr=np.append(tr,root)

    print(tr)

    curcost=float(0)

    for i in range(n):
        u=tr[i]
        v=tr[i+1]
        curcost+=adj[u,v]

    return curcost

adj, edges, n, m=get_graph()
print("The 2-approximation MST approach to the TSP problem yields: ")
print(mstapprox(adj,edges,n,m))


