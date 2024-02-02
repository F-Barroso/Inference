import networkx as nx
import numpy as np
import pandas as pd
import itertools as it

from castle.common import independence_tests

def condition(node_list, arr ,m):
    H = nx.DiGraph()
    H.add_nodes_from(node_list)
    H.add_edges_from(np.unique(arr[m:],axis=1))
    return nx.is_connected(H.to_undirected())

def binary_search(node_list, arr):
    '''arr - sorted array'''
    L=0
    R=len(arr)-1
    while L<=R:
        if L==R:
            return L
        
        m = int((L+R)/2)+1
        if not condition(node_list, arr, m): #is it not connected?
            R = m-1
        else:
            L = m
    return "fail"

def triangulation(data, node_list, edge_list, thres, states):
    key = {node_list[i]:i for i in range(len(node_list))}

    DAG_w2 = nx.DiGraph()
    DAG_w2.add_nodes_from(node_list)
    DAG_w2.add_edges_from(edge_list)
    
    for b in np.array(DAG_w2.nodes)[np.array(DAG_w2.in_degree)[:,1].astype("int")>1]:
        parents = np.array(list(DAG_w2.in_edges(b)))[:,0]
        
        for a in parents: #a-> b <-u (test a->b; b=node)
            if (a,b) in DAG_w2.edges: #was it already removed?
                
                #Note: it.permutations(parents,2) can't be used in conjuction with the last break since that will cause the cycle to skip triangles for different a's.
                for c in parents[parents!=a]:
                    #print(b,a,parents[parents!=a])
                    survives=False
                    
                    for st_varB,st_varA,st_varC in it.product(*[states[b],states[a],states[c]]):
                        PC = (Y[key[c]]==st_varC).sum()
                        PAC= ((Y[key[a]]==st_varA)&(Y[key[c]]==st_varC)).sum()
                        PBC= ((Y[key[b]]==st_varB)&(Y[key[c]]==st_varC)).sum()
                        PABC= ((Y[key[b]]==st_varB)&(Y[key[a]]==st_varA)&(Y[key[c]]==st_varC)).sum()
                        if PAC!=0 and PAC!=PC: #conditioned to c
                            if np.abs(PABC*PC-PAC*PBC)/(PAC*(PC-PAC)) > thres:
                                survives=True
                                break
                    if not survives:
                        DAG_w2.remove_edge(a,b)
                        break                        
    return DAG_w2

def triangulation_fisher(data, node_list, edge_list, thres):
    '''Triangulation method tailored for fisher metric'''
    key = {node_list[i]:i for i in range(len(node_list))}
    DAG_w2 = nx.DiGraph()
    DAG_w2.add_nodes_from(node_list)
    DAG_w2.add_edges_from(edge_list)
    
    for b in np.array(DAG_w2.nodes)[np.array(DAG_w2.in_degree)[:,1].astype("int")>1]:
        parents = np.array(list(DAG_w2.in_edges(b)))[:,0]
        
        for a in parents: #a-> b <-u (test a->b; b=node)
            if (a,b) in DAG_w2.in_edges: #was it already removed?
                
                #Note: it.permutations(parents,2) can't be used in conjuction with the last break since that will cause the cycle to skip triangles for different a's.
                for c in parents[parents!=a]:
                    
                    if independence_tests.CITest.fisherz_test(data,key[a],key[b],[key[c]])[2] > thres: #does not survive
                        DAG_w2.remove_edge(a,b)
                        break
                    
    return DAG_w2
