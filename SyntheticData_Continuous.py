import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import itertools as it
import knee.rdp as rdp
import knee.kneedle as kneedle
from scipy.special import comb

import time

from controlled_zeros import *
#from data_generation import *
from weight_computer import *

from castle.common import GraphDAG, independence_tests
from castle.metrics import MetricsDAG
from castle.algorithms import PC

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

def triangulation2(data, node_list, edge_list, thres):
    key = {node_list[i]:i for i in range(len(node_list))}
    DAG_w2 = nx.DiGraph()
    DAG_w2.add_nodes_from(node_list)
    DAG_w2.add_edges_from(edge_list)
    
    for b in np.array(DAG_w2.nodes)[np.array(DAG_w2.in_degree)[:,1].astype("int")>1]:
        parents = np.array(list(DAG_w2.in_edges(b)))[:,0]
        
        for a in parents: #a-> b <-u (test a->b; b=node)
            if (a,b) not in DAG_w2.edges: #was it already removed?
                continue
            #Note: it.permutations(parents,2) can't be used in conjuction with the last break since that will cause the cycle to skip triangles for different a's.
            for c in parents[parents!=a]:
                #print(b,a,parents[parents!=a])
                if independence_tests.CITest.fisherz_test(data,key[a],key[b],[key[c]])[2] > thres: #does not survive
                    DAG_w2.remove_edge(a,b)
                    break
                    
    return DAG_w2

n=10000
for n_nodes in [20,80,100,120]:

    data=np.zeros([20,30])
    
    for i in range(20):
        density = 1 #mean degree
        s = 2*density/(n_nodes-1) #sparseness
        A = rd.binomial(1,s,size=(n_nodes,n_nodes)) #Adjency matrix
        for k,j in it.product(range(n_nodes),repeat=2):
            if k>=j: A[k,j]=0
        DAGt = nx.convert_matrix.from_numpy_array(A,create_using=nx.DiGraph)
        DAGt = nx.relabel_nodes(DAGt,{node:node for node in DAGt.nodes})

        X = np.zeros([n,n_nodes])
        X[:,0] = rd.normal(size=n) #noise
        for j in range(1,n_nodes):
            X[:,j] = np.sum(A[:j,j]*X[:,:j],axis=1) + rd.normal(size=n) #connections + noise
        
        data[i,0] = n_nodes
        data[i,1] = np.mean((np.array(DAGt.in_degree)[:,1]).astype("int"))
        
        print(i)
        
        #PC Algorithm
        ti = time.process_time_ns()
        pc = PC(alpha=0.05)
        pc.learn(X)
        data[i,2] = (time.process_time_ns() - ti)*1e-9 #time in seconds
        FN = int(np.sum((A-pc.causal_matrix)>0)) #False Negatives
        FP = int(np.sum((A-pc.causal_matrix)<0)) #False Positives
        TP = len(DAGt.edges) - FN #True Positives = P - FN
        TN = (comb(n_nodes,2).astype(int) - len(DAGt.edges)) - FP #True Negatives = N - FP
        data[i,3] = FP/(comb(n_nodes,2).astype(int) - len(DAGt.edges)) #FPR = FP/N
        data[i,4] = FN/len(DAGt.edges) #FNR = FN/P
        data[i,5] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
    
        #Connected with Fisher
        ti = time.process_time_ns()
        fish_vals = [independence_tests.CITest.fisherz_test(X,x,y,[])[2] for x,y in it.permutations(range(len(X[0])),2)]
        fish_vars = [(x,y) for x,y in it.permutations(list(range(n_nodes)),2)]
    
        unique_edges, unique_vals = np.array(fish_vars)[np.argsort(fish_vals)], np.sort(fish_vals)
        unique_edges = np.flip(unique_edges)
        unique_vals = np.flip(unique_vals)
    
        ##Threshold in first step
        for j in np.unique(unique_vals,return_index=True)[1]:
            H = nx.Graph()
            H.add_nodes_from(list(range(n_nodes)))
            H.add_edges_from(unique_edges[j:])
            if nx.is_connected(H):
                m = j
                break
        del H
        thres = unique_vals[m]
        data[i,6] = m
        data[i,7] = thres
    
        ##Second Step
        DAG_w2 = triangulation2(X, list(DAGt.nodes), unique_edges[m:], thres)
        
        data[i,8] = (time.process_time_ns() - ti)*1e-9 #time in seconds
        FN = len(DAGt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-DAGt.edges) #False Positives
        TP = len(DAGt.edges) - FN #True Positives = P - FN
        TN = (comb(n_nodes,2).astype(int) - len(DAGt.edges)) - FP #True Negatives = N - FP
        data[i,9] = FP/(comb(n_nodes,2).astype(int) - len(DAGt.edges)) #FPR = FP/N
        data[i,10] = FN/len(DAGt.edges) #FNR = FN/P
        data[i,11] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
        
        #Knee with Fisher
        ti = time.process_time_ns()
        
        fish_vals = [independence_tests.CITest.fisherz_test(X,x,y,[])[2] for x,y in it.permutations(range(len(X[0])),2)]
        fish_vars = [(x,y) for x,y in it.permutations(list(range(n_nodes)),2)]
        
        unique_edges, unique_vals = np.array(fish_vars)[np.argsort(fish_vals)], np.sort(fish_vals)
        unique_edges = np.flip(unique_edges)
        unique_vals = np.flip(unique_vals)
    
        ##Threshold in first step
        gcc_nodes=np.zeros(len(unique_edges))
        for j in range(len(unique_edges)):
            H = nx.Graph()
            H.add_edges_from(unique_edges[j:])
            gcc_nodes[j] = len(sorted(nx.connected_components(H), key=len, reverse=True)[0])
        del H
        m = kneedle.auto_knee(np.column_stack((np.arange(len(gcc_nodes)),gcc_nodes)))
        thres = unique_vals[m]
        data[i,12] = m
        data[i,13] = thres
    
        ##Second Step
        DAG_w2 = triangulation2(X, list(DAGt.nodes), unique_edges[m:], thres)
        
        data[i,14] = (time.process_time_ns() - ti)*1e-9 #time in seconds
        FN = len(DAGt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-DAGt.edges) #False Positives
        TP = len(DAGt.edges) - FN #True Positives = P - FN
        TN = (comb(n_nodes,2).astype(int) - len(DAGt.edges)) - FP #True Negatives = N - FP
        data[i,15] = FP/(comb(n_nodes,2).astype(int) - len(DAGt.edges)) #FPR = FP/N
        data[i,16] = FN/len(DAGt.edges) #FNR = FN/P
        data[i,17] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
    
    f = open("synthmeasuresContinuous_data.txt", "a+")
    np.savetxt(f,data)
    f.close()
    del data
    print(n_nodes)
