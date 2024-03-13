import networkx as nx
import numpy as np
import itertools as it
import knee.rdp as rdp
import knee.kneedle as kneedle

import time

from controlled_zeros import *
from data_generation import *
from thresholding import *
from weight_computer import *

from castle.common import GraphDAG, independence_tests
from castle.metrics import MetricsDAG
from castle.algorithms import PC

n=10000
for n_nodes in [20,40,60,80,100,120,200]:
    
    for i in range(10):
        data = np.zeros([1,46])
    	
        density = 3 #mean degree
        s = 2*density/(n_nodes-1) #sparseness
        A = rd.binomial(1,s,size=(n_nodes,n_nodes)) #Adjency matrix
        for k,j in it.product(range(n_nodes),repeat=2): #removes bottom half of matrix
            if k>=j: A[k,j]=0
        DAGt = nx.convert_matrix.from_numpy_array(A,create_using=nx.DiGraph)
        DAGt = nx.relabel_nodes(DAGt,{node:str(node) for node in DAGt.nodes})
        Gt = nx.to_undirected(DAGt)
        
        states = stater(DAGt, min_states=2, max_states=4)
        X = generator(DAGt, states, n)
        #order = {node:int(node) for node in DAGt.nodes}
        
        data[0,0] = n_nodes
        data[0,1] = np.mean((np.array(DAGt.in_degree)[:,1]).astype("int"))
        
        print(i)
            
        true_matrix=nx.adjacency_matrix(DAGt,nodelist=list(states)).toarray()
    
        #PC Algorithm
        ti = time.process_time()
        pc = PC(alpha=0.05)
        pc.learn(X)
        data[0,2] = time.process_time() - ti #time in seconds

        FN = int(np.sum((true_matrix-pc.causal_matrix)>0)) #False Negatives
        FP = int(np.sum((true_matrix-pc.causal_matrix)<0)) #False Positives
        TP = len(DAGt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*n_nodes - len(DAGt.edges)) - FP #True Negatives = N - FP
        data[0,3] = FP/(n_nodes*n_nodes - len(DAGt.edges)) #FPR = FP/N
        data[0,4] = FN/len(DAGt.edges) #FNR = FN/P
        data[0,5] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
    
        #Connected with Fisher
        ti = time.process_time()
        fish_vals = [independence_tests.CITest.fisherz_test(X,x,y,[])[2] for x,y in it.permutations(range(len(X[0])),2)]
        fish_vars = [(x,y) for x,y in it.permutations(list(states),2)]
    
        unique_edges, unique_vals = (lambda x: (np.array(fish_vars)[x],np.array(fish_vals)[x]))(np.argsort(fish_vals))
        unique_edges, unique_vals = np.flip(unique_edges,axis=0), np.flip(unique_vals,axis=0)
    
        ##Threshold in first step
        m = binary_search(list(states), unique_edges)
        thres = unique_vals[m]
        data[0,6] = m
        data[0,7] = thres
        data[0,8] = time.process_time() - ti #time in seconds
        
        ti = time.process_time()
    	##Second Step
        DAG_w2 = triangulation_fisher(X, list(DAGt.nodes), unique_edges[m:], thres)
        data[0,30] = time.process_time() - ti #time in seconds
        
        FN = len(DAGt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-DAGt.edges) #False Positives
        TP = len(DAGt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*n_nodes - len(DAGt.edges)) - FP #True Negatives = N - FP
        data[0,9] = FP/(n_nodes*n_nodes - len(DAGt.edges)) #FPR = FP/N
        data[0,10] = FN/len(DAGt.edges) #FNR = FN/P
        data[0,11] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC

        #Skeleton?
        G_w2 = nx.to_undirected(DAG_w2)
        FN = len(Gt.edges-G_w2.edges) #False Negatives
        FP = len(G_w2.edges-Gt.edges) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*(n_nodes-1)/2 - len(Gt.edges)) - FP #True Negatives = N - FP
        data[0,34] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,35] = FN/len(Gt.edges) #FNR = FN/P
        data[0,36] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
        
        #Knee with Fisher
        ti = time.process_time()
        fish_vals = [independence_tests.CITest.fisherz_test(X,x,y,[])[2] for x,y in it.permutations(range(len(X[0])),2)]
        fish_vars = [(x,y) for x,y in it.permutations(list(states),2)]

        unique_edges, unique_vals = (lambda x: (np.array(fish_vars)[x], np.array(fish_vals)[x]))(np.argsort(fish_vals))
        unique_edges, unique_vals = np.flip(unique_edges,axis=0), np.flip(unique_vals,axis=0)
    
        ##Threshold in first step
        gcc_nodes=np.zeros(len(unique_edges))
        for j in range(len(unique_edges)):
            H = nx.Graph()
            H.add_edges_from(unique_edges[j:])
            gcc_nodes[j] = len(max(nx.connected_components(H),key=len))
        del H
        m = kneedle.auto_knee(np.column_stack((np.arange(len(gcc_nodes)),gcc_nodes)))
        thres = unique_vals[m]
        data[0,12] = m
        data[0,13] = thres
        data[0,14] = time.process_time() - ti #time in seconds
    
        ti = time.process_time()
    	##Second Step
        DAG_w2 = triangulation_fisher(X, list(DAGt.nodes), unique_edges[m:], thres)
        data[0,31] = time.process_time() - ti #time in seconds
        
        FN = len(DAGt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-DAGt.edges) #False Positives
        TP = len(DAGt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*n_nodes - len(DAGt.edges)) - FP #True Negatives = N - FP
        data[0,15] = FP/(n_nodes*n_nodes - len(DAGt.edges)) #FPR = FP/N
        data[0,16] = FN/len(DAGt.edges) #FNR = FN/P
        data[0,17] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC

        G_w2 = nx.to_undirected(DAG_w2)
        FN = len(Gt.edges-G_w2.edges) #False Negatives
        FP = len(G_w2.edges-Gt.edges) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*(n_nodes-1)/2 - len(Gt.edges)) - FP #True Negatives = N - FP
        data[0,37] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,38] = FN/len(Gt.edges) #FNR = FN/P
        data[0,39] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
    
        #Connected with NI
        ti = time.process_time()
        weight_num_writer(X, states,filename='weights_num_DN')
        wn_var = np.array(weight_var_importer('weights_num_DN.txt'))
        wn_val = np.array(weight_val_importer('weights_num_DN.txt'))
        
        unique_edges = np.unique(wn_var[:,:2],axis=0)
        unique_vals = np.zeros(len(unique_edges))
        for j in range(len(unique_edges)):
            pair = unique_edges[j]
            unique_vals[j] = (np.max( np.abs( wn_val[np.all(wn_var[:,:2]==pair,axis=1)]) ) )
            
        unique_vals=np.abs(unique_vals)
        unique_edges, unique_vals = (lambda x: (unique_edges[x], unique_vals[x]))(np.argsort(unique_vals))
            
        ##Threshold in first step
        m=binary_search(list(states), unique_edges)
        thres = unique_vals[m]
        data[0,18] = m
        data[0,19] = thres
        data[0,20] = time.process_time() - ti #time in seconds
        
        ti = time.process_time()
        ##Second Step
        DAG_w2 = triangulation(X, list(states), unique_edges[m:], thres, states)
        data[0,32] = time.process_time() - ti #time in seconds
        
        FN = len(DAGt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-DAGt.edges) #False Positives
        TP = len(DAGt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*n_nodes - len(DAGt.edges)) - FP #True Negatives = N - FP
        data[0,21] = FP/(n_nodes*n_nodes - len(DAGt.edges)) #FPR = FP/N
        data[0,22] = FN/len(DAGt.edges) #FNR = FN/P
        data[0,23] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC

        G_w2 = nx.to_undirected(DAG_w2)
        FN = len(Gt.edges-G_w2.edges) #False Negatives
        FP = len(G_w2.edges-Gt.edges) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*(n_nodes-1)/2 - len(Gt.edges)) - FP #True Negatives = N - FP
        data[0,40] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,41] = FN/len(Gt.edges) #FNR = FN/P
        data[0,42] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
    
        #Knee with NI
        ti = time.process_time()
        weight_num_writer(X, states,filename='weights_num_DN')
        wn_var = np.array(weight_var_importer('weights_num_DN.txt'))
        wn_val = np.array(weight_val_importer('weights_num_DN.txt'))
            
        unique_edges = np.unique(wn_var[:,:2],axis=0)
        unique_vals = np.zeros(len(unique_edges))
        for j in range(len(unique_edges)):
            pair = unique_edges[j]
            unique_vals[j] = (np.max( np.abs( wn_val[np.all(wn_var[:,:2]==pair,axis=1)]) ) )
            
        unique_vals=np.abs(unique_vals)
        unique_edges, unique_vals = (lambda x: (unique_edges[x], unique_vals[x]))(np.argsort(unique_vals))
        
        ##Threshold in first step
        gcc_nodes=np.zeros(len(unique_edges))
        for j in range(len(unique_edges)):
            H = nx.Graph()
            H.add_edges_from(unique_edges[j:])
            gcc_nodes[j] = len(max(nx.connected_components(H),key=len))
        del H
        m = kneedle.auto_knee(np.column_stack((np.arange(len(gcc_nodes)),gcc_nodes)))
        thres = unique_vals[m]
        data[0,24] = m
        data[0,25] = thres
        data[0,26] = time.process_time() - ti #time in seconds
        
        ti = time.process_time()
    	##Second Step
        DAG_w2 = triangulation(X, list(states), unique_edges[m:], thres, states)
        data[0,33] = time.process_time() - ti #time in seconds

        FN = len(DAGt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-DAGt.edges) #False Positives
        TP = len(DAGt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*n_nodes - len(DAGt.edges)) - FP #True Negatives = N - FP
        data[0,27] = FP/(n_nodes*n_nodes - len(DAGt.edges)) #FPR = FP/N
        data[0,28] = FN/len(DAGt.edges) #FNR = FN/P
        data[0,29] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC

        G_w2 = nx.to_undirected(DAG_w2)
        FN = len(Gt.edges-G_w2.edges) #False Negatives
        FP = len(G_w2.edges-Gt.edges) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*(n_nodes-1)/2 - len(Gt.edges)) - FP #True Negatives = N - FP
        data[0,43] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,44] = FN/len(Gt.edges) #FNR = FN/P
        data[0,45] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
        
        f = open("synthmeasuresDense_NOrder.txt", "a+")
        np.savetxt(f,data)
        f.close()
        del data
        
    print(n_nodes)
