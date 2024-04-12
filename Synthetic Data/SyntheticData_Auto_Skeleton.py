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
density = {20:4.9, 40:4.1, 60:3.8, 80:3.7, 100:3.6, 120:3.5, 200:3.4, 300:3.3} #chosen to yield a real mean degree close to 3
#density = {20:1.71, 40:1.55, 60:1.49, 80:1.46, 100:1.44, 120:1.42, 200:1.39} #chosen to yield a real mean degree close to 1.3
for n_nodes in [20,40,60,80,100,120,200]:
    
    for turn in range(10):
        data = np.zeros([1,34])
    	
        orphans = 0.01
        DAGt = controlled_zeros(n_nodes, density[n_nodes], orphans)
        DAGt = nx.relabel_nodes(DAGt,{node:str(node) for node in DAGt.nodes})
        Gt = nx.to_undirected(DAGt)
        
        states = stater(DAGt, min_states=2, max_states=4)
        X = generator(DAGt, states, n)
        node_list = list(states)

        data[0,0] = n_nodes
        data[0,1] = np.mean((np.array(Gt.degree)[:,1]).astype("int"))

        print(turn)
            
        true_skeleton=nx.adjacency_matrix(Gt,nodelist=list(states)).toarray()
        for k,j in it.product(range(n_nodes),repeat=2): #removes bottom half of matrix
            if k>=j: true_skeleton[k,j]=0
    
        #PC Algorithm
        ti = time.process_time()
        skeleton = find_skeleton(X, 0.05, "fisherz")[0]
        data[0,2] = time.process_time() - ti #time in seconds
        for k,j in it.product(range(n_nodes),repeat=2): #removes bottom half of matrix
            if k>=j: skeleton[k,j]=0

        FN = int(np.sum((true_skeleton-skeleton)>0)) #False Negatives
        FP = int(np.sum((true_skeleton-skeleton)<0)) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*(n_nodes-1)/2 - len(Gt.edges))-FP #True Negatives = N-FP
        data[0,3] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,4] = FN/len(Gt.edges) #FNR = FN/P
        data[0,5] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
    
        #Connected with Fisher
        ti = time.process_time()
        key = {node_list[i]:i for i in range(len(node_list))}
        fish_vals = [independence_tests.CITest.fisherz_test(X,key[x],key[y],[])[2] for x,y in it.combinations(node_list,2)]
        fish_vars = [(x,y) for x,y in it.combinations(node_list,2)]
    
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
        DAG_w2 = triangulation_fisher_und(X, node_list, unique_edges[m:], thres)
        data[0,30] = time.process_time() - ti #time in seconds
        
        FN = len(Gt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-Gt.edges) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (int(n_nodes*(n_nodes-1)/2) - len(Gt.edges)) - FP #True Negatives = N - FP
        data[0,9] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,10] = FN/len(Gt.edges) #FNR = FN/P
        data[0,11] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
        
        #Knee with Fisher
        ti = time.process_time()
        key = {node_list[i]:i for i in range(len(node_list))}
        fish_vals = [independence_tests.CITest.fisherz_test(X,key[x],key[y],[])[2] for x,y in it.combinations(node_list,2)]
        fish_vars = [(x,y) for x,y in it.combinations(node_list,2)]

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
        DAG_w2 = triangulation_fisher_und(X, node_list, unique_edges[m:], thres)
        data[0,31] = time.process_time() - ti #time in seconds
        
        FN = len(Gt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-Gt.edges) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (int(n_nodes*(n_nodes-1)/2) - len(Gt.edges)) - FP #True Negatives = N - FP
        data[0,15] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,16] = FN/len(Gt.edges) #FNR = FN/P
        data[0,17] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
    
        #Connected with NI
        ti = time.process_time()
        weight_num_writer(X, states,filename='weights_num_AS')
        wn_var = np.array(weight_var_importer('weights_num_AS.txt'))
        wn_val = np.array(weight_val_importer('weights_num_AS.txt'))
        
        unique_edges = np.unique(wn_var[:,:2],axis=0)
        unique_vals = np.zeros(len(unique_edges))
        for j in range(len(unique_edges)): #for a given direction, keep the biggest state
            pair = unique_edges[j]
            unique_vals[j] = (np.max( np.abs( wn_val[np.all(wn_var[:,:2]==pair,axis=1)]) ) )
            
        unique_vals=np.abs(unique_vals)
        for j in range(len(unique_edges)): #keep the biggest direction
            pair = unique_edges[j]
            anti_j = np.where(np.all(unique_edges==np.flip(pair),axis=1))[0][0]
        
            if unique_vals[j] >= unique_vals[anti_j]:
                
                unique_vals[anti_j] = -np.infty
        unique_vals, unique_edges = (lambda x: (np.delete(unique_vals,x), np.delete(unique_edges,x,axis=0)))(np.where(unique_vals<0))
        unique_edges, unique_vals = (lambda x: (unique_edges[x], unique_vals[x]))(np.argsort(unique_vals))
            
        ##Threshold in first step
        m=binary_search(list(states), unique_edges)
        thres = unique_vals[m]
        data[0,18] = m
        data[0,19] = thres
        data[0,20] = time.process_time() - ti #time in seconds
        
        ti = time.process_time()
        ##Second Step
        DAG_w2 = triangulation_und(X, node_list, unique_edges[m:], thres, states)
        data[0,32] = time.process_time() - ti #time in seconds
        
        FN = len(Gt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-Gt.edges) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*(n_nodes-1)/2 - len(Gt.edges)) - FP #True Negatives = N - FP
        data[0,21] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,22] = FN/len(Gt.edges) #FNR = FN/P
        data[0,23] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
   
        #Knee with NI
        ti = time.process_time()
        weight_num_writer(X, states,filename='weights_num_AS')
        wn_var = np.array(weight_var_importer('weights_num_AS.txt'))
        wn_val = np.array(weight_val_importer('weights_num_AS.txt'))
            
        unique_edges = np.unique(wn_var[:,:2],axis=0)
        unique_vals = np.zeros(len(unique_edges))
        for j in range(len(unique_edges)): #for a given direction, keep the biggest state
            pair = unique_edges[j]
            unique_vals[j] = (np.max( np.abs( wn_val[np.all(wn_var[:,:2]==pair,axis=1)]) ) )
            
        unique_vals=np.abs(unique_vals)
        for j in range(len(unique_edges)): #keep the biggest direction
            pair = unique_edges[j]
            anti_j = np.where(np.all(unique_edges==np.flip(pair),axis=1))[0][0]
        
            if unique_vals[j] >= unique_vals[anti_j]:
                
                unique_vals[anti_j] = -np.infty
        unique_vals, unique_edges = (lambda x: (np.delete(unique_vals,x), np.delete(unique_edges,x,axis=0)))(np.where(unique_vals<0))
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
        DAG_w2 = triangulation_und(X, node_list, unique_edges[m:], thres, states)
        data[0,33] = time.process_time() - ti #time in seconds

        FN = len(Gt.edges-DAG_w2.edges) #False Negatives
        FP = len(DAG_w2.edges-Gt.edges) #False Positives
        TP = len(Gt.edges) - FN #True Positives = P - FN
        TN = (n_nodes*(n_nodes-1)/2 - len(Gt.edges)) - FP #True Negatives = N - FP
        data[0,27] = FP/(n_nodes*(n_nodes-1)/2 - len(Gt.edges)) #FPR = FP/N
        data[0,28] = FN/len(Gt.edges) #FNR = FN/P
        data[0,29] = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) #MCC
        
        f = open("synthmeasuresAuto_Skeleton_1p3.txt", "a+")
        np.savetxt(f,data)
        f.close()
        del data
        
    print(n_nodes)
