import networkx as nx
import numpy as np
rd = np.random
import itertools as it
import pandas as pd

def stater(G, min_states=2, max_states=4):
    '''Returns a dictionary of random states'''
    return {node:list(range(rd.randint(min_states, max_states+1))) for node in G.nodes}

########################################################################

def probabiliter_controlled(G, states):
    '''Returns a dictionary of arrays containing probabilities generated from random parameters for each combination of states and the respective cumulative distributions.'''
   
    probs = {} #dictionary to hold the dataframes
    
    for node in G.nodes:
        parents = list(G.predecessors(node)) #list of predecessor nodes
        d = len(states[node])
        
        # generate betas that sum to 1, representing the probabilities of states i in the absence of node influences
        betas = np.diff(np.sort(rd.random(d-1)),prepend=0,append=1)
                
        lps = sum(1 for dummy in it.product(*[states[parent] for parent in parents])) #number of unique combinations of parent states
        r = rd.random(lps)**(1/d)
        alphas=r[:, np.newaxis]*np.diff(np.sort(rd.random([lps,d-1]),axis=1),axis=1,prepend=0,append=1) #r^T*M
        
        problist = np.array([[betas[i] * (1-r[u]) for i in range(d)] for u in range(lps)])+alphas # P(i|u)
        cumlist=np.cumsum(problist,axis=1)

        cumlist_ = np.zeros(lps*d)
        
        #Loops over the number of products over parent states; if zero it has a single cycle with where selecting all lines (thus all node_states)
        for i in range(lps):
            where = [i + lps*j for j in range(d)] # lines in cumlist_ (thus different node_states) with the same parent_states
            cumlist_[where] = cumlist[i]

        probs[node] = cumlist_
                
    return probs
    
########################################################################

def generator(G, states, states_prob, n):
    '''Generates n lines of data consistent with a given DAG, node states and their probabilities.'''
    data = {} #initialize a dictionary
    
    #These two loops are computationally equivalent to a single loop over topologically ordered nodes
    for gen in nx.topological_generations(G): #loop over DAG generations
        for node in gen: #loop over the nodes in the generation
            
            parents = list(G.predecessors(node))
            probs = states_prob[node] #array with the conditional probabilities to a node
        
            rng = rd.random(n) #generate n random numbers, used to determine states according probability distributions
            st_counter=np.zeros(n, dtype=int) #array that shall contain the n realizations of the node
            
            i=0 #counter of system states/lines of probs
            #Loops over the combinations
            for sys_state in it.product(states[node],*[states[parent] for parent in parents]):
                
                truths = np.ones(n,dtype=bool) #Initialize array of Trues:
                                
                #each position of the array shall indicate after the loop if the realizations already generated for the parent states
                #are compatible with the parent_states in line i of probs. If the node is orphan, the array will remain unchanged.
                for j in range(len(parents)):
                    truths &= (data[parents[j]] == sys_state[1+j]) #find which realizations are not compatible with parent_states
                        
                #Update the st_counter in the positions that are compatible with the already generated parent states.
                #Note: at the end of current loop, one and only one value will be assigned to each position of st_counter
                st_counter[truths] += probs[i] < rng[truths]
                i+=1

            data[node]=st_counter #add an entry to the dictionary with the realizations for the node
            
    return pd.DataFrame(data)
