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
    '''Returns a dictionary of pandas DataFrames containing probabilities generated from random parameters for each combination of states
    and the respective cumulative distributions.'''
   
    probs = {} #dictionary to hold the dataframes
    
    for node in G.nodes:
        parents = list(G.predecessors(node)) #list of predecessor nodes
        d = len(states[node])
        
        # generate betas that sum to 1, representing the probabilities of states i in the absence of node influences
        betas = np.diff(np.sort(rd.random(d-1)),prepend=0,append=1)
                
        nj = sum(1 for dummy in it.product(*[states[parent] for parent in parents])) #number of unique combinations of parent states
        r = rd.random(nj)**(1/d)
        alphas=r[:, np.newaxis]*np.diff(np.sort(rd.random([nj,d-1]),axis=1),axis=1,prepend=0,append=1) #r^T*M
        
        problist = np.array([[betas[i] * (1-r[u]) for i in range(d)] for u in range(nj)])+alphas # P(i|u)
        cumlist=np.cumsum(problist,axis=1)

        problist_, cumlist_ = np.zeros(nj*d), np.zeros(nj*d)
        #Loops over the number of products over parent states; if zero it has a single cycle with where selecting all lines (thus all node_states)
        for i in range(nj):
            where = [i + nj*j for j in range(d)] # lines in df (thus different node_states) with the same parent_states
            problist_[where] = problist[i]
            cumlist_[where] = cumlist[i]
            
        df = pd.DataFrame(it.product(states[node],*[states[parent] for parent in parents]), columns=[node]+parents) #Heavy memory consumption! FIXME!
        df['Probability'] = problist_
        df['Cumulative'] = cumlist_
        probs[node] = df
                
    return probs
                             
########################################################################

def generator(G, states, states_prob, n):
    '''Generates n lines of data consistent with a given DAG, node states and their probabilities.'''
    data = {} #initialize a dictionary
    
    #These two loops are computationally equivalent to a single loop over topologically ordered nodes
    for gen in nx.topological_generations(G): #loop over DAG generations
        for node in gen: #loop over the nodes in the generation
            
            parents = list(G.predecessors(node))
            df = states_prob[node] #dataframe with the conditional probabilities to a node
        
            rng = rd.random(n) #generate n random numbers, used to determine states according probability distributions
            st_counter=np.zeros(n, dtype=int) #array that shall contain the n realizations of the node
            
            lps = sum([1 for cnt in it.product(*[states[parent] for parent in parents])]) #number of product of parent states
            #Loops over the number of lines of df; each maps to an unique system state=node_state+parent_states
            for i in range(len(df)):
                
                truths = np.ones(n,dtype=bool) #Initialize array of Trues:
                #each position of the array shall indicate after the loop if the realizations already generated for the parent states
                #are compatible with the parent_states in line i of df. If the node is orphan, the array will remain unchanged.
                for j in range(len(parents)):
                    truths &= (data[parents[j]] == df.loc[i][parents[j]]) #find which realizations are not compatible with parent_states
                    
                #Update the st_counter in the positions that are compatible with the already generated parent states.
                #Note: at the end of current loop, one and only one value will be assigned to each position of st_counter
                st_counter[truths] += df.loc[i].Cumulative < rng[truths]

            data[node]=st_counter #add an entry to the dictionary with the realizations for the node
            
    return pd.DataFrame(data)
