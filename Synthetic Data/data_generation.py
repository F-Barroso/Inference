import networkx as nx
import numpy as np
rd = np.random
import itertools as it

def stater(G, min_states=2, max_states=4):
    '''Returns a dictionary of random states'''
    return {node:list(range(rd.randint(min_states, max_states+1))) for node in G.nodes}


def generator(G, states, n):
    '''Generates n lines of data consistent with a given DAG, node states and their probabilities.'''
    key = {list(states)[i]:i for i in range(len(states))} #key used to mantain the order of the array

    data = np.zeros([n,len(states)])
    #These two loops are computationally equivalent to a single loop over topologically ordered nodes
    for gen in nx.topological_generations(G): #loop over DAG generations
        for node in gen: #loop over the nodes in the generation
            
            parents = list(G.predecessors(node))
            d = len(states[node])

            #### Generate cumulative probabilities ####
            # generate betas that sum to 1, representing the probabilities of states i in the absence of node influences
            betas = np.diff(np.sort(rd.random(d-1)),prepend=0,append=1)
                    
            lps = sum(1 for dummy in it.product(*[states[parent] for parent in parents])) #number of unique combinations of parent states
            r = rd.random(lps)**(1/d)
            alphas=r[:, np.newaxis]*np.diff(np.sort(rd.random([lps,d-1]),axis=1),axis=1,prepend=0,append=1) #r^T*M
            
            cumlist=np.cumsum( np.array([[betas[i] * (1-r[u]) for i in range(d)] for u in range(lps)])+alphas ,axis=1).flatten('F') #cumulative of P(i|u)
            ###########################################
    
            #### Generate data ####
            rng = rd.random(n) #generate n random numbers, used to determine states according probability distributions
            st_counter=np.zeros(n, dtype=int) #array that shall contain the n realizations of the node
            
            i=0 #counter of system states/lines of cumlist
            #Loops over the combinations
            for sys_state in it.product(states[node],*[states[parent] for parent in parents]):
                
                truths = np.ones(n,dtype=bool) #Initialize array of Trues:
                                
                #each position of the array shall indicate after the loop if the realizations already generated for the parent states
                #are compatible with the parent_states in line i of probs. If the node is orphan, the array will remain unchanged.
                for j in range(len(parents)):
                    truths &= (data[:,key[parents[j]]] == sys_state[1+j]) #find which realizations are not compatible with parent_states
                        
                #Update the st_counter in the positions that are compatible with the already generated parent states.
                #Note: at the end of current loop, one and only one value will be assigned to each position of st_counter
                st_counter[truths] += cumlist[i] < rng[truths]
                i+=1

            data[:,key[node]]=st_counter #add an entry to the dictionary with the realizations for the node
            
    return data
