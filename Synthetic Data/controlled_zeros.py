import networkx as nx
import numpy as np
rd = np.random
import itertools as it
from scipy.stats import poisson

# CONTROLLED ZEROS 03/06/2022
def controlled_zeros(n, q, z):
    '''Builds a DAG G(n,q,z) - directed acyclic graph -, with n nodes and a Poisson in/out-degree distribution (with average q), where each new node is linked with a predefined probability function, according to an assigned percentage of nodes w/o parents, z.
    n    : number of nodes
    q    : mean degree
    z    : percentage of nodes without parents'''
    
    logic = True # Logical value for cycle
    error = 0  # Number of times that the in-slots sequence assignment fails
    
    lendom=int(10*q)
    inslots_dom = np.arange(0, lendom)  #Domain of in-degree

    inslots_dist = np.zeros(lendom)  # Distribution of in-degree
    qin_alt = q/(1-z)-1 # Corrected q for in-degree
    inslots_dist[1:] = (1-z)*poisson.pmf(inslots_dom[1:]-1, qin_alt)  # Setup of the in-slots distribution
    inslots_dist[0] = z  # Assignment of the probability of happening 0 in-slots (nodes w/o parents), according to the normalization

    outslots_dom = np.arange(1, lendom)  # Domain of out-degree
    qout_alt = q-1 # Corrected q for out-degree
    outslots_dist = poisson.pmf(outslots_dom-1, qout_alt)  # Distribution of out-degree

    while logic:
        G = nx.DiGraph()  # Creation of the graph
        G.add_node(n - 1)  # Addition of the initial node (which represents the final station)
        
        in_slots = rd.choice(inslots_dom, n, p=inslots_dist)  # Assignment of a given number of in-slots to each node, according to the in-slot distribution
        out_slots = rd.choice(outslots_dom, n, p=outslots_dist)  # Assignment of a given number of out-slots to each node, according to the out-slot distribution
        out_slots[-1] = 0  # Update on the out-slots of the final node (which has 0 out-slots)
        in_slots[0] = 0  # Update on the in-slots of the intial node (which has 0 in-slots)
        in_degree = np.zeros(n, dtype=int)  # Initialization of the array that stores the in-degree of each node
        out_degree = np.zeros(n, dtype=int)  # Initialization of the array that stores the in-degree of each node

        P = np.zeros(n)  # Linking probability of each node currently present in the graph
        P[-1] = in_slots[-1]  # Initialization of the linking probabilities, which are a measure of the available slots at each step
        sumP = in_slots[-1]  # Initialization of the sum of linking probabilities (for later normalization)

        logic = False #If unchanged in the following for-cycle, it stops the while-cycle
        for i in range(n - 2, -1, -1):  # For each of the other nodes, iterating backwards (adding new nodes from the final station to the initial station)

            if sumP == 0: # The in-slots sequence is invalid and the algorithm repeats itself to generate another one
                # Example: when in_slots[-1] == 0, the final node is wrong
                logic = True
                error += 1
                #print("error:",error)
                break
            
            G.add_node(i)  # Addition of the new node i
            j_array = rd.choice(np.arange(0,n), out_slots[i], p=P/sumP)  # Choice of node(s) j to which node it will link itself, according to the linking probabilities
            j = np.unique(j_array)  # Removal of repeated node(s) j, to avoid making the same edge more than once (pruning)

            in_degree[j] += 1  # Update on the in-degree of node(s) j (which increases by 1)
            out_degree[i] += len(j)  # Update on the out-degree of node i (which increases by len(j))
            P[j] = in_slots[j] - in_degree[j]  # Update of the linking probabilities for node(s) j (decreases by in_degree[j])
            P[i] = in_slots[i] - in_degree[i]  # Update of the linking probabilities for node(s) i (changes from 0 to in_slots[i])
            sumP += (in_slots[i] - len(j))  # Update on the sum of linking probabilities (node i now has in-slots available for the next added nodes, and node(s) j lose one slot each)

            e = [[i], j]  # Creation the set of edges with the list of the chosen ending nodes j
            G.add_edges_from(list(it.product(*e)))  # Additon of every edge that comes out of the node i
    return G