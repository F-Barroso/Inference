import networkx as nx
import numpy as np
import itertools as it

from castle.common import independence_tests

def condition(node_list, arr ,m):
    H = nx.Graph()
    H.add_nodes_from(node_list)
    H.add_edges_from(arr[m:])
    return nx.is_connected(H)

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

    DAG = nx.DiGraph()
    DAG.add_nodes_from(node_list)
    DAG.add_edges_from(edge_list)
    
    for b in np.array(DAG.nodes)[np.array(DAG.in_degree)[:,1].astype("int")>1]:
        parents = np.array(list(DAG.in_edges(b)))[:,0]
        
        for a in parents: #a-> b <-u (test a->b; b=node)
            if (a,b) in DAG.edges: #was it already removed?
                
                #Note: it.permutations(parents,2) can't be used in conjuction with the last break since that will cause the cycle to skip triangles for different a's.
                for c in parents[parents!=a]:
                    #print(b,a,parents[parents!=a])
                    survives=False
                    
                    for st_varB,st_varA,st_varC in it.product(*[states[b],states[a],states[c]]):
                        PC = (data[:,key[c]]==st_varC).sum()
                        PAC= ((data[:,key[a]]==st_varA)&(data[:,key[c]]==st_varC)).sum()
                        PBC= ((data[:,key[b]]==st_varB)&(data[:,key[c]]==st_varC)).sum()
                        PABC= ((data[:,key[b]]==st_varB)&(data[:,key[a]]==st_varA)&(data[:,key[c]]==st_varC)).sum()
                        if PAC!=0 and PAC!=PC: #conditioned to c
                            if np.abs(PABC*PC-PAC*PBC)/(PAC*(PC-PAC)) > thres:
                                survives=True
                                break
                    if not survives:
                        DAG.remove_edge(a,b)
                        break                        
    return DAG

def triangulation_und(data, node_list, edge_list, thres, states):
    key = {node_list[i]:i for i in range(len(node_list))}

    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    #for b in np.array(G.nodes)[np.array(G.degree)[:,1].astype("int")>1]: #might be problematic 
    for b in G.nodes:
        if G.degree(b)==0: continue
        parents = np.array(list(G.edges(b)))[:,1]
        
        for a in parents: #a-- b --u (test a--b; b=node)
            if (a,b) in G.edges: #was it already removed?
                
                #Note: it.permutations(parents,2) can't be used in conjuction with the last break since that will cause the cycle to skip triangles for different a's.
                for c in parents[parents!=a]:
                    #print(b,a,parents[parents!=a])
                    survives=False
                    
                    for st_varB,st_varA,st_varC in it.product(*[states[b],states[a],states[c]]):
                        PC = (data[:,key[c]]==st_varC).sum()
                        PAC= ((data[:,key[a]]==st_varA)&(data[:,key[c]]==st_varC)).sum()
                        PBC= ((data[:,key[b]]==st_varB)&(data[:,key[c]]==st_varC)).sum()
                        PABC= ((data[:,key[b]]==st_varB)&(data[:,key[a]]==st_varA)&(data[:,key[c]]==st_varC)).sum()
                        if (PAC!=0) and (PAC!=PC) and (np.abs(PABC*PC-PAC*PBC)/(PAC*(PC-PAC)) > thres): #conditioned to c a->b
                            survives=True
                            break
                        if (PBC!=0) and (PBC!=PC) and (np.abs(PABC*PC-PAC*PBC)/(PBC*(PC-PBC)) > thres): #conditioned to c b->a
                            survives=True
                            break
                    if not survives: #removes only if all states in both directions fail the threshold
                        G.remove_edge(a,b)
                        break                        
    return G
    
def triangulation_metric(data, node_list, edge_list, thres, metric = "fisherz", pval=False):
    '''Triangulation method tailored for metric'''

    if metric == "fisherz":
        func = independence_tests.CITest.fisherz_test
    if metric == "g2":
        func = independence_tests.CITest.g2_test
    
    key = {node_list[i]:i for i in range(len(node_list))}
    DAG = nx.DiGraph()
    DAG.add_nodes_from(node_list)
    DAG.add_edges_from(edge_list)
    
    for b in np.array(DAG.nodes)[np.array(DAG.in_degree)[:,1].astype("int")>1]:
        parents = np.array(list(DAG.in_edges(b)))[:,0]
        
        for a in parents: #a-> b <-u (test a->b; b=node)
            if (a,b) in DAG.in_edges: #was it already removed?
                
                #Note: it.permutations(parents,2) can't be used in conjuction with the last break since that will cause the cycle to skip triangles for different a's.
                for c in parents[parents!=a]:

                    if pval:
                        if func(data,key[a],key[b],[key[c]])[2] > thres: #does not survive
                            DAG.remove_edge(a,b)
                            break
                    else:
                        if func(data,key[a],key[b],[key[c]])[0] < thres: #does not survive
                            DAG.remove_edge(a,b)
                            break
    return DAG

def triangulation_metric_und(data, node_list, edge_list, thres, metric = "fisherz", pval=False):
    '''Triangulation method tailored for metric undirected'''

    if metric == "fisherz":
        func = independence_tests.CITest.fisherz_test
    if metric == "g2":
        func = independence_tests.CITest.g2_test
    
    key = {node_list[i]:i for i in range(len(node_list))}
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    #for b in np.array(G.nodes)[np.array(G.degree)[:,1].astype("int")>1]: #might be problematic 
    for b in G.nodes:
        if G.degree(b)==0: continue
        parents = np.array(list(G.edges(b)))[:,1]

        for a in parents: #a-- b --u (test a--b; b=node)
            if (a,b) in G.edges: #was it already removed?
                
                #Note: it.permutations(parents,2) cannot be used in conjuction with the last break since that will cause the cycle to skip triangles for different a's.
                for c in parents[parents!=a]:
                    
                    if pval:
                        if func(data,key[a],key[b],[key[c]])[2] > thres: #does not survive
                            G.remove_edge(a,b)
                            break
                    else:
                        if func(data,key[a],key[b],[key[c]])[0] < thres: #does not survive
                            G.remove_edge(a,b)
                            break
    return G

def _loop(G, d):
    assert G.shape[0] == G.shape[1]

    pairs = [(x, y) for x, y in it.combinations(set(range(G.shape[0])), 2)]
    less_d = 0
    for i, j in pairs:
        adj_i = set(np.argwhere(G[i] != 0).reshape(-1, ))
        z = adj_i - {j}  # adj(C, i)\{j}
        if len(z) < d:
            less_d += 1
        else:
            break
    if less_d == len(pairs):
        return False
    else:
        return True

def find_skeleton(data, alpha, ci_test, variant='original',
                  priori_knowledge=None, base_skeleton=None,
                  p_cores=1, s=None, batch=None):
    '''This algorithm was extracted from PC algorithm github repository'''
    
    def test(x, y):

        K_x_y = 1
        sub_z = None
        # On X's neighbours
        adj_x = set(np.argwhere(skeleton[x] == 1).reshape(-1, ))
        z_x = adj_x - {y}  # adj(X, G)\{Y}
        if len(z_x) >= d:
            # |adj(X, G)\{Y}| >= d
            for sub_z in itcombinations(z_x, d):
                sub_z = list(sub_z)
                _, _, p_value = ci_test(data, x, y, sub_z)
                if p_value >= alpha:
                    K_x_y = 0
                    # sep_set[(x, y)] = sub_z
                    break
            if K_x_y == 0:
                return K_x_y, sub_z

        return K_x_y, sub_z

    def parallel_cell(x, y):

        # On X's neighbours
        K_x_y, sub_z = test(x, y)
        if K_x_y == 1:
            # On Y's neighbours
            K_x_y, sub_z = test(y, x)

        return (x, y), K_x_y, sub_z

    if ci_test == 'fisherz':
        ci_test = independence_tests.CITest.fisherz_test
    elif ci_test == 'g2':
        ci_test = independence_tests.CITest.g2_test
    elif ci_test == 'chi2':
        ci_test = independence_tests.CITest.chi2_test
    elif callable(ci_test):
        ci_test = ci_test
    else:
        raise ValueError(f'The type of param `ci_test` expect callable,'
                         f'but got {type(ci_test)}.')

    n_feature = data.shape[1]
    if base_skeleton is None:
        skeleton = np.ones((n_feature, n_feature)) - np.eye(n_feature)
    else:
        row, col = np.diag_indices_from(base_skeleton)
        base_skeleton[row, col] = 0
        skeleton = base_skeleton
    nodes = set(range(n_feature))

    # update skeleton based on priori knowledge
    for i, j in it.combinations(nodes, 2):
        if priori_knowledge is not None and (
                priori_knowledge.is_forbidden(i, j)
                and priori_knowledge.is_forbidden(j, i)):
            skeleton[i, j] = skeleton[j, i] = 0

    sep_set = {}
    d = -1
    while _loop(skeleton, d):  # until for each adj(C,i)\{j} < l
        d += 1
        if variant == 'stable':
            C = deepcopy(skeleton)
        else:
            C = skeleton
        if variant != 'parallel':
            for i, j in it.combinations(nodes, 2):
                if skeleton[i, j] == 0:
                    continue
                adj_i = set(np.argwhere(C[i] == 1).reshape(-1, ))
                z = adj_i - {j}  # adj(C, i)\{j}
                if len(z) >= d:
                    # |adj(C, i)\{j}| >= l
                    for sub_z in it.combinations(z, d):
                        sub_z = list(sub_z)
                        _, _, p_value = ci_test(data, i, j, sub_z)
                        if p_value >= alpha:
                            skeleton[i, j] = skeleton[j, i] = 0
                            sep_set[(i, j)] = sub_z
                            break
        else:
            J = [(x, y) for x, y in it.combinations(nodes, 2)
                 if skeleton[x, y] == 1]
            if not s or not batch:
                batch = len(J)
            if batch < 1:
                batch = 1
            if not p_cores or p_cores == 0:
                raise ValueError(f'If variant is parallel, type of p_cores '
                                 f'must be int, but got {type(p_cores)}.')
            for i in range(int(np.ceil(len(J) / batch))):
                each_batch = J[batch * i: batch * (i + 1)]
                parallel_result = joblib.Parallel(n_jobs=p_cores,
                                                  max_nbytes=None)(
                    joblib.delayed(parallel_cell)(x, y) for x, y in
                    each_batch
                )
                # Synchronisation Step
                for (x, y), K_x_y, sub_z in parallel_result:
                    if K_x_y == 0:
                        skeleton[x, y] = skeleton[y, x] = 0
                        sep_set[(x, y)] = sub_z

    return skeleton, sep_set
