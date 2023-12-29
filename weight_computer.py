import pandas as pd
import itertools as it
import numpy as np

##################################
def weight_num_writer(df, states, order,filename='weights_num'):
    '''Computes numerical weights and writes them in a txt file.'''
    
    file = open(filename+'.txt', 'w')
                 
    for (var1,var2) in it.permutations(states.keys(),2):
        if order[var1]<order[var2]:
            for (st_var1,st_var2) in it.product(*[states[var1],states[var2]]):
                PA = (df[var1]==st_var1).sum()/len(df)
                if PA == 0 or PA==1:
                    continue
                PB = (df[var2]==st_var2).sum()/len(df)
                PAB= ((df[var1]==st_var1)&(df[var2]==st_var2)).sum()/len(df)
                file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str((PAB - PA*PB)/(PA*(1-PA)))+"\n" )
          
        
def weight_theo_writer(G, states, states_prob):
    '''Computes theoretical weights and writes them in a txt file.'''
    probs = pd.concat(states_prob.values(),ignore_index=True).drop('Cumulative',axis=1)
    
    file = open('weights_theo.txt', 'w')
                 
    for (var1,var2) in it.permutations(states.keys(),2):
        if var1<var2:
            for (st_var1,st_var2) in it.product(*[states[var1],states[var2]]):
                if (var1,var2) in G.edges:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(markover(G, states, probs, [[var1,st_var1],[var2,st_var2]]))+"\n" )
                else:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(0.)+"\n" )
                    
def npmi_num_writer(df, states, order):
    '''Computes numerical NPMI and writes them in a txt file.'''
    
    file = open('mi_num.txt', 'w')
    for (var1,var2) in it.permutations(states.keys(),2):
        if order[var1]<order[var2]:
            for (st_var1,st_var2) in it.product(*[states[var1],states[var2]]):
                PA = (df[var1]==st_var1).sum()/len(df)
                PB = (df[var2]==st_var2).sum()/len(df)
                if PA==0 or PB==0:
                    continue
                PAB= ((df[var1]==st_var1)&(df[var2]==st_var2)).sum()/len(df)
            
                if PAB==0 and PA!=0 and PB!=0:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(-1.)+"\n" )

                else:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(np.log(PAB/(PA*PB))/(-np.log(PAB)))+"\n" )
                    
def pmi_num_writer(df, states):
    '''Computes numerical PMI and writes them in a txt file.'''
    
    file = open('pmi_num.txt', 'w')
                 
    for (var1,var2) in it.permutations(states.keys(),2):
        if var1<var2:
            for (st_var1,st_var2) in it.product(*[states[var1],states[var2]]):
                PA = (df[var1]==st_var1).sum()/len(df)
                PB = (df[var2]==st_var2).sum()/len(df)
                PAB= ((df[var1]==st_var1)&(df[var2]==st_var2)).sum()/len(df)
            
                if PAB==0 and PA!=0 and PB!=0:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(-np.inf)+"\n" )

                else:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(np.log(PAB/(PA*PB)))+"\n" )

                    
##################################

def weight_var_importer(file_name):
    '''Imports weight's var from txt file.'''
    
    file = open(file_name, 'r')
    return [[e for e in line.split(":")[0].split(";")] for line in file.readlines()]


def weight_val_importer(file_name):
    '''Imports weight's val from txt file.'''
    
    file = open(file_name, 'r')
    return [float(line.split(":")[1]) for line in file.readlines()]

##################################

#NOTE: This function scales with exponentially. If all N nodes have 2 states, the function runs over 2^N iterations.
def markover(G, states, probs, node_tips):
    '''Computes theoretical weights between two node states
    G - nx.Digraph,
    states - dictionary with the possible states for each node
    probs - pandas dataframe,
    node_tips - [[start_node,start_node_state], [end_node,end_node_state]]
    '''
    up1,dw1,up2=0,0,0

    for x in it.product(*[states[i] for i in sorted(states)]):
        trues = x[node_tips[0][0]] == node_tips[0][1], x[node_tips[1][0]] == node_tips[1][1]
        
        if trues[0]|trues[1]:
            val = np.prod(probs.loc[((probs[sorted(G.nodes())]==x)|(probs[sorted(G.nodes())].isna())).all(axis=1)].Probability)
        
            if trues[0]&trues[1]: #P(A,B)
                up1+=val
                dw1+=val
                up2+=val
                continue
                
            if trues[0]: #P(B)
                dw1+=val
                
            if trues[1]: #P(A)
                up2+=val
                
    return up1/dw1 - (up2-up1)/(1-dw1) #P(A|B) - P(A|~B)