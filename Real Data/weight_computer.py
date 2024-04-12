import itertools as it
import numpy as np

##################################
def weight_num_writer(data, states, filename='weights_num'):
    '''Computes numerical weights and writes them in a txt file.'''
    key = {list(states)[i]:i for i in range(len(states))}

    file = open(filename+'.txt', 'w')
    dsize = len(data)
    for (var1,var2) in it.permutations(states.keys(),2):
        for (st_var1,st_var2) in it.product(*[states[var1],states[var2]]):
            PA = (data[:,key[var1]]==st_var1).sum()/dsize
            if PA == 0 or PA==1:
                continue
            PB = (data[:,key[var2]]==st_var2).sum()/dsize
            PAB= ((data[:,key[var1]]==st_var1)&(data[:,key[var2]]==st_var2)).sum()/dsize
            file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str((PAB - PA*PB)/(PA*(1-PA)))+"\n" )

def weight_num_writer_ord(data, states, order, filename='weights_num'):
    '''Computes numerical weights that satisfy an ordering of the nodes, and writes them in a txt file.'''
    key = {list(states)[i]:i for i in range(len(states))}

    file = open(filename+'.txt', 'w')
    dsize = len(data)
    for (var1,var2) in it.permutations(states.keys(),2):
        if order[var1]<order[var2]:
            for (st_var1,st_var2) in it.product(*[states[var1],states[var2]]):
                PA = (data[:,key[var1]]==st_var1).sum()/dsize
                if PA == 0 or PA==1:
                    continue
                PB = (data[:,key[var2]]==st_var2).sum()/dsize
                PAB= ((data[:,key[var1]]==st_var1)&(data[:,key[var2]]==st_var2)).sum()/dsize
                file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str((PAB - PA*PB)/(PA*(1-PA)))+"\n" )          


def weight_var_importer(file_name):
    '''Imports weight's var from txt file.'''
    
    file = open(file_name, 'r')
    return [[e for e in line.split(":")[0].split(";")] for line in file.readlines()]


def weight_val_importer(file_name):
    '''Imports weight's val from txt file.'''
    
    file = open(file_name, 'r')
    return [float(line.split(":")[1]) for line in file.readlines()]


################################## LEGACY CODE ##################################
import pandas as pd

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
    
    dsize = len(df)
    file = open('mi_num.txt', 'w')
    for (var1,var2) in it.permutations(states.keys(),2):
        if order[var1]<order[var2]:
            for (st_var1,st_var2) in it.product(*[states[var1],states[var2]]):
                PA = (df[var1]==st_var1).sum()/dsize
                PB = (df[var2]==st_var2).sum()/dsize
                if PA==0 or PB==0:
                    continue
                PAB= ((df[var1]==st_var1)&(df[var2]==st_var2)).sum()/dsize
            
                if PAB==0 and PA!=0 and PB!=0:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(-1.)+"\n" )

                else:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(np.log(PAB/(PA*PB))/(-np.log(PAB)))+"\n" )
                    
def pmi_num_writer(df, states):
    '''Computes numerical PMI and writes them in a txt file.'''
    
    file = open('pmi_num.txt', 'w')

    dsize = len(df)
    for (var1,var2) in it.permutations(states.keys(),2):
        if var1<var2:
            for (st_var1,st_var2) in it.product(*[states[var1],states[var2]]):
                PA = (df[var1]==st_var1).sum()/dsize
                PB = (df[var2]==st_var2).sum()/dsize
                PAB= ((df[var1]==st_var1)&(df[var2]==st_var2)).sum()/dsize
            
                if PAB==0 and PA!=0 and PB!=0:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(-np.inf)+"\n" )

                else:
                    file.write( str(var1)+";"+str(var2)+";"+str(st_var1)+";"+str(st_var2) + ":" + str(np.log(PAB/(PA*PB)))+"\n" )
