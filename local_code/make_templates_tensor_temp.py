from variables import *
import os
from os import stat
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools

from collections import Counter
import pandas as pd
import json
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.artist as artists
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs,lobpcg,eigsh


import time

import pickle
import multiprocessing
from multiprocessing import Pool
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import inspect
import gc
import dill
import torch
import copy


from base_classes import saved_template_matrix
import load_templates

from base_functions import *
import cProfile

#TO DO:
#1. For larger Hamiltonians you're generate H algorithm is going to take a long time. 
# So you can improve it by only generating the positions that need to be updated with various k's at each location
# and then updating those locations instead of reconstructing the matrix each time. 
# That way you only need to generate the matrix once, and then populating it is easy.

######################################################
######################################################GLOBAL VARIABLES



class particle():
    def __init__(self,dof_dic):
        self.dof_dic=dof_dic
    
    def __eq__(self, other): 
        if not isinstance(other, particle):
        # don't attempt to compare against unrelated types
            return NotImplemented
        initial=True
        for k in self.dof_dic.keys():
            initial=initial and (self.dof_dic[k] == other.dof_dic[k]) #This requires you to have particle positions as keys so that you're 
            if initial==False:
                break
        return initial

class basis():
    # def __init__(self,**kwargs): #allows arbitrary length particle strings.
    #     for attr in kwargs.keys():
    #         self.__dict__[attr] = kwargs[attr]

    def __init__(self,particle_dic):
        self.particle_dic=particle_dic #particle positions are keys and particle - have to respect this otherwise it doesn't work!

    def __eq__(self, other): 
        if not isinstance(other, basis):
            # don't attempt to compare against unrelated types
            return NotImplemented
        initial=True
        for k in self.particle_dic.keys():
            initial=initial and (self.particle_dic[k] == other.particle_dic[k]) #This requires you to have particle positions as keys so that you're 
            if initial==False:
                break
        return initial


    def eqnoorder(self,other):
        if not isinstance(other,basis):
            return NotImplemented
        #Note that this goes in two parts:
        #1. The first for k tests whether self has all of the states in other.
        #2. The second 'for j' tests whether other has all states in self.
        #You need both otherwise all states in self could be in other but not vice versa.
        if self.particle_dic.keys()==other.particle_dic.keys():
            particle_keys=self.particle_dic.keys()
            for k in particle_keys:
                tempk=True
                match_count=0 #Accounting for having more than one match - forbidden by Pauli, but this is just another check.
                for l in particle_keys:
                    if self.particle_dic[k]==other.particle_dic[l]:
                        match_count+=1
                if match_count==1:
                    tempk=True
                else:
                    tempk=False
                    break
            particle_keys=self.particle_dic.keys()
            for j in particle_keys:
                tempj=True
                match_count=0 #Accounting for having more than one match - forbidden by Pauli, but this is just another check.
                for l in particle_keys:
                    if other.particle_dic[j]==self.particle_dic[l]:
                        match_count+=1
                if match_count==1:
                    tempj=True
                else:
                    tempj=False
                    break
        else:
            tempj=False
        return tempk and tempj

                    
        
    def swaps(self,other):
        if not isinstance(other, basis):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if basis.eqnoorder(self,other)==False:
            # Meaningless on states that are not the same up to swaps.
            return NotImplemented
        list0=[x for x in self.particle_dic.keys()]#Just to make sure the order doesn't change
        list1=[self.particle_dic[x] for x in list0]
        list2=[other.particle_dic[x] for x in list0]
        swapcount=0
        for i in range(0,len(list2)):
            index_difference=(list1.index(list1[i])-list2.index(list1[i]))
            if index_difference==0:
                continue
            else:
                if index_difference>0:
                    for j in range(0,index_difference):
                        list2=swap(list2,list2.index(list1[i]),list2.index(list1[i])+1)
                        swapcount+=1
                if index_difference<0:
                    for j in range(0,-index_difference):
                        list2=swap(list2,list2.index(list1[i]),list2.index(list1[i])-1)
                        swapcount+=1
        return swapcount


    
    
def swap(alist,index1,index2): #Have defined outside the class, maybe it would have been better for this to be a method.
    alist[index1],alist[index2]=alist[index2],alist[index1]
    return alist


#########################################################Testing
def eyepreservation_0(state):#Leave this at three states
    particle_list=[state.particle_dic[x] for x in state.particle_dic.keys()]
    symbol_map0={0:'Q0',1:'Q1',2:'Q2',3:'Q3'}
    symbol_map1={0:'A',1:'B'}
    symbol_map2={0:u'\u2191',1:u'\u2193'}
    helpful_string=''
    for i in particle_list:
        for k in i.dof_dic.keys():
            if k==1:
                helpful_string=helpful_string+symbol_map1[i.dof_dic[k]]
            elif k==2:
                helpful_string=helpful_string+symbol_map2[i.dof_dic[k]]
            elif k==0:
                helpful_string=helpful_string+symbol_map0[i.dof_dic[k]]

        helpful_string=helpful_string+';'
    return helpful_string

def eyepreservation(state):#Leave this at three states
    particle_list=[state.particle_dic[x] for x in state.particle_dic.keys()]
    layer_map={0:'L1',1:'L2'}
    q_map={0:'q2',1:'q3'}
    symbol_map0={0:'Q0',1:'Q1',2:'Q2',3:'Q3'}
    symbol_map1={0:'A',1:'B'}
    symbol_map2={0:u'\u2191',1:u'\u2193'}
    helpful_string=''
    for i in particle_list:
        for k in i.dof_dic.keys():
            if k==0:
                helpful_string=helpful_string+layer_map[i.dof_dic[k]]
            elif k==1:
                helpful_string=helpful_string+'('+str(i.dof_dic[k][0])+'q2'+str(i.dof_dic[k][1])+'q3'+')'
            elif k==3:
                helpful_string=helpful_string+symbol_map2[i.dof_dic[k]]
            elif k==2:
                helpful_string=helpful_string+symbol_map1[i.dof_dic[k]]

        helpful_string=helpful_string+';'
    return helpful_string


def flatten(l):
    return [item for sublist in l for item in sublist]

def generate_basis(number_of_particles,particle_dofno_dic):
    basis_list=[]
    particle_dic={}
    dof_dic={}
    dof_keys=list(particle_dofno_dic.keys())
    single_particle_dofs=[x for x in range(particle_dofno_dic[dof_keys[0]])]
    single_particle_dofst=[x for x in range(particle_dofno_dic[dof_keys[1]])]
    single_particle_dofs=list(itertools.product(single_particle_dofs,single_particle_dofst))
    for k in range(2,len(dof_keys)):
        single_particle_dofst=[x for x in range(particle_dofno_dic[dof_keys[k]])]
        single_particle_dofs=list(itertools.product(single_particle_dofs,single_particle_dofst))
        single_particle_dofs=[(*rest, a) for rest, a in single_particle_dofs]
    initial_list=list(itertools.combinations(single_particle_dofs,number_of_particles))
    for i in initial_list:
        for k in range(0,number_of_particles):
            for j in range(len(i[k])):
                dof_dic[j]=i[k][j]#+1 just to index it as 1,2,3 rather than 0,1,2
            particlen=particle(dof_dic=dict(dof_dic))
            particle_dic[k+1]=particlen
        # print([vars(particle_dic[x]) for x in particle_dic.keys()])
        temp=basis(particle_dic=dict(particle_dic))
        # print(eyepreservation(temp))
        basis_list.append(temp)
    return basis_list

def generate_layer_q_pairs(shell_count,q_vecs):
    #probably good to give the q_vecs in terms of q2,q3 i.e q1=np.array([-1,-1]), q2=[1,0],q3=[0,1]
    #1. First generate all the allowed k vectors
    layer_q_pairs=[]
    layer_q_pairs_duplicates=[]
    layer_index=0
    layer_q_pairs.append([(0,np.array([0,0]))])#To choose Gamma or K centered lattice can just control by different initializations (K -centered just one at 0 shell, Gamma have six)
    for shell in range(1,shell_count):
        previous_pairs=layer_q_pairs[shell-1]#remember that in gamma centered case you have a mix of layers, in K case you a definite layer at each shell
        shell_pairs=[]
        shell_pairs_dup=[]
        for pair in previous_pairs:
            current_layer=(pair[0]+1)%2
            for q in q_vecs:
                q=((-1)**pair[0])*q#second layer has negative q vectors, first layer has positive
                temp_pair=(current_layer,pair[1]+q)
                shell_pairs_dup.append(temp_pair)
                #Module to remove duplicate (layer,q) pairs
                inlist=False
                for i in flatten(layer_q_pairs):
                    if (i[0]==temp_pair[0] and np.array_equal(i[1],temp_pair[1])):
                        inlist=True
                        break
                if inlist==False:
                    shell_pairs.append(temp_pair)
                # if temp_pair in flatten(layer_q_pairs):
                #     continue
                # else:
                #     shell_pairs.append(temp_pair)
        layer_q_pairs_duplicates.append(shell_pairs_dup)#Just to have information on what's present at each shell
        layer_q_pairs.append(shell_pairs)
        layer_q_pairs_tuple=[]
        for shell in layer_q_pairs:
            shell_pairs=[]
            for pair in shell:
                temp_tuple=tuple(pair[1])
                new_pair=(pair[0],temp_tuple)
                shell_pairs.append(new_pair)
            layer_q_pairs_tuple.append(shell_pairs)

    #Note - by keeping it as nested lists, I'm retaining the information of which shell each unique site was generated at.
    return layer_q_pairs_tuple




testnonlayer2={'sublattice':2,'spin':2}



# ################################################GAMMA CENTERING##############################################

def generate_layer_q_pairs_gamma(shell_count,q_vecs,center):
    #probably good to give the q_vecs in terms of q2,q3 i.e q1=np.array([-1,-1]), q2=[1,0],q3=[0,1]
    #1. First generate all the allowed k vectors
    layer_q_pairs=[]
    layer_q_pairs_duplicates=[]
    layer_index=0
    #Note here that q_vecs is a list of q_1, q_2, q_3 with each expressed as as an array of q2, q3
    first_layer=[]
    if center=='Gamma':
        for qi in q_vecs:
            first_layer.append((0,qi))#To choose Gamma or K centered lattice can just control by different initializations (K -centered just one at 0 shell, Gamma have six)
        for qi in q_vecs:
            first_layer.append((1,-qi))
    elif center=='K':
        first_layer.append((0,np.array([0,0])))
    layer_q_pairs.append(first_layer)
    for shell in range(1,shell_count):
        previous_pairs=layer_q_pairs[shell-1]#remember that in gamma centered case you have a mix of layers, in K case you a definite layer at each shell
        shell_pairs=[]
        shell_pairs_dup=[]
        for pair in previous_pairs:
            current_layer=(pair[0]+1)%2
            for q in q_vecs:
                q=((-1)**pair[0])*q#second layer has negative q vectors, first layer has positive
                temp_pair=(current_layer,pair[1]+q)
                shell_pairs_dup.append(temp_pair)
                #Module to remove duplicate (layer,q) pairs
                inlist=False
                for i in flatten(layer_q_pairs):
                    if (i[0]==temp_pair[0] and np.array_equal(i[1],temp_pair[1])):
                        inlist=True
                        break
                if inlist==False:
                    shell_pairs.append(temp_pair)
                # if temp_pair in flatten(layer_q_pairs):
                #     continue
                # else:
                #     shell_pairs.append(temp_pair)
        layer_q_pairs_duplicates.append(shell_pairs_dup)#Just to have information on what's present at each shell
        layer_q_pairs.append(shell_pairs)
        layer_q_pairs_tuple=[]
        for shell in layer_q_pairs:
            shell_pairs=[]
            for pair in shell:
                temp_tuple=tuple(pair[1])
                new_pair=(pair[0],temp_tuple)
                shell_pairs.append(new_pair)
            layer_q_pairs_tuple.append(shell_pairs)
    if shell_count==1:
        layer_q_pairs_tuple=[]
        for shell in layer_q_pairs:
            shell_pairs=[]
            for pair in shell:
                temp_tuple=tuple(pair[1])
                new_pair=(pair[0],temp_tuple)
                shell_pairs.append(new_pair)
            layer_q_pairs_tuple.append(shell_pairs)


    #Note - by keeping it as nested lists, I'm retaining the information of which shell each unique site was generated at.
    return layer_q_pairs_tuple



testnonlayer={'sublattice':2,'spin':2}
def generate_shell_basis_gamma(shell_count,q_vecs,number_of_particles,nonlayer,center):
    basis_list=[]
    particle_dic={}
    dof_dic={}
    layer_q_pairs=flatten(generate_layer_q_pairs_gamma(shell_count=shell_count,q_vecs=q_vecs,center=center))#function outputs nested lists for shells
    nonlayerdofs=[range(nonlayer[x]) for x in nonlayer.keys()]
    single_particle_dofs=list(itertools.product(layer_q_pairs,nonlayerdofs[0]))
    #single_particle_dofs=list(set(single_particle_dofs)) Annoying that this changes the order. Let me try
    single_particle_dofs=sorted(set(single_particle_dofs), key=single_particle_dofs.index)
    if len(nonlayerdofs)>1:
        for d in range(1,len(nonlayerdofs)):
            single_particle_dofs=list(itertools.product(single_particle_dofs,nonlayerdofs[d]))
    single_particle_dofs=[(*rest, a) for rest, a in single_particle_dofs]

    for i in range(len(single_particle_dofs)):
        single_particle_dofs[i]=(single_particle_dofs[i][0][0],single_particle_dofs[i][0][1])+single_particle_dofs[i][1:]


    initial_list=list(itertools.combinations(single_particle_dofs,number_of_particles))
    for i in initial_list:
        for k in range(0,number_of_particles):
            for j in range(len(i[k])):
                dof_dic[j]=i[k][j]#+1 just to index it as 1,2,3 rather than 0,1,2
            particlen=particle(dof_dic=dict(dof_dic))
            particle_dic[k+1]=particlen
        # print([vars(particle_dic[x]) for x in particle_dic.keys()])
        temp=basis(particle_dic=dict(particle_dic))
        # print(eyepreservation(temp))
        basis_list.append(temp)
    return basis_list






            

            



    
    
    





###########################################################Defining the pauli tensors...#############################



def tpp(state_list,pauli_dic,prefactor):
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    particle_dic_temp1={}#Idea is to constantly overwrite
    dof_dic_temp={}
    if len(pauli_dic.keys())!=len(state_list[0].particle_dic[1].dof_dic.keys()):
        print("Need to give a pauli for each particle!")
        print(pauli_dic.keys())
        return NotImplemented
    else:
        for state in state_list:
            result_list=[]
            for k in state.particle_dic.keys():
                coeff=1
                for l in range(0,dof):
                    dof_dic_temp[l]=pauli_dic[l](state.particle_dic[k].dof_dic[l])[1]
                    coeff=coeff*pauli_dic[l](state.particle_dic[k].dof_dic[l])[0]
                particlen=particle(dof_dic=dict(dof_dic_temp))
                particle_dic_temp1[k]=particlen
                for j in state.particle_dic.keys():
                    if j!=k:
                        particle_dic_temp1[j]=state.particle_dic[j]
                new_state=basis(particle_dic=dict(particle_dic_temp1))
                # print(eyepreservation(new_state))
                result=(coeff,new_state)
                result_list.append(result)
            # for result in result_list:
                    # print(f"Input state {eyepreservation(state)}")
                    # print(f"Output state {eyepreservation(result[1])}")
                    # print(len(result_list))
                # particle_dic_temp1.clear()
        #2.6 Now I need to map the states into some standard Hamiltonian ordering.
        #I need to do an unordered match of the state to a state in the basis, and then I need to count the number of swaps
    #Have to do this because, since I've introduced the basis as a class - seperate instances of the class count as different objects...    
            for result in result_list:
                for i in state_list:
                    if basis.eqnoorder(i,result[1]):
                        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
                        swapcount=basis.swaps(result[1],i)
                        position1=(state_list.index(i),state_list.index(state))
                        temp_H[position1]=((-1)**swapcount)*prefactor*result[0]
                        H1=H1+temp_H
            # position1=(ordered_basis.index(basis1),ordered_basis.index(state)) #in the matrix, row index is final state, col. index is final state
            #the SO coupling brings imaginary terms in, unfortunately...
            # bit of a hack but I'm just using the pair to seperate out the coefficent from the state label
            
        return H1

def tpp_new(state_list,pauli_dic,prefactor):
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    particle_dic_temp1={}#Idea is to constantly overwrite
    dof_dic_temp={}
    if len(pauli_dic.keys())!=len(state_list[0].particle_dic[1].dof_dic.keys()):
        print("Need to give a pauli for each particle!")
        print(pauli_dic.keys())
        return NotImplemented
    else:
        for state in tqdm(state_list):
            result_list=[]
            for k in state.particle_dic.keys():
                coeff=1
                for l in range(0,dof):
                    dof_dic_temp[l]=pauli_dic[l](state.particle_dic[k].dof_dic[l])[1]
                    coeff=coeff*pauli_dic[l](state.particle_dic[k].dof_dic[l])[0]
                particlen=particle(dof_dic=dict(dof_dic_temp))
                particle_dic_temp1[k]=particlen
                for j in state.particle_dic.keys():
                    if j!=k:
                        particle_dic_temp1[j]=state.particle_dic[j]
                new_state=basis(particle_dic=dict(particle_dic_temp1))
                # print(eyepreservation(new_state))
                result=(coeff,new_state)
                result_list.append(result)
            # for result in result_list:
                    # print(f"Input state {eyepreservation(state)}")
                    # print(f"Output state {eyepreservation(result[1])}")
                    # print(len(result_list))
                # particle_dic_temp1.clear()
        #2.6 Now I need to map the states into some standard Hamiltonian ordering.
        #I need to do an unordered match of the state to a state in the basis, and then I need to count the number of swaps
    #Have to do this because, since I've introduced the basis as a class - seperate instances of the class count as different objects...    
            for result in result_list:
                for i in state_list:
                    if basis.eqnoorder(i,result[1]):
                        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
                        swapcount=basis.swaps(result[1],i)
                        position1=(state_list.index(i),state_list.index(state))
                        H1[position1]=H1[position1]+((-1)**swapcount)*prefactor*result[0]
            # position1=(ordered_basis.index(basis1),ordered_basis.index(state)) #in the matrix, row index is final state, col. index is final state
            #the SO coupling brings imaginary terms in, unfortunately...
            # bit of a hack but I'm just using the pair to seperate out the coefficent from the state label
            
        return H1


def tpp_time(state_list,pauli_dic,prefactor):
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    particle_dic_temp1={}#Idea is to constantly overwrite
    dof_dic_temp={}
    if len(pauli_dic.keys())!=len(state_list[0].particle_dic[1].dof_dic.keys()):
        print("Need to give a pauli for each particle!")
        print(pauli_dic.keys())
        return NotImplemented
    else:
        times=[]
        for state in state_list:
            start=time.time()
            result_list=[]
            for k in state.particle_dic.keys():
                coeff=1
                for l in range(0,dof):
                    dof_dic_temp[l]=pauli_dic[l](state.particle_dic[k].dof_dic[l])[1]
                    coeff=coeff*pauli_dic[l](state.particle_dic[k].dof_dic[l])[0]
                particlen=particle(dof_dic=dict(dof_dic_temp))
                particle_dic_temp1[k]=particlen
                for j in state.particle_dic.keys():
                    if j!=k:
                        particle_dic_temp1[j]=state.particle_dic[j]
                new_state=basis(particle_dic=dict(particle_dic_temp1))
                # print(eyepreservation(new_state))
                result=(coeff,new_state)
                result_list.append(result)
            # for result in result_list:
                    # print(f"Input state {eyepreservation(state)}")
                    # print(f"Output state {eyepreservation(result[1])}")
                    # print(len(result_list))
                # particle_dic_temp1.clear()
        #2.6 Now I need to map the states into some standard Hamiltonian ordering.
        #I need to do an unordered match of the state to a state in the basis, and then I need to count the number of swaps
    #Have to do this because, since I've introduced the basis as a class - seperate instances of the class count as different objects...    
            end=time.time()
            time1=end-start
            start=time.time()
            for result in result_list:
                for i in state_list:
                    start=time.time()
                    if basis.eqnoorder(i,result[1]):
                        end=time.time()
                        time3=end-start
                        start=time.time()
                        swapcount=basis.swaps(result[1],i)
                        end=time.time()
                        time5=end-start
                        position1=(state_list.index(i),state_list.index(state))
                        start=time.time()
                        end=time.time()
                        time6=end-start
                        H1[position1]=H1[position1]+((-1)**swapcount)*prefactor*result[0]
                        times.append([time1,time3,time5,time6])
            end=time.time()
            time2=end-start
            #times.append([time1,time3,time5,time6])
            # position1=(ordered_basis.index(basis1),ordered_basis.index(state)) #in the matrix, row index is final state, col. index is final state
            #the SO coupling brings imaginary terms in, unfortunately...
            # bit of a hack but I'm just using the pair to seperate out the coefficent from the state label
            
        return H1,times




def inspect_elements(matrix,state_list,state_array=None):
    if isinstance(state_array,np.ndarray)==False:
        states_considered=state_list
    else:
        states_considered=list(np.array(state_list)[state_array])

    for i in states_considered:
        state_index=state_list.index(i)
        testwf=np.zeros(len(state_list),dtype=complex)
        testwf[state_index]=1
        reswf=matrix.dot(testwf)
        print(f"STATE {eyepreservation(i)}")
        for j in range(len(reswf)):
            if (np.isclose(np.abs(reswf[j]),0))==False:
                # ktheta=np.sqrt(np.vdot(kd,kd))*np.sin(theta0/2)
                # q1=2*ktheta*k1/(np.sqrt(np.vdot(k1,k1)))
                # q2=2*ktheta*k2/(np.sqrt(np.vdot(k2,k2)))
                # q3=2*ktheta*k3/(np.sqrt(np.vdot(k3,k3)))
                # kdic={0:k0,1:k0+q1,2:k0+q2,3:k0+q3}
                #kvalue1=kdic[state_list[i].particle_dic[1].dof_dic[0]]
                #kvalue2=kdic[state_list[i].particle_dic[2].dof_dic[0]]
                #theta1=get_theta(kvalue1)-theta0/2
                #theta2=get_theta(kvalue2)-theta0/2
                #coeff1=np.cos(theta1)*v*np.sqrt(np.dot(kvalue1,kvalue1))
                #coeff2=np.cos(theta2)*v*np.sqrt(np.dot(kvalue2,kvalue2))
                print(f"Initial state: {eyepreservation(i)}, Output state: {eyepreservation(state_list[j])}, Amplitude {reswf[j]}, mod amp {np.abs(reswf[j])}")#, c1: {coeff1},c2:{coeff2}
        
        #exit()
        






def tq0(sigma):
    if sigma==(0,0):
        return (1,sigma)
    else:
        return (0,sigma)
    
def tqproj1(sigma):
    if sigma!=(0,0):
        return (1/3,(-1,-1))
    else:
        return (0,sigma)

def tqproj2(sigma):
    if sigma!=(0,0):
        return (1/3,(1,0))
    else:
        return (0,sigma)

def tqproj3(sigma):
    if sigma!=(0,0):
        return (1/3,(0,1))
    else:
        return (0,sigma)



def find_matching_indices(A, B):
    # A: N x m x d
    # B: N x m x m x d
    N, m, d = A.shape
    matching_indices = []

    # Sort rows of each mxd tensor in A to treat them as sets
    sorted_A = torch.sort(A, dim=1)[0]  # Sort rows along the m dimension

    for i in range(N):
        a_sorted_rows = sorted_A[i]  # Nth mxd tensor in A sorted by rows

        for j in range(m):
            # Sort the rows of the mxd tensor B[i, j]
            b_sorted_rows = torch.sort(B[i, j], dim=1)[0]

            # Compare the sorted row tensors (row-order-agnostic comparison)
            if torch.equal(a_sorted_rows, b_sorted_rows):
                matching_indices.append((i, j))
    
    return matching_indices

def find_mapped_tensor(input_tensor,output_tensor):
    
    # Step 1: Sort along the n-axis for each tensor along the d-axis
    A_sorted = torch.sort(input_tensor, dim=1)[0]
    B_sorted = torch.sort(output_tensor, dim=1)[0]

    # Step 2: Compare each sorted tensor in A_sorted with all tensors in B_sorted
    A_expanded = A_sorted.unsqueeze(1)  # Shape: (N, 1, n, d)
    B_expanded = B_sorted.unsqueeze(0)  # Shape: (1, N, n, d)

    print(f'A expanded shape: {A_expanded.shape}')
    print(f'B expanded shape: {B_expanded.shape}')

    # Perform element-wise comparison and reduce across the n-axis and d-axis
    comparison = (A_expanded == B_expanded).all(dim=-1).all(dim=-1)  # Shape: (N, N)
    
    # Step 3: Find the indices where there is a match
    # We can use nonzero to get the matching indices
    matching_indices = comparison.nonzero(as_tuple=False)  # Shape: (num_matches, 2)

    # `matching_indices` will contain pairs of indices (i, j)
    # where A_sorted[i] matches B_sorted[j]

    # To return a list of tuples:
    #matching_indices_list = [(i.item(), j.item()) for i, j in matching_indices]
    
    return matching_indices

def find_permutation_equivalents(A, B):
    # Sort the rows of each nxd tensor in A and B along the second dimension (rows)
    A_sorted = torch.sort(A, dim=1)[0]  # Sort each nxd tensor by rows
    B_sorted = torch.sort(B, dim=1)[0]

    # Compare the sorted tensors element-wise
    matches = (A_sorted == B_sorted).all(dim=(1, 2))  # Compare over the last two dimensions (n, d)
    
    # Find the indices where A tensors match with B tensors
    matching_indices = matches.nonzero(as_tuple=True)[0]  # Indices of matching tensors
    
    return matching_indices

def count_zeros(tensor):
    N,n,_=tensor.shape
    zero_mask = (tensor == 0)

    # Count the number of zeros in each mxm tensor
    zero_counts = zero_mask.sum(dim=(1, 2))  # NxNx tensor, where each element is the count of zeros in the corresponding mxm tensor

    # Find indices where the count of zeros is exactly m
    indices = (zero_counts == n).nonzero(as_tuple=True)
    
    

    return indices

def find_matches(result_tensor,basis_tensor):
    matched_indices=[]
    start_indices=[]
    #Can insert prep code here like finding the row sums etc to cut down the number of states you need to compare.

    N,n,n,d=result_tensor.shape
    for state_index in tqdm(range(N)):
        for particle_index in range(n):
            state_tensor=result_tensor[state_index,particle_index]
            unique_rows,counts=torch.unique(state_tensor,dim=0,return_counts=True)
            duplicate_rows = unique_rows[counts > 1]
            
            if len(duplicate_rows)>0:#If there are duplicate rows, then the state is not a valid fermionic state.
                
                #Could instead append an mxd tensor of all zeros and then exclude those.
                continue
            else:
                basis_expanded = basis_tensor.unsqueeze(1)  # Shape becomes (N, 1, m, d)
                state_tensor=state_tensor.unsqueeze(0).unsqueeze(2)
                row_diffs=state_tensor-basis_expanded
                row_diffs=row_diffs.transpose(1,2)

                sum_row_diffs=torch.abs(row_diffs).sum(dim=3)
                # print(f'sum row diffs')
                # print(sum_row_diffs[15])
                # exit()
                zeros=count_zeros(sum_row_diffs)
                matched_basis_indices=[z.item() for z in zeros[0]]
                
                zero_indices=torch.nonzero(sum_row_diffs[zeros]==0)
                
                #zero_indices_dims= #non-zero index, #particle index, #non zero particle index
                index_pairs=zero_indices[:,np.arange(zero_indices.shape[1])]

                permutation_sum=((np.abs(zero_indices[:,1]-zero_indices[:,2]))/2).sum()

                matched_indices.append((state_index,particle_index,matched_basis_indices,permutation_sum))
                # print(f'result state tensor')
                # print(state_tensor)
                # print(f'comparison basis tensor:')
                # print(basis_expanded[15])
                # print(f'difference tensor:')
                # print(row_diffs[15])
                # sum_row_diffs=torch.abs(row_diffs).sum(dim=3)
                # print(f'sum row diffs')
                # print(sum_row_diffs[15])
                # print(f'zero counts')
                # zeros=count_zeros(sum_row_diffs)
                # print(zeros)
                # print(f'inital result state')
                # print(state_tensor)
                # print(f'claimed matching basis state')
                # print(basis_tensor[zeros[0][0]])
                # print(f'zero indices: {torch.nonzero(sum_row_diffs[zeros]==0)}')
                
                

                # print(f'zero indices: {zero_indices.shape}')
                
                # print(f'permutation: {zero_indices[:,1].sum()-zero_indices[:,2].sum()}')
                
                                    
            
    return matched_indices

import torch.multiprocessing as mp

def process_chunk(chunk_start, chunk_end, result_tensor, basis_tensor):
    matched_indices = []
    N, n, _, d = result_tensor.shape
    
    for state_index in range(chunk_start, chunk_end):
        for particle_index in range(n):
            state_tensor = result_tensor[state_index, particle_index]
            unique_rows, counts = torch.unique(state_tensor, dim=0, return_counts=True)
            duplicate_rows = unique_rows[counts > 1]
            
            if len(duplicate_rows) > 0:
                continue
            else:
                basis_expanded = basis_tensor.unsqueeze(1)
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(2)
                row_diffs = state_tensor - basis_expanded
                row_diffs = row_diffs.transpose(1, 2)

                sum_row_diffs = torch.abs(row_diffs).sum(dim=3)
                zeros = count_zeros(sum_row_diffs)
                matched_basis_indices = [z.item() for z in zeros[0]]
                
                zero_indices = torch.nonzero(sum_row_diffs[zeros] == 0)
                permutation_sum = ((torch.abs(zero_indices[:, 1] - zero_indices[:, 2])) / 2).sum()

                matched_indices.append((state_index, particle_index, matched_basis_indices, permutation_sum))
    
    return matched_indices

def find_matches_parallel(result_tensor, basis_tensor, num_processes=4):
    N = result_tensor.shape[0]
    chunk_size = N // num_processes
    
    pool = mp.Pool(processes=num_processes)
    chunks = [(i * chunk_size, (i + 1) * chunk_size if i < num_processes - 1 else N) for i in range(num_processes)]
    
    results = []
    for chunk_start, chunk_end in chunks:
        results.append(pool.apply_async(process_chunk, (chunk_start, chunk_end, result_tensor, basis_tensor)))
    
    matched_indices = []
    for result in tqdm(results):
        matched_indices.extend(result.get())
    
    pool.close()
    pool.join()
    
    return matched_indices

def make_H_from_indices(N,matching_indices,coeff_tensor,pauli_dic):
    H0=torch.zeros((N,N),dtype=precision)
    #Let's first do it in sequence and then we can parrallelize
    # print(matching_indices)
    for matched_state in matching_indices:
        if matched_state[2]==[]:
            continue
        else:
            coeff_res=coeff_tensor[matched_state[0],matched_state[1],matched_state[1],:]

            # print(f'matched state: {test_tensor_states[matched_state[0]]}')
            # print(f'initial state: {matched_state[0]}')
            # print(f'particle index: {matched_state[1]}')
            # print(f'matched basis states: {test_tensor_states[matched_state[2]]}')
            # print(f'coeff tensor: {coeff_res}')
            # print(f' coeff {torch.prod(coeff_res)}')
            # print(f'permutation sum: {matched_state[-1]}')
            # exit()
            
            coeff=torch.prod(coeff_tensor[matched_state[0],matched_state[1],matched_state[1],:],dim=-1)
            #coeff=coeff_tensor[matched_state[0],matched_state[1],matched_state[1],4]
            H0[matched_state[2][0],matched_state[0]]=H0[matched_state[2][0],matched_state[0]]+1*((-1)**(matched_state[-1].item()))*coeff#will change with pauli action
    return H0

def time_it(func):
    def wrapper(*args, **kwargs):
        if kwargs.get('track_time', False):
            timing_info = {}  # Dictionary to store sub-function runtimes
            def timed_func(func_to_time, func_name, *args):
                start_time = time.time()
                result = func_to_time(*args)
                end_time = time.time()
                elapsed_time = end_time - start_time
                timing_info[func_name] = elapsed_time
                return result

            # Replace original function calls with timed ones
            result = func(timed_func, *args, **kwargs)
            return result, timing_info  # Return result and timing info
        else:
            # Call the original function without timing
            return func(*args, **kwargs)
    return wrapper

def pauli_action(pauli_dic,basis_tensor):
    N,n,d=basis_tensor.shape
    input_tensor=basis_tensor.clone()
    input_expanded = input_tensor.unsqueeze(2)  # Shape: (N, m, 1, d)
    input_expanded=input_expanded.repeat(1,1,n,1)
    input_expanded.transpose_(1,2)
    res_states=input_expanded.clone()
    res_coeff=res_states.clone()
    res_coeff=torch.complex(res_coeff.float(),torch.zeros_like(res_coeff).float())
    
    #plus_tensor=torch.zeros((N,n,n,d),dtype=torch.int)

    # for particle_ind in range(n):
    #     #for key in pauli_dic.keys():
    #     pmask[:,particle_ind,particle_ind,4]=1
    #     plus_tensor=plus_tensor+pmask*(pauli_dic[3](res_states)[1])
    # for particle_ind in range(n):
    #     res_states[:,particle_ind,:,]

    for particle_ind in range(n):
        for pauli_dic_key in pauli_dic.keys():
            if pauli_dic_key<1: 
                dof_key=pauli_dic_key
                res_states[:,particle_ind,particle_ind,dof_key]=pauli_dic[pauli_dic_key](input_expanded[:,particle_ind,particle_ind,dof_key])[1]
                res_coeff[:,particle_ind,particle_ind,dof_key]=pauli_dic[pauli_dic_key](input_expanded[:,particle_ind,particle_ind,dof_key])[0]
            elif pauli_dic_key==1: #The q dof which need to be processed as a pair
                dof_key=[1,2]
                res_states[:,particle_ind,particle_ind,dof_key]=pauli_dic[pauli_dic_key](input_expanded[:,particle_ind,particle_ind,dof_key])[1]
                res_coeff[:,particle_ind,particle_ind,dof_key]=pauli_dic[pauli_dic_key](input_expanded[:,particle_ind,particle_ind,dof_key])[0]
            else:
                dof_key=pauli_dic_key+1
                res_states[:,particle_ind,particle_ind,dof_key]=pauli_dic[pauli_dic_key](input_expanded[:,particle_ind,particle_ind,dof_key])[1]
                res_coeff[:,particle_ind,particle_ind,dof_key]=pauli_dic[pauli_dic_key](input_expanded[:,particle_ind,particle_ind,dof_key])[0]        
    
    return res_states,res_coeff

@time_it
def tpp_from_tensor(timed_func,state_list,tensor_list,pauli_dic,prefactor,**kwargs):
    with torch.no_grad():
        N,n,d=tensor_list.shape
        px_tensor=torch.zeros((N,n,n,d),dtype=torch.int)
        
        for particle_ind in range(n):
            #would just add other pauli actions here if need be.
            px_tensor[:,particle_ind,particle_ind,3]=1

        
        tensor_list_torch=torch.tensor(tensor_list)


        res_tensor=tensor_list_torch.clone()
        
        res_expanded = res_tensor.unsqueeze(2)  # Shape: (N, m, 1, d)
        
        # Repeat A_expanded along the new dimension to get shape (N, m, m, d)
        input_repeated = res_expanded.repeat(1, 1, n, 1)
        input_repeated.transpose_(1,2)
        

        

        test_states,test_coeff=pauli_action(pauli_dic,tensor_list_torch)

        matching_indices=timed_func(find_matches,"find_matches",test_states,tensor_list_torch)
        #matching_indices=find_matches_parallel(test_states,tensor_list_torch,num_processes=4)

        res_H=timed_func(make_H_from_indices,"make_H_from_indices",N,matching_indices,test_coeff,pauli_dic)
    return res_H


def runtime_comparison(new_func,old_func,particle_range):
    fig=make_subplots(rows=1,cols=1)
    old_runtimes=[]
    new_runtimes=[]
    truth=[]
    for p_count in tqdm(particle_range):
        shells=2
        particles=p_count
        center='gamma'
        shell_basis_dicts=generate_shell_basis_gamma(shell_count=shells,q_vecs=tqs,number_of_particles=particles,nonlayer=testnonlayer,center=center)
        test_tensor_states=make_basis_tensors(shell_basis_dicts)
        
        
        test_new,test_times=new_func(shell_basis_dicts,test_tensor_states,{0:p0,1:t0,2:p0,3:px},1,track_time=True)
        # for key in test_times.keys():
        #     new_runtimes.append(test_times[key])
        new_runtimes.append([test_times[k] for k in sorted(test_times.keys())])

        start=time.time()
        test_old=old_func(shell_basis_dicts,{0:p0,1:t0,2:p0,3:px},1)
        end=time.time()
        old_runtimes.append(end-start)
        truth.append(np.allclose(test_new,test_old))
        



    fig.add_trace(go.Scatter(x=np.arange(1,len(old_runtimes)+1),y=old_runtimes,mode='lines',name='Old method'),row=1,col=1)
    fig.add_trace(go.Scatter(x=np.arange(1,len(new_runtimes)+1),y=np.array(new_runtimes).sum(axis=1),mode='lines',name='New method'),row=1,col=1)       
    fig.update_layout(title_text=f'Comparison of old and new methods for tpp, same result:{truth} ',showlegend=True)
    fig.show()
        

def tpp_performance_metrics(tpp_func,tpp_func_kwargs_dict,particle_range):
    construction_times=[]
    frac_nonzero=[]
    memory=[]
    for particle_nos in particle_range:
        print(f'Making {particle_nos} particle matrix')
        shell_basis_dicts=generate_shell_basis_gamma(shell_count=shells,q_vecs=tqs,number_of_particles=particle_nos,nonlayer=testnonlayer,center=center)
        H1=tpp_func(shell_basis_dicts,**tpp_func_kwargs_dict)
        frac_nonzero.append(np.count_nonzero(H1)/np.prod(H1.shape))
        memory.append(H1.nbytes/1024/1024) #In MB
        start=time.time()
        H1=tpp_func(shell_basis_dicts,**tpp_func_kwargs_dict)
        end=time.time()
        construction_times.append(end-start)
    

    return np.array(construction_times).flatten(),np.array(frac_nonzero).flatten(),np.array(memory).flatten()


def all_elements_component(state_list):
    particle_dof_keys=state_list[0].particle_dic[1].dof_dic.keys()
    dofs_values={x:[] for x in particle_dof_keys}
    
    
    for dof_key in dofs_values.keys():
        dofs_list = [particleobj.dof_dic[dof_key] for state in state_list for particleobj in state.particle_dic.values()]
        dofs_set=list(set(dofs_list))
        dofs_values[dof_key]=dofs_set
    
    return dofs_values

#Let's test constructing the tensor

def make_basis_tensors(state):
    tensor_list=[]
    for particle_key in state.particle_dic.keys():
        particle_tensor_list=[]
        for dof_key in state.particle_dic[particle_key].dof_dic.keys():
            dof_value=state.particle_dic[particle_key].dof_dic[dof_key]
            if type(dof_value)==int:
                particle_tensor_list.append(dof_value)
            elif type(dof_value)==tuple:
                particle_tensor_list.extend(dof_value)
        tensor_list.append(particle_tensor_list)
    tensor_array=torch.tensor(tensor_list)
    return tensor_array

from concurrent.futures import ThreadPoolExecutor

def process_particle(particle):
    particle_tensor_list = []
    for dof_value in particle.dof_dic.values():
        if isinstance(dof_value, int):
            particle_tensor_list.append(dof_value)
        elif isinstance(dof_value, tuple):
            particle_tensor_list.extend(dof_value)
    return particle_tensor_list

def make_state_tensors(state):
    particle_keys = sorted(state.particle_dic.keys())  # Sort keys to ensure the order
    with ThreadPoolExecutor() as executor:
        tensor_list = list(executor.map(process_particle, 
                                        (state.particle_dic[key] for key in particle_keys)))
    return np.array(tensor_list)

def make_basis_tensors(states):
    with ThreadPoolExecutor() as executor:
        # Parallel processing of each state
        all_tensor_lists = list(executor.map(make_state_tensors, states))
        
    # Flatten the results: Combine tensors from all states into one large array
    return torch.tensor(np.array(all_tensor_lists))



###################NEED TO REWRITE THE HK_N CODE TO BE IN TENSOR FORM



def HK_N(state_list,pdic_n,U_N): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        U_count=0
        list1=list(state.particle_dic.values())
        list2=list(state.particle_dic.values())
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        p1count=0
        for p1 in list1:
            p1count+=1
            coeff=1
            for p2 in list2[p1count:]:
                initial=True
                for d in range(dof):#Hardcoded that dof=2!
                    initial=initial and (p1.dof_dic[d]==pdic_n[d](p2.dof_dic[d])[1])
                    coeff=coeff*pdic_n[d](p2.dof_dic[d])[0]
                # initial=initial and (p2.dof_dic[2]==pdic_n[2-1](p1.dof_dic[2])[1])
                # coeff=coeff*pdic_n[1-1](p1.dof_dic[1])[0]
                if initial:
                    U_count+=1
                    # list2.remove(p1)
                    # list2.remove(p2)
                    # list1.remove(p1)
                    # list1.remove(p2)
                    position1=(state_list.index(state),state_list.index(state))
                    temp_H[position1]=U_N*coeff+temp_H[position1]
        H1=H1+temp_H
    return H1

def find_matches_HKN(basis_tensor,result_tensor):
    N,n,d=basis_tensor.shape
    input_tensor=basis_tensor.clone()
    input_expanded = input_tensor.unsqueeze(2)  # Shape: (N, m, 1, d)
    input_expanded=input_expanded.repeat(1,1,n,1)
    input_expanded=input_expanded.transpose_(1,2)
    output_tensor=result_tensor.clone()

    unique_rows,duplicate_inverses,duplicate_counts=torch.unique(output_tensor,dim=0,return_counts=True,return_inverse=True)
    
    duplicate_rows = unique_rows[duplicate_counts > 1]
    duplicate_inverses_tensor=duplicate_inverses[duplicate_counts>1]
    duplicate_counts=duplicate_counts[duplicate_counts>1]

    return duplicate_rows,duplicate_inverses_tensor,duplicate_counts



    # diff_tensor=input_expanded-output_tensor
    # sum_row_diffs=torch.abs(diff_tensor).sum(dim=3)
    # print(sum_row_diffs.shape)
    # print(f'sum row diffs')
    # print(sum_row_diffs[:2])
    # print(torch.where(sum_row_diffs==0))
    # exit()
    # zeros=count_zeros(sum_row_diffs)
    # print(f'zeros shape: {zeros[1].shape}')
    # exit()
    # return None

def count_duplicates(matrix):
    N,n,_,d=matrix.shape
    #You need to first exclude any states that have more than three rows the same
    # repeated_tensor=matrix.unsqueeze(1).repeat(1,1,1,n,1)
    # sub_tensor=repeated_tensor-matrix.unsqueeze(3)
    #sub_tensor=repeated_tensor-original_matrix.unsqueeze(3)
    print(f'matrix shape: {matrix.shape}')
    print(f'unsq matrix shape: {matrix.unsqueeze(-2).shape}')
    
    repeated_tensor=matrix.unsqueeze(-2).repeat(1,1,1,n,1)
    sub_tensor=repeated_tensor-matrix.unsqueeze(-2)
    
    sum_tensor=torch.abs(sub_tensor).sum(dim=4)
    
    
    
    
    
    zero_tensor=((sum_tensor==0).sum(dim=3))/(2)#I think you need the two because of the reciprocity.
    zero_tensor[:,torch.arange(zero_tensor.shape[1]),torch.arange(zero_tensor.shape[2])]+=-1

    print(f'zero tensor shape: {zero_tensor.shape}')
    print(f'zero tensor:\n {zero_tensor[:2]}')
    exit()
    duplicate_tensor=zero_tensor.sum(dim=(1,2))
    print(f'duplicate tensor shape: {duplicate_tensor.shape}')
    print(f'duplicate tensor: {duplicate_tensor.sum()}')
    # print(f'zero tensor shape: {zero_tensor[:2]}')
    exit()

    return None

def count_duplicates_seq(matrix):
    U_counts = []
    N, n, _, _ = matrix.shape
    for i in range(N):
        U_counts_i=0
        for j in range(n):
            state = matrix[i, j]
            _, counts = torch.unique(state, dim=0, return_counts=True)
            duplicates = counts[counts > 1]
            U_counts_i+=(duplicates.sum() - duplicates.numel()).item() / 2
        U_counts.append(U_counts_i)
    return torch.tensor(U_counts)

def count_duplicates_vectorized(matrix):
    N, n, n_rows, d = matrix.shape
    result = torch.zeros((N, n), dtype=torch.float32, device=matrix.device)
    
    # Reshape to (N*n, n_rows, d) to process each n×d tensor separately
    reshaped = matrix.reshape(-1, n_rows, d)
    
    # Use torch.unique with return_inverse=True for each n×d tensor
    unique_flat = torch.unique(reshaped.reshape(-1, d), dim=0, return_inverse=True)
    unique_counts = torch.bincount(unique_flat[1]).reshape(-1, n_rows)
    
    # Count duplicates (elements appearing more than once)
    duplicates = unique_counts[unique_counts > 1]
    duplicate_counts = (duplicates.sum(dim=1) - duplicates.size(1)) / 2
    
    # Reshape result back to (N, n)
    result = duplicate_counts.reshape(N, n)
    
    return result
    
    

def HK_N_tensor(basis_state_list,basis_tensor,pdic_n,U_N): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    N,n,d=basis_tensor.shape
    input_state_tensor=basis_tensor.clone()
    #input_state_tensor=input_state_tensor.repeat(1,1,n,1)
    input_state_tensor_cheeky,_=pauli_action({0:p0,1:t0,2:p0,3:p0},basis_tensor)
    res_states,res_coeffs=pauli_action(pdic_n,basis_tensor)

    duplicate_matrix=count_duplicates_seq(res_states)

    duplicate_matrix=torch.complex(duplicate_matrix.float(),torch.zeros_like(duplicate_matrix.float()))

    HK_N=torch.zeros((N,N),dtype=torch.complex64)

    HK_N[torch.arange(N),torch.arange(N)]=U_N*duplicate_matrix[torch.arange(N)]



    return HK_N


def HK_rot(state_list,U_rot): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        down_spins=0
        for particle in state.particle_dic.keys():
            down_spins+=state.particle_dic[particle].dof_dic[dof-1]
        total_spins=len(state.particle_dic)
        up_spins=total_spins-down_spins
        coeff=up_spins*down_spins
        position1=(state_list.index(state),state_list.index(state))
        H1[position1]=U_rot*coeff+H1[position1]
    
    return H1

def HK_rot_tensor(basis_state_list,basis_tensor,U_rot):
    basis_copy=basis_tensor.clone()
    N,n,d=basis_tensor.shape
    total_spins=n
    up_spins=(basis_tensor[:,:,4]==0).sum(dim=1) #Because it's just 0 or 1!
    down_spins=(basis_tensor[:,:,4]==1).sum(dim=1) #Because it's just 0 or 1!
    coeff=up_spins*down_spins
    coeff=torch.complex(coeff.float(),torch.zeros_like(coeff.float()))
    HK_rot=torch.zeros((N,N),dtype=torch.complex64)
    HK_rot[torch.arange(N),torch.arange(N)]=U_rot*coeff

    # print(f'basis tensor sample: \n {basis_tensor[:2]}')
    # print(f'HK rot sample: \n {HK_rot.diagonal()[:2]}')
    
    return HK_rot



#Code for making templates
############################################################################

def make_matrices(basis_state_list,basis_tensor,pauli_dic,kfunction,variable_names,variable_factors,variable_functions,final_matrix_description):
    if ((qkx_tensor in pauli_dic.values()) or (qkx in pauli_dic.values()) or (qky in pauli_dic.values()) or (qky_tensor in pauli_dic.values())):
        print('qkx or qky in pauli_dic')
        matrix,_=tpp_from_tensor(state_list=basis_state_list,tensor_list=basis_tensor,pauli_dic=pauli_dic,prefactor=1,track_time=True)
        matrix=matrix/torch.sin(torch.tensor(theta)/2)
    else:
        matrix,_=tpp_from_tensor(state_list=basis_state_list,tensor_list=basis_tensor,pauli_dic=pauli_dic,prefactor=1,track_time=True)
    sparse_matrix=csr_matrix(matrix)
    matrix_object=saved_template_matrix(matrix=matrix,kfunction=kfunction,variable_functions=variable_functions,variable_names=variable_names,variable_factors=variable_factors,final_matrix_description=final_matrix_description)
    matrix_object.sparse_matrix=sparse_matrix
    #to save space
    save_space=True
    if save_space:
        matrix_object.matrix=None
    #for saving
    parameterdic={'particles':particle_no,'shells':shells_used,'center':center,'nonlayer':testnonlayer}#angle shouldn't matter

    return matrix_object

def make_matrices_U(state_list,basis_tensor,type,pdic_n,U,final_matrix_description):
    save_space=True
    if type=='HK_N':
        matrix=HK_N(state_list=state_list,pdic_n=pdic_n,U_N=U)
        nstring=''
        for key in sorted(list(pdic_n.keys())):
            nstring+=f'{pdic_n[key].__name__}'
        matrix_name='UHK_N_'+nstring
        print(f' matrix name: {matrix_name}')
        matrix_object=saved_template_matrix(matrix=matrix,kfunction=g0,variable_functions=[g00],variable_names=[matrix_name],variable_factors=[1],final_matrix_description=final_matrix_description)
        matrix_object.sparse_matrix=csr_matrix(matrix)
        if save_space:
            matrix_object.matrix=None
    elif type=='HK_rot':
        matrix=HK_rot(state_list=state_list,U_rot=U)
        matrix_object=saved_template_matrix(matrix=matrix,kfunction=g0,variable_functions=[g00],variable_names=['UHK_rot'],variable_factors=[1],final_matrix_description=final_matrix_description)
        matrix_object.sparse_matrix=csr_matrix(matrix)
        if save_space:
            matrix_object.matrix=None
    return matrix_object

def make_matrices_U_tensor(state_list,basis_tensor,type,pdic_n,U,final_matrix_description):
    save_space=True
    if type=='HK_N':
        matrix=HK_N_tensor(basis_state_list=state_list,pdic_n=pdic_n,U_N=U,basis_tensor=basis_tensor)
        nstring=''
        for key in sorted(list(pdic_n.keys())):
            nstring+=f'{pdic_n[key].__name__}'
        matrix_name='UHK_N_'+nstring
        print(f' matrix name: {matrix_name}')
        matrix_object=saved_template_matrix(matrix=matrix,kfunction=g0,variable_functions=[g00],variable_names=[matrix_name],variable_factors=[1],final_matrix_description=final_matrix_description)
        matrix_object.sparse_matrix=csr_matrix(matrix)
        if save_space:
            matrix_object.matrix=None
    elif type=='HK_rot':
        matrix=HK_rot_tensor(basis_state_list=state_list,basis_tensor=basis_tensor,U_rot=U)
        matrix_object=saved_template_matrix(matrix=matrix,kfunction=g0,variable_functions=[g00],variable_names=['UHK_rot'],variable_factors=[1],final_matrix_description=final_matrix_description)
        matrix_object.sparse_matrix=csr_matrix(matrix)
        if save_space:
            matrix_object.matrix=None
    return matrix_object

def make_each_matrix(term_list,basis_state_list,basis_tensor,dirname,matrix_name,type,term_number):
    os.makedirs(dirname, exist_ok=True)
    filename=f'{dirname}/{matrix_name}'

    for i in range(len(term_list)):
        print(f'i = {i}')
        start=time.time()
        if type=='nonint':
            test_matrix=make_matrices(basis_state_list=basis_state_list,basis_tensor=basis_tensor,pauli_dic=term_list[i][0],kfunction=term_list[i][1],
            variable_functions=term_list[i][2],variable_names=term_list[i][3],
            variable_factors=term_list[i][4],final_matrix_description='linear kx non-zero angle')

            
        elif (type=='HK_N' or type=='HK_rot'):
            test_matrix=make_matrices_U_tensor(state_list=basis_state_list,basis_tensor=basis_tensor,type=type,pdic_n=term_list[i][0],U=1,
                                        final_matrix_description='linear kx non-zero angle')
        
        end=time.time()
        print(f'time taken to make matrix {term_list[i]}: {end-start}')
        if len(term_list)>1:
            with open(filename+'_'+f'{i}'+'.dill', 'wb') as file:
                dill.dump(test_matrix, file)
                print(filename+'_'+f'{i}'+'.dill')
                #reconstructed_matrix=test_matrix.form_matrix()                
        else:
            print(filename+'_'+f'{term_number}'+'.dill')
            with open(filename+'_'+f'{term_number}'+'.dill', 'wb') as file:
                dill.dump(test_matrix, file)
        
        

        #reconstructed_matrix=test_matrix.form_matrix()
    

def construct_templates(dir_path,term_list_dic,term_number,basis_state_list,basis_tensor,make_all=True,make_int=True):
    k0=B#shouldn't matter
    #dir_path=f'{particle_no}particles_{shells_used}shells_center{center}_matrices_new/kindmatrices_particles{particle_no}_shells{shells_used}_center{center}'
    #dir_path=f'UHK_N_taux_matrix_particles{particle_no}_shells{shells_used}_center{center}'
    #print(dir_path)
    # for the cluster
    print('start make')
    
    print(f'make all {make_all}')
    
    if make_all:
        for term_key in term_list_dic.keys():
            term_type=term_list_dic[term_key][2]
            if make_int and (term_type=='HK_N' or term_type=='HK_rot'):
                make_each_matrix(term_list=term_list_dic[term_key][0],basis_state_list=basis_state_list,basis_tensor=basis_tensor,dirname=f'{dir_path}/{term_list_dic[term_key][1]}',matrix_name=term_list_dic[term_key][1],type=term_list_dic[term_key][2],term_number=None)
            elif term_type=='nonint':
                make_each_matrix(term_list=term_list_dic[term_key][0],basis_state_list=basis_state_list,basis_tensor=basis_tensor,dirname=f'{dir_path}/{term_list_dic[term_key][1]}',matrix_name=term_list_dic[term_key][1],type=term_list_dic[term_key][2],term_number=None)
            else:
                pass
    else:
        #term matching
        term_counts=[(len(term_list_dic[key][0]),key) for key in term_list_dic.keys()]
        counts=[term_counts[i][0] for i in range(len(term_counts))]
        sumcounts=np.array([np.sum(np.array(counts[:i])) for i in range(len(counts))])
        sumcounts=sumcounts-(term_number-1)
        print(sumcounts)
        termindex=np.max(np.where((sumcounts+np.abs(sumcounts))==0)[0])
        termkey=term_counts[termindex][1]
        termno=int(np.abs(sumcounts[termindex]))
        #end term matching
        make_each_matrix(term_list=[term_list_dic[termkey][0][termno]],basis_state_list=basis_state_list,basis_tensor=basis_tensor,dirname=f'{dir_path}/{term_list_dic[termkey][1]}',matrix_name=term_list_dic[termkey][1],type=term_list_dic[termkey][2],term_number=termno)


#loading templates

def load_matrices(filelist,sparse=True):
    first=True
    for matrix_file in filelist:
        #print(matrix_file)
        
        if first:
            with open(matrix_file,'rb') as f:
                combined_matrix_object=dill.load(f)
                print(type(combined_matrix_object))
                exit()
                combined_matrix=combined_matrix_object.form_matrix(sparse)
                del combined_matrix_object
            first=False
        else:
            with open(matrix_file,'rb') as f:
                temp_matrix_object=dill.load(f)
            temp_matrix=temp_matrix_object.form_matrix()
            #print(temp_matrix.shape)
            del temp_matrix_object
            combined_matrix=combined_matrix+temp_matrix
            gc.collect()#To remove the overwritten variable if not done already.

    return combined_matrix
    

def make_template_files(particle_no,basis_state_list,shell_count,target_terms,name,indiviudal,type):
    shells_particle=basis_state_list#generate_shell_basis_gamma(shell_count=shell_count,q_vecs=tqs,number_of_particles=particle_no,nonlayer=testnonlayer,center=center)
    k0=B
    cluster_arg=int(sys.argv[1])-1
    dir_path=f'{particle_no}particles_{shells_used}shells_center{center}_matrices/{name}_particles{particle_no}_shells{shell_count}_center{center}'
    
    # for the cluster
    if indiviudal:
        make_each_matrix(term_list=[target_terms[cluster_arg]],state_list=shells_particle,dirname=dir_path,matrix_name=name,type=type)
    else:
        make_each_matrix(term_list=target_terms,state_list=shells_particle,dirname=dir_path,matrix_name=name,type=type)

if __name__ == "__main__":
    shells=shells_used
    particles=particle_no
    center=center
    shell_basis_dicts=generate_shell_basis_gamma(shell_count=shells,q_vecs=tqs,number_of_particles=particles,nonlayer=testnonlayer,center=center)
    print(f'Number of basis states: {len(shell_basis_dicts)}')
    test_tensor_states=make_basis_tensors(shell_basis_dicts)
    

    # test_new,_=tpp_from_tensor(state_list=shell_basis_dicts,tensor_list=test_tensor_states,pauli_dic={0:px,1:t3_plus_tensor,2:p0,3:p0},prefactor=1,track_time=True)
    # tpp_old=tpp(shell_basis_dicts,{0:px,1:t3_plus,2:p0,3:p0},1)
    
    # print(f'q0 block: {test_new[:4,12:16]}')
    # #print(f'tensor new {np.nonzero(test_new)}')
    # #print(f'all zeros new {np.allclose(test_new,0)}')
    
    # print(f'q0 block old: {tpp_old[:4,12:16]}')

    # # print(f' q1 block: \n {test_new[12:16,:4]}')
    # # print(f'q1 block old: \n  {tpp_old[12:16,12:16]}')

    # print(f'same? {np.allclose(test_new,tpp_old)}')
    # print(f'type tpp tensor {type(test_new)}')
    # exit()

    # teststate=shell_basis_dicts[0]
    # print(f'eyepreservation for state: {eyepreservation(teststate)}')
    # print(f'list of state dics:')
    # print(vars(teststate))
    # print(teststate.particle_dic.keys())
    # print([vars(teststate.particle_dic[x]) for x in teststate.particle_dic.keys()])

    dir_path=f"/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/tensor/test/{particles}particles_{shells}shells_center{center}_matrices"
    #term_list_dic_int={key:term_list_dic[key] for key in term_list_dic.keys() if term_list_dic[key][2] in ['HK_N','HK_rot']}
    sub_dir=config['sub_dir']
    term_number=int(config['term_number'])
    term_list_dic_term={key:term_list_dic[key] for key in term_list_dic.keys() if term_list_dic[key][1] in [sub_dir]}

    #mp.set_start_method('spawn')
    construct_templates(dir_path=dir_path,term_list_dic=term_list_dic_term,term_number=term_number,basis_state_list=shell_basis_dicts,basis_tensor=test_tensor_states,make_all=False,make_int=True)
    exit()

    # for i in range(16):
    #     print(f'qdjust index {i}')
    #     dir0='tun'
    #     particles1=particle_no
    #     saved_mat_path=f"/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/tensor/{particles1}particles_{shells_used}shells_center{center}_matrices/{dir0}/{dir0}_{i}.dill"
    #     saved_mat_path_old=f"/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/matrix_templates/{particles1}particles_2shells_centerK_matrices/ham_terms/{dir0}/{dir0}_{i}.dill"
    #     # print(f'q3 kx {v*qvecs[1][0]}')
        
    #     # print(f'Checking matrix {dir0}, {i}')
    #     test_matrix_tensor=load_matrices([saved_mat_path])
        
        



        
        
    #     #print(f'test matrix shape {test_matrix_tensor.shape}')
    #     #print(f'sample test matrix new \n {test_matrix_tensor[12:16,12:16]}')
    #     #print(f'sample new /npsin \n {test_matrix_tensor[12:16,12:16]/np.sin(theta/2)}')
    #     test_matrix_old=load_matrices([saved_mat_path_old],sparse=False)
    #     #print(f'sample test matrix old \n {test_matrix_old[12:16,12:16]}')
    #     print(f'same? {np.allclose(test_matrix_tensor.todense(),test_matrix_old)}')
    # exit()

    # with open(filename,'rb') as f:
    #     test_matrix_object=dill.load(f)
    # hkpath="/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/tensor/4particles_2shells_centerK_matrices/HK_N_taux/HK_N_taux_None.dill"
    # hktest=load_matrices([hkpath],sparse=True)
    # print(type(hktest))
    # exit()

    # randomk=np.pi*np.array([np.random.random(),np.random.random()])
    # HkA=load_templates.gen_Hk2_tensor(kx=A[0],ky=A[1],particles_used=2,sparse=True)
    # print(HkA.shape)
    
    # #print(f'sample {HkA[:4,:4]}')

    # HKA2=load_templates.gen_Hk2(kx=A[0],ky=A[1],particles_used=2,sparse=False)

    # print(HKA2.shape)
    # #print(f'sample old {HKA2[:4,:4]}')
    # print(f'matrices same? {np.allclose(HkA.todense(),HKA2)}')

    # size_bytes = (HkA.data.nbytes + 
    #         HkA.indptr.nbytes + 
    #         HkA.indices.nbytes)
    
    # print(f'Sparse size: {size_bytes/1024/1024} MB')
    # print(f'Dense size: {HKA2.nbytes/1024/1024} MB')
    # print(type(HkA))
    # exit()

    # start=time.time()
    # #eigval_sparse,eigvec_sparse= eigs(HkA, k=16, which='SR', sigma=None)
    # k=8
    # eigval_sparse,eigvec_sparse= eigsh(HkA, k=k, which='SA', sigma=None)
    # end=time.time()
    # print(f'Sparse time: {end-start}')
    # start=time.time()
    # eigval_dense,eigvec_dense=np.linalg.eigh(HKA2)
    # end=time.time()
    # print(f'Dense time: {end-start}')
    # print(f'MSE eigval errors: {np.abs((np.sort(eigval_sparse)-eigval_dense[:k]))}')
    # print(f'Exact Eigvals {eigval_dense[:k]}')
    # print(f'Approx Eigvals {eigval_sparse}')
    # exit()
    # print(f'MSE Eigvecs? {np.mean(np.sqrt(np.sum((eigvec_sparse-eigvec_dense[:,:k])**2,axis=1)))}')
    
    #Testing HKN tensor terms:

    
    #test_HKN=HK_N_tensor(shell_basis_dicts,test_tensor_states,{0:p0,1:t0,2:p0,3:px},1)
    #test_HKrot=HK_rot_tensor(shell_basis_dicts,test_tensor_states,1)
    #test_HKN=HK_N_tensor(shell_basis_dicts,test_tensor_states,{0:p0,1:t0,2:p0,3:px},1)

    # def check_HK_quick(HK_term,samples=5):
    #     print(f'is HK diagonal? {np.allclose(HK_term,np.diag(np.diag(HK_term)))}')
    #     for _ in range(samples):
    #         random_sample=np.random.randint(0,len(shell_basis_dicts))
    #         print(f'Input state: {eyepreservation(shell_basis_dicts[random_sample])}, HK coeff: {HK_term[random_sample,random_sample]}')
    #     print(f' {samples} non zero terms:')
    #     non_zero_terms=torch.nonzero(HK_term)
    #     for i in range(samples):
    #         random_sample=np.random.randint(0,len(non_zero_terms))
    #         print(f'non zero term {i}, state: {eyepreservation(shell_basis_dicts[non_zero_terms[i][0]])}, HK_coeff {HK_term[non_zero_terms[i][0],non_zero_terms[i][1]]}')

    
    

