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

from base_functions import *


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

def px(sigma):
    sigma_pair=(1,(sigma+1)%2)
    return sigma_pair
def py(sigma):
    sigma_pair=((1j)*((-1)**(sigma)),(sigma+1)%2)
    return sigma_pair
def pz(sigma):
    sigma_pair=((-1)**(sigma),sigma)
    return sigma_pair
def p0(sigma):
    return (1,sigma)
def pexpz(sigma):
    sigma_exp=(-1)**(sigma)
    return (np.exp(1j*np.pi*sigma_exp/4),sigma)

def pexpz3(sigma):
    sigma_exp=(-1)**(sigma)
    return (np.exp(1j*2*np.pi*sigma_exp/3),sigma)

def pexpz6(sigma):
    sigma_exp=(-1)**(sigma)
    return (np.exp(1j*2*np.pi*sigma_exp/6),sigma)

def qkx(sigma):
    q=sigma[0]*qvecs[0]+sigma[1]*qvecs[1]
    return (q[0],sigma)
def qky(sigma):
    q=sigma[0]*qvecs[0]+sigma[1]*qvecs[1]
    return (q[1],sigma)

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
        


#Now let's build up the tunnelling matrices
def t1_plus(sigma):
    newsigma=(sigma[0]-1,sigma[1]-1)#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)
#Now let's build up the tunnelling matrices
def t1_minus(sigma):
    newsigma=(sigma[0]+1,sigma[1]+1)#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)

def t2_plus(sigma):
    newsigma=(sigma[0]+1,sigma[1])#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)
def t2_minus(sigma):
    newsigma=(sigma[0]-1,sigma[1])#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)

def t3_plus(sigma):
    newsigma=(sigma[0],sigma[1]+1)#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)
def t3_minus(sigma):
    newsigma=(sigma[0],sigma[1]-1)#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)

    


#sigma is here understood to be the pair of q states.
def t0(sigma):
    return (1,sigma)

def tz(sigma):
    return ((2*sigma[0]+3*sigma[1])%6,sigma)

def tqx(sigma,qs=qvecs):
    q=sigma[0]*qs[0]+sigma[1]*qs[1]
    qx=q[0]
    
    return (-qx,sigma)#Note - sign because it's k-Q

def tqy(sigma,qs=qvecs):
    q=sigma[0]*qs[0]+sigma[1]*qs[1]
    qy=q[1]
    
    return (-qy,sigma)#Note - sign because it's k-Q

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




if __name__ == "__main__":
    shells=2
    particles=2
    center='K'
    shell_basis_dicts=generate_shell_basis_gamma(shell_count=shells,q_vecs=tqs,number_of_particles=particles,nonlayer=testnonlayer,center=center)
    print(f'Number of basis states: {len(shell_basis_dicts)}')

    teststate=shell_basis_dicts[0]
    print(f'eyepreservation for state: {eyepreservation(teststate)}')
    print(f'list of state dics:')
    print(vars(teststate))
    print(teststate.particle_dic.keys())
    print([vars(teststate.particle_dic[x]) for x in teststate.particle_dic.keys()])

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
        tensor_array=np.array(tensor_list)
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
        return np.array(all_tensor_lists)
                    
        
        # for particle_key in teststate.particle_dic.keys():
        #     for dof_key in teststate.particle_dic[particle_key].dof_dic.keys():
                        
        
        # state_object
        # components_dict=all_elements_component(state_list)
        # tensor_component_list=[]

        # for component in components_dict.keys():
        #     if type(components_dict[component][0])==int:
        #         tensor_component_list.append()
        #     elif type(components_dict[component][0])==tuple:
        #         tensor_component_list.append(len(components_dict[component]))
        #         tuple_length=len(components_dict[component][0])


        
        
    test_tensor_states=make_basis_tensors(shell_basis_dicts)
    print(test_tensor_states.shape)
    #all_elements_component(shell_basis_dicts)
    
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

    def tpp_from_tensor(state_list,tensor_list,pauli_dic,prefactor):
        N,n,d=tensor_list.shape
        px_tensor=torch.zeros((N,n,n,d),dtype=torch.int)
        
        for particle_ind in range(n):
            #would just add other pauli actions here if need be.
            px_tensor[:,particle_ind,particle_ind,4]=1
        print(px_tensor.count_nonzero())
        
        
        
        tensor_list_torch=torch.tensor(tensor_list)


        res_tensor=tensor_list_torch.clone()
        
        res_expanded = res_tensor.unsqueeze(2)  # Shape: (N, m, 1, d)
        
        # Repeat A_expanded along the new dimension to get shape (N, m, m, d)
        input_repeated = res_expanded.repeat(1, 1, n, 1)
        input_repeated.transpose_(1,2)

        
        res_repeated = input_repeated.clone()
        

        res_repeated[:,:,:,4]=(input_repeated[:,:,:,4]+px_tensor[:,:,:,4])%2

        def find_duplicate_indices(A):
            # A: N x m x m x d tensor
            N, m, _, d = A.shape
            
            # Reshape A into a 2D matrix where each row is one of the mxd rows
            A_flat = A.reshape(N * m * m, d)  # Now a (N*m*m) x d tensor
            
            # Find unique rows and get the inverse indices that map to unique rows
            unique_rows, inverse_indices = torch.unique(A_flat, dim=0, return_inverse=True)
            
            # Count occurrences of each unique row
            row_counts = torch.bincount(inverse_indices)
            
            # Find unique rows that occur more than once (i.e., duplicates)
            duplicate_unique_indices = torch.nonzero(row_counts > 1, as_tuple=True)[0]
            
            # Check which rows in the flattened tensor correspond to duplicated unique rows
            duplicate_mask = torch.isin(inverse_indices, duplicate_unique_indices)
            
            # Get the original indices (flattened index) where duplicates occur
            duplicate_indices_flat = torch.nonzero(duplicate_mask, as_tuple=True)[0]
            
            # Convert flattened indices back to Nxm indices (N and m)
            duplicate_indices_Nm = duplicate_indices_flat // (m * m), (duplicate_indices_flat % (m * m)) // m

            return duplicate_indices_Nm

        #I think this failed...        
        # duplicate_indices = find_duplicate_indices(res_repeated)
        # print(len(duplicate_indices[0]))
        # exit()
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
            for state_index in range(N):
                for particle_index in range(n):
                    state_tensor=result_tensor[state_index,particle_index]
                    unique_rows,counts=torch.unique(state_tensor,dim=0,return_counts=True)
                    duplicate_rows = unique_rows[counts > 1]
                    if len(duplicate_rows)>0:
                        #Could instead append an mxd tensor of all zeros and then exclude those.
                        continue
                    else:
                        basis_expanded = basis_tensor.unsqueeze(1)  # Shape becomes (N, 1, m, d)
                        state_tensor=state_tensor.unsqueeze(0).unsqueeze(2)
                        row_diffs=state_tensor-basis_expanded
                        row_diffs=row_diffs.transpose(1,2)

                        sum_row_diffs=torch.abs(row_diffs).sum(dim=3)
                        zeros=count_zeros(sum_row_diffs)
                        matched_basis_indices=[z.item() for z in zeros[0]]
                        matched_indices.append((state_index,particle_index,matched_basis_indices))

                        
                        # print(f'result state tensor')
                        # print(state_tensor)
                        # print(f'comparison basis tensor:')
                        # print(basis_expanded[0])
                        # print(f'difference tensor:')
                        # print(row_diffs[0])
                        # sum_row_diffs=torch.abs(row_diffs).sum(dim=3)
                        # print(f'sum row diffs')
                        # print(sum_row_diffs[0])
                        # print(f'zero counts')
                        # zeros=count_zeros(sum_row_diffs)
                        # print(zeros)
                        # print(f'inital result state')
                        # print(state_tensor)
                        # print(f'claimed matching basis state')
                        # print(basis_tensor[zeros[0][0]])
                        # exit()
                        
                        
                        

                    

                    
            return matched_indices
        
        test=find_matches(res_repeated,tensor_list_torch)
        input_tensor_index=32
        print(f'starting state:')
        print(tensor_list_torch[test[input_tensor_index][0]])
        print(f'matching state:,particle acted on: {test[input_tensor_index][1]}')
        print(tensor_list_torch[test[input_tensor_index][2]])
        #print(len(test))
        exit()
        


        
        def seq_diff(A,B):
            N,n,_,d=A.shape
            res_tensor_list=[]
            for state_index in range(N):
                state_mat=A[state_index] #nxd state

                #Now need to find all the differences
                
                state_mat_expanded = state_mat.unsqueeze(0).expand(N, -1, -1)  # Nxmxd tensor

                # Compute the differences
                differences = A.unsqueeze(2) - state_mat_expanded.unsqueeze(1)  # Nxmxmx d tensor

                # Compute the absolute differences and sum along the last dimension
                abs_differences = torch.abs(differences)  # Nxmxmx d tensor
                sum_abs_differences = abs_differences.sum(dim=3)  # Nxmxm tensor
                res_tensor_list.append(sum_abs_differences)
            res_tensor_list=torch.stack(res_tensor_list,dim=0)

            return res_tensor_list
        






        def seq_diff2(A,B):
            N,n,d=A.shape
            res_tensor_list=[]
            for state_index in range(N):
                state_mat=A[state_index] #nxd state
                state_matches_list=[]
                for particle_index in range(n):
                    B_comp=B[:,particle_index,:,:] #Nxd state
                    print(f'b comp')
                    print(B_comp)
                    same_rows = (B_comp.unsqueeze(1) == B_comp.unsqueeze(0)).all(-1)
                    row_pairs = torch.nonzero(same_rows.triu(1), as_tuple=True)
                    print(row_pairs)
                    if row_pairs[0].shape[0] != 0:
                        continue
                    else:
                        print(f'the else')
                        #Now need to find all the differences
                        
                        # B_comp = B_comp.unsqueeze(1)  # Nx1xd tensor
                        # B_comp = B_comp.repeat(1,n,1,1)  # Nxmxd tensor
                        B_comp_1 = B_comp.unsqueeze(1)  # Shape becomes (N, 1, m, d)
                        state_mat=state_mat.unsqueeze(0).unsqueeze(2)
                        row_diffs=B_comp_1-state_mat
                        row_diffs=row_diffs.transpose(1,2)
                        print(row_diffs.shape)
                        exit()
                        print(row_diffs[0])
                        
                        print(B_comp_1.shape)
                        
                        print(f'initial state')
                        print(B_comp_1[0][0])
                        print(B_comp_1[1][0])
                        print(f'state mat subtracting off')
                        print(state_mat[0])
                        print(f'difference')
                        print(row_diffs[0])
                        print(f'sum of abs diff:')
                        print(torch.abs(row_diffs[0]).sum(dim=2))
                        exit()
                        # B_comp_2 = B_comp.unsqueeze(2)  # Shape becomes (N, m, 1, d)
                        
                        # # Step 2: Perform outer product
                        # print(B_comp.shape)
                        # exit()
                        # B_comp = B_comp_1 - B_comp_2  # Broadcasting creates (N, m, m, d)
                        # print(B_comp[0].shape)
                        # print(B_comp[0])
                        # print(state_mat)
                        
                        # exit()
                        
                        # Compute the differences
                        #differences = B_comp-state_mat  # Nxmxmxd tensor
                        
                        # Compute the absolute differences and sum along the last dimension
                        abs_differences = torch.abs(row_diffs)  # Nxmxmx d tensor
                        print(abs_differences.shape)
                        
                        sum_abs_differences = abs_differences.sum(dim=3)  # Nxmxm tensor
                        print(sum_abs_differences.shape)
                        
                        #Instead of keeping the whole thing, at this point you may as well identify which of the N indices you map to return an n tensor

                        
                        zeros=count_zeros(sum_abs_differences)
                        print(zeros[0])
                        exit()
                        print(zeros[0][0])

                        print(sum_abs_differences[zeros[0]])
                        print(state_mat)
                        print(B_comp[zeros[0]][2])

                        exit()
                        print(f'len zeros {len(zeros)}')
                        print(f'A first entry {A(zeros[0][0])}')
                        exit()
                        
                        
                        state_matches_list.append(sum_abs_differences)
                        #state_matches_tensor=torch.stack(state_matches_list,dim=1)
            
                res_tensor_list.append(state_matches_list)
            #res_tensor_list=torch.stack(res_tensor_list,dim=0)

            
            exit()

            return res_tensor_list

                


        test2=find_matching_indices(A=tensor_list_torch,B=res_repeated)
        print(f'first function {test2}')
        exit()

        test_seq=seq_diff2(tensor_list_torch,res_repeated)
        print(test_seq.shape)
        print(test_seq[-1][-2])
        print(len(count_zeros(test_seq)[0]))
        exit()
        #test=compute_all_pairs(tensor_list_torch,res_tensor)
        # print(tensor_list_torch[:2])
        # print(res_tensor[:2])

        # diff_test=diff_all_pairs(tensor_list_torch,res_tensor)

        # print(diff_test.shape)

        exit()
        
        
        print(test[0][1])
        exit()
        print(test.shape)
        print(torch.linalg.det(test))
        print(tensor_list_torch[-1])
        print(res_tensor[1])
        print(test[-1][1])
        exit()

        print(f'input state: {tensor_list[0]}')
        print(f'res state: {res_tensor[0]}')

        print(type(tensor_list_torch))
        print(type(res_tensor))

        #print(f'first few dets:')

        

        multp=torch.bmm(tensor_list_torch, res_tensor.transpose(2,1))
        multp_sqrt=torch.sqrt(multp)
        print(multp_sqrt.shape)
        print(multp_sqrt[-1])

        
        

        exit()

        print(f'dets tensor shape:{dets_tensor.shape}')
        print(f'are close shape: {are_close.shape}')

        exit()
        matching_indices=find_mapped_tensor(tensor_list_torch,tensor_list_torch)
        print(f'matching indices shape first function {matching_indices.shape}')
        matching_indices=find_permutation_equivalents(res_tensor,tensor_list_torch)
        print(f'matching indices shape second function {matching_indices.shape}')
        exit()
        print(matching_indices[0])
        print(f'tensor 1:\n {tensor_list[matching_indices[1][0]]}')
        print(f'tensor 2:\n {tensor_list[matching_indices[1][1]]}')
        exit()
        
        return None
    

    tpp_from_tensor(shell_basis_dicts,test_tensor_states,{0:p0,1:t0,2:px,3:p0},1)
    exit()
        
    
    

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
        
    tpp_func=tpp
    tpp_func_kwargs_dict={'pauli_dic':{0:p0,1:t0,2:px,3:p0},'prefactor':1}
    particles=np.arange(1,5)
    construction_times,frac_nonzero,memory=tpp_performance_metrics(tpp_func,tpp_func_kwargs_dict,particles)
    print(construction_times,frac_nonzero,memory)
    
    fig=make_subplots(rows=2,cols=2,specs=[[{"colspan": 2}, None], [{}, {}]],subplot_titles=['Construction times','Fraction of non-zero elements','Memory'])
    fig.add_trace(go.Scatter(x=particles,y=construction_times,mode='markers',name='Construction times'),row=1,col=1)
    fig.update_yaxes(title_text='Time (s)',type='log',row=1,col=1)

    fig.add_trace(go.Scatter(x=particles,y=frac_nonzero,mode='markers',name='Fraction of non-zero elements'),row=2,col=1)
    fig.update_yaxes(title_text='Fraction of non-zero elements',row=2,col=1)

    fig.add_trace(go.Scatter(x=particles,y=memory,mode='markers',name='Memory (MB)'),row=2,col=2)
    fig.update_yaxes(title_text='Memory (MB)',row=2,col=2)

    fig.update_xaxes(title_text='Number of particles')

    fig.update_layout(title_text='Performance metrics for tpp',showlegend=False)

    fig.show()



