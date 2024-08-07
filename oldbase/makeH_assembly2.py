from os import stat
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
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
from variables_cluster2 import *
import pickle
import multiprocessing
from multiprocessing import Pool
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import inspect
import gc
import dill


#TO DO:
#1. For larger Hamiltonians you're generate H algorithm is going to take a long time. 
# So you can improve it by only generating the positions that need to be updated with various k's at each location
# and then updating those locations instead of reconstructing the matrix each time. 
# That way you only need to generate the matrix once, and then populating it is easy.

######################################################
######################################################GLOBAL VARIABLES
dof=4
testpdd={'k':4,'sublattice':2,'spin':2}
kd2=np.array([0,4*np.pi/3])
k1=np.array([0,-1])
k2=np.array([np.sqrt(3)/2,1/2])
k3=np.array([-np.sqrt(3)/2,1/2])

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


        
class saved_template_matrix():
    def __init__(self,matrix,kfunction,variable_names,variable_factors,variable_functions,final_matrix_description):
        self.matrix=matrix
        self.kfunction=kfunction#the g0 the gkx and so on
        self.variable_names=variable_names#The things that go inside the functions
        self.variable_factors=variable_factors #i.e if you divide the variable by 2...
        self.variable_functions=variable_functions
        self.final_matrix_description=final_matrix_description
        
    def form_matrix(self):
        #The idea here is that this function will allow me to construct a matrix for arbitrary parameters once I have the template matrix saved
        coeff=1
        for i in range(len(self.variable_functions)):
            print(self.variable_names,[globals()[x] for x in self.variable_names],self.variable_factors)
            arg=globals()[self.variable_names[i]]*self.variable_factors[i]
            coeff=coeff*self.variable_functions[i](arg)

        return self.matrix*coeff
    
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


# tqs=[np.array([-1,-1]),np.array([1,0]),np.array([0,1])]
# tp=generate_layer_q_pairs(shell_count=3,q_vecs=tqs)
# for i in tp:
#     print(i)
# exit()

testnonlayer2={'sublattice':2,'spin':2}
def generate_shell_basis(shell_count,q_vecs,number_of_particles,nonlayer):
    basis_list=[]
    particle_dic={}
    dof_dic={}
    layer_q_pairs=flatten(generate_layer_q_pairs(shell_count=shell_count,q_vecs=q_vecs))#function outputs nested lists for shells
    nonlayerdofs=[range(nonlayer[x]) for x in nonlayer.keys()]
    single_particle_dofs=list(itertools.product(layer_q_pairs,nonlayerdofs[0]))
    single_particle_dofs=list(set(single_particle_dofs))
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



# basis1=generate_shell_basis(shell_count=2,q_vecs=tqs,number_of_particles=1,nonlayer=testnonlayer)
# basis2=generate_shell_basis(shell_count=2,q_vecs=tqs,number_of_particles=2,nonlayer=testnonlayer)
# # for g in tg:
# #     print(eyepreservation(g))
# print(len(basis1))
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


# tqs=[np.array([-1,-1]),np.array([1,0]),np.array([0,1])]
# tp=generate_layer_q_pairs(shell_count=3,q_vecs=tqs)
# for i in tp:
#     print(i)
# exit()

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



def generate_shell_basis_gamma_tp(shell_count,q_vecs,number_of_particles,nonlayer,center):
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

    
    initial_list=list(itertools.product(single_particle_dofs,repeat=number_of_particles))
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


def tpp_gpt0(state_list, pauli_dic, prefactor):
    size = len(state_list)
    H1 = np.zeros((size, size), dtype=complex)
    state_to_index = {i: state for i, state in enumerate(state_list)}

    if len(pauli_dic.keys()) != len(state_list[0].particle_dic[1].dof_dic.keys()):
        print("Need to give a pauli for each particle!")
        print(pauli_dic.keys())
        return NotImplemented

    for state in state_list:
        result_list = []
        for k, particle in state.particle_dic.items():
            coeff = 1
            dof_dic_temp = {}
            for l, pauli_op in pauli_dic.items():
                coef, new_dof = pauli_op(particle.dof_dic[l])
                dof_dic_temp[l] = new_dof
                coeff *= coef
            new_particle = particle(dof_dic=dof_dic_temp)
            particle_dic_temp1 = {k: new_particle, **{j: p for j, p in state.particle_dic.items() if j != k}}
            new_state = basis(particle_dic=particle_dic_temp1)
            result_list.append((coeff, new_state))

        for coeff, new_state in result_list:
            for i in state_list:
                if basis.eqnoorder(i, new_state):
                    swapcount = basis.swaps(new_state, i)
                    position1 = state_to_index[i]
                    position2 = state_to_index[state]
                    H1[position1, position2] += ((-1) ** swapcount) * prefactor * coeff

    return H1

def compute_for_state(state, state_list, pauli_dic, prefactor):
    # Your refactored computation here, returning necessary parts of H1
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    particle_dic_temp1={}#Idea is to constantly overwrite
    dof_dic_temp={}
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
        result=(coeff,new_state)
        result_list.append(result)

    for result in result_list:
        for i in state_list:
            if basis.eqnoorder(i,result[1]):
                temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
                swapcount=basis.swaps(result[1],i)
                position1=(state_list.index(i),state_list.index(state))
                temp_H[position1]=((-1)**swapcount)*prefactor*result[0]
                H1=H1+temp_H

    return H1

def tpp_parallel(state_list, pauli_dic, prefactor):
    H1 = np.zeros((len(state_list), len(state_list)), dtype=complex)
    # Prepare arguments for each process (if necessary)
    args = [(state, state_list, pauli_dic, prefactor) for state in state_list]

    with Pool(processes=2) as pool:  # Adjust the number of processes as needed
        results = pool.starmap(compute_for_state, args)

    # Aggregate results into H1
    for result in results:
        H1=H1+result

    return H1

def tpp_k(state_list,pauli_dic,prefactor):
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    particle_dic_temp1={}#Idea is to constantly overwrite
    dof_dic_temp={}
    if len(pauli_dic.keys())!=dof:
        print("Need to give a pauli for each particle!")
        print(pauli_dic.keys())
        return NotImplemented
    else:
        for state in state_list:
            result_list=[]
            for k in state.particle_dic.keys():
                coeff=1
                qval=state.particle_dic[k].dof_dic[1][0]*qvecs[0]+state.particle_dic[k].dof_dic[1][1]*qvecs[1]
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
                result=(coeff,new_state,qval)
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

def get_theta(k):
    zerosdic={(0,1):np.pi/2,(0,-1):-np.pi/2,(1,0):0,(-1,0):np.pi,(0,0):0}
    signtuple=(np.sign(k)[0],np.sign(k)[1])
    if signtuple in zerosdic.keys():
        return zerosdic[signtuple]
    else:
        calc=np.arctan(k[1]/k[0])
        map_dic={(1,1):calc,(1,-1):calc,(-1,1):np.pi+calc,(-1,-1):np.pi+calc}
        return map_dic[signtuple]



def diagtpp(state_list,theta,k0,layer_pauli_dic,prefactor):
    ktheta=np.sqrt(np.vdot(kd,kd))*np.sin(theta/2)
    q1=2*ktheta*k1/(np.sqrt(np.vdot(k1,k1)))
    q2=2*ktheta*k2/(np.sqrt(np.vdot(k2,k2)))
    q3=2*ktheta*k3/(np.sqrt(np.vdot(k3,k3)))
    kdic={0:k0,1:k0+q1,2:k0+q2,3:k0+q3}
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    particle_dic_temp1={}#Idea is to constantly overwrite
    dof_dic_temp={}
    if len(layer_pauli_dic.keys())!=len(list(testpdd.values())):
        print("Need to give a pauli for each particle!")
        print(layer_pauli_dic.keys())
        return NotImplemented
    else:
        for state in state_list:
            result_list=[]
            for k in state.particle_dic.keys():
                coeff=1
                dof_dic_temp[0]=layer_pauli_dic[0](state.particle_dic[k].dof_dic[0])[1]
                coeff=coeff*layer_pauli_dic[0](state.particle_dic[k].dof_dic[0])[0]
                temp_k=kdic[state.particle_dic[k].dof_dic[0]]
                temp_thetak=get_theta(k=temp_k)-theta/2
                coeff_dic={px:np.cos(temp_thetak),py:np.sin(temp_thetak)}
                #sublattice and spin part
                for l in range(1,len(layer_pauli_dic.keys())):
                    dof_dic_temp[l]=layer_pauli_dic[l](state.particle_dic[k].dof_dic[l])[1]
                    coeff=coeff*layer_pauli_dic[l](state.particle_dic[k].dof_dic[l])[0]
                coeff=coeff_dic[layer_pauli_dic[1]]*np.sqrt(np.vdot(temp_k,temp_k))
                particlen=particle(dof_dic=dict(dof_dic_temp))
                # print(vars(particlen))
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

def diagtpp2(state_list,theta,layer_pauli_dic,prefactor):

    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    particle_dic_temp1={}#Idea is to constantly overwrite
    dof_dic_temp={}
    if len(layer_pauli_dic.keys())!=len(list(testpdd.values())):
        print("Need to give a pauli for each particle!")
        print(layer_pauli_dic.keys())
        return NotImplemented
    else:
        for state in state_list:
            result_list=[]
            for k in state.particle_dic.keys():
                coeff=1
                dof_dic_temp[0]=layer_pauli_dic[0](state.particle_dic[k].dof_dic[0])[1]
                coeff=coeff*layer_pauli_dic[0](state.particle_dic[k].dof_dic[0])[0]
                temp_k=kdic[state.particle_dic[k].dof_dic[0]]
                temp_thetak=get_theta(k=temp_k)-theta/2
                coeff_dic={px:np.cos(temp_thetak),py:np.sin(temp_thetak)}
                #sublattice and spin part
                for l in range(1,len(layer_pauli_dic.keys())):
                    dof_dic_temp[l]=layer_pauli_dic[l](state.particle_dic[k].dof_dic[l])[1]
                    coeff=coeff*layer_pauli_dic[l](state.particle_dic[k].dof_dic[l])[0]
                coeff=coeff_dic[layer_pauli_dic[1]]*np.sqrt(np.vdot(temp_k,temp_k))
                particlen=particle(dof_dic=dict(dof_dic_temp))
                # print(vars(particlen))
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





def inspect_elements(matrix,state_list,state_number):
    if isinstance(state_number,np.ndarray)==False:
        states_considered=state_list
    else:
        states_considered=list(np.array(state_list)[state_number])

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
                print(f"Initial state: {eyepreservation(i)}, Output state: {eyepreservation(state_list[j])}, Amplitude {reswf[j]}")#, c1: {coeff1},c2:{coeff2}
        
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




#Diagonal term for new basis convention

def gkx(kx,ky):
    return kx
def gky(kx,ky):
    return ky
def g0(kx,ky):
    return 1
def g00(x):
    return x
def gw(w):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * g0(kx, ky)
    return multiplied_function
def gkxw(w):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * gkx(kx, ky)
    return multiplied_function
def gkyw(w):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * gky(kx, ky)
    return multiplied_function

def gcos(w,theta):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * np.cos(theta)* g0(kx, ky)
    return multiplied_function
def gsin(w,theta):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * np.sin(theta)* g0(kx, ky)
    return multiplied_function
def gtx(phi):
    return np.cos(phi)
def gty(phi):
    return np.sin(phi)



diag_term=({0:p0,1:t0,2:px,3:p0},gkx)
linear_terms=[({0:p0,1:t0,2:px,3:p0},gkx),({0:p0,1:t0,2:py,3:p0},gky),({0:p0,1:tqx,2:px,3:p0},g0),({0:p0,1:tqy,2:py,3:p0},g0)]
# tun_terms=[({0:px,1:t1,2:p0,3:p0},1),({0:px,1:t1,2:px,3:p0},gtx(phi=0)),({0:px,1:t1,2:py,3:p0},gty(phi=0)),
#            ({0:px,1:t2,2:p0,3:p0},1),({0:px,1:t2,2:px,3:p0},gtx(phi=phi)),({0:px,1:t2,2:py,3:p0},gty(phi=phi)),
#            ({0:px,1:t3,2:p0,3:p0},1),({0:px,1:t2,2:px,3:p0},gtx(phi=-phi)),({0:px,1:t3,2:py,3:p0},gty(phi=-phi))]#
def h_linear_diag(km,v,term_list,state_list):
    H0=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for term in term_list:
        H0=H0+tpp(state_list=state_list,pauli_dic=term[0],prefactor=term[1](km[0],km[1]))
    H0=v*H0
    
    return H0

def h_linear_diag2(km,v,term_list,state_list):
    H0=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for term in term_list:
        H0=H0+tpp_new(state_list=state_list,pauli_dic=term[0],prefactor=term[1](km[0],km[1]))
    H0=v*H0
    
    return H0


#test1=h_linear_diag(km=np.array([0,0]),v=1,term=diag_term,state_list=basis1)
# test2=tpp(state_list=basis1,pauli_dic=diag_term[0],prefactor=diag_term[1](2,0))
# print(test2)
# inspect_elements(test2,basis1)
# testh=h_linear_diag(km=np.array([1,2]),v=1,term_list=linear_terms,state_list=basis2)
# inspect_elements(testh,basis2)
# exit()


def h_tun(km,tun_terms,state_list):
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for term in tun_terms:
        H1+=tpp(state_list=state_list,pauli_dic=term[0],prefactor=term[1])

    H1=H1+np.conjugate(np.transpose(H1))

    return H1

# testtun=h_tun(km=np.array([1,1]),tun_terms=tun_terms,state_list=basis1)
# inspect_elements(testtun,state_list=basis1)
# print(phi/np.pi)
# exit()

def Schrieffer_Wolff(state_list,h0,V):
    #1. Diagonalize H0
    h0_eigvals,h0_eigvecs=np.linalg.eigh(h0)
    diag_h0=np.dot(np.conjugate(np.transpose(h0_eigvecs)),np.dot(h0,h0_eigvecs))
    diag_check=np.allclose(np.diag(diag_h0)-h0_eigvals,np.zeros((len(state_list),len(state_list))))
    #2. Compute the second term in the new basis
    #2a. Transform V0 into the eigenbasis
    Vprime=np.dot(np.conjugate(np.transpose(h0_eigvecs)),np.dot(V,h0_eigvecs))
    V2prime=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for i in range(len(state_list)):
        for j in range(len(state_list)):
            matrix_element=0
            for k in range(len(state_list)):
                tempv=Vprime[i,k]*Vprime[k,j]
                if np.isclose(np.abs(tempv),0)==False:
                    matrix_element=matrix_element+tempv*(1/(diag_h0[i,i]-diag_h0[k,k])+1/(diag_h0[j,j]-diag_h0[k,k]))

            V2prime[i,j]=matrix_element
    V2prime=(1/2)*V2prime
    V2new=np.dot(h0_eigvecs,np.dot(V2prime,np.conjugate(np.transpose(h0_eigvecs))))
            
    

    return V2new

#h_linear_diag(km=np.array([1,2]),v=1,term_list=linear_terms,state_list=basis2)


def tunn_blocks_linear(state_list,tun_pauli_dic,prefactor,phi):
    
    return None

def tunnelling_blocks(k0,theta,phi,state_list,tun_pauli_dic,prefactor,hopping):
    ktheta=np.sqrt(np.vdot(kd,kd))*np.sin(theta/2)
    q1=2*ktheta*k1/(np.sqrt(np.vdot(k1,k1)))
    q2=2*ktheta*k2/(np.sqrt(np.vdot(k2,k2)))
    q3=2*ktheta*k3/(np.sqrt(np.vdot(k3,k3)))
    kdic={0:k0,1:k0+q1,2:k0+q2,3:k0+q3}
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    #Idea is to constantly overwrite
    for state in state_list:
        result_list=[]
        for k in state.particle_dic.keys():
            
            if state.particle_dic[k].dof_dic[0]==hopping[0]:
                #hopping part

                #sublattice part
                result_pairs=tun_pauli_dic[1](state.particle_dic[k].dof_dic[1],phi)
                for output in result_pairs:
                    coeff=1
                    particle_dic_temp1={}
                    dof_dic_temp={}
                    dof_dic_temp[0]=hopping[1]
                    dof_dic_temp[1]=output[0]
                    coeff=coeff*output[1]
                    #spin part
                    dof_dic_temp[2]=tun_pauli_dic[2](state.particle_dic[k].dof_dic[2])[1]
                    coeff=coeff*tun_pauli_dic[2](state.particle_dic[k].dof_dic[2])[0]

                    particlen=particle(dof_dic=dict(dof_dic_temp))
                    # print(vars(particlen))
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
            
    return np.asarray(H1+np.asmatrix(H1).getH())









def h_k(k0,theta,q,state_list):
    H0=np.zeros((4,4),dtype=complex)
    theta_arg=get_theta(k0)-theta/2
    H1=tpp(state_list=state_list,pauli_dic={0:p0,1:px,2:p0},prefactor=np.cos(theta_arg))
    H2=tpp(state_list=state_list,pauli_dic={0:p0,1:py,2:p0},prefactor=-np.sin(theta_arg))
    H0=H0-v*np.sqrt(np.vdot(k0+q,k0+q))*(H1+H2)

    return H0





def nzindices(wf):
    nzindices=[]
    for j in range(0,len(wf)):
        if np.isclose(np.abs(wf[j]),0):
                pass
        else:
            nzindices.append(j)
    return nzindices




def HK_sublattice(state_list,U): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        U_count=0
        list1=list(state.particle_dic.values())
        list2=list(state.particle_dic.values())
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        for p1 in list1:
            for p2 in list2:
                initial=True
                for d in range(0,dof-1):
                    initial=initial and (p1.dof_dic[d]==p2.dof_dic[d])
                initial=initial and (p1.dof_dic[dof-1]==((p2.dof_dic[dof-1]+1)%2))
                if initial:
                    U_count+=1
                    list2.remove(p1)
                    list2.remove(p2)
                    list1.remove(p1)
                    list1.remove(p2)
                    position1=(state_list.index(state),state_list.index(state))
                    temp_H[position1]=U+temp_H[position1]
        H1=H1+temp_H
    return H1

def HK_sublattice2(state_list,U): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        U_count=0
        particles=list(state.particle_dic.values())
        list2=list(state.particle_dic.values())
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        for pdic1 in particles:
            temp_dof_dic={}
            for d in range(dof-1):
                temp_dof_dic[d]=pdic1.dof_dic
            temp_dof_dic[dof]=(pdic1.dof_dic[d]+1)%2
            #particles.count({0:pdic1.dof_dic[0],1:pdic1.dof_dic[1],2:(pdic1.dof_dic[2]+1)%2})
            tempp=particle(dof_dic=temp_dof_dic)
            for j in particles:
                if j==tempp:
                    U_count+=1
        U_count=U_count/2
        position1=(state_list.index(state),state_list.index(state))
        temp_H[position1]=U_count*U+temp_H[position1]
    
            
        H1=H1+temp_H
    return H1

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


def HK_Nmusigma(state_list,U): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        U_count=0
        list1=list(state.particle_dic.values())
        list2=list(state.particle_dic.values())
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        for p1 in list1:
            for p2 in list2:
                initial=True
                for d in range(0,dof-1):
                    initial=initial and (p1.dof_dic[d]==((p2.dof_dic[d]+1)%2))
                initial=initial and (p1.dof_dic[dof-1]==((p2.dof_dic[dof-1]+1)%2))
                if initial:
                    U_count+=1
                    list2.remove(p1)
                    list2.remove(p2)
                    list1.remove(p1)
                    list1.remove(p2)
                    position1=(state_list.index(state),state_list.index(state))
                    temp_H[position1]=U+temp_H[position1]
        H1=H1+temp_H
    return H1


def HK_mutu(state_list,U): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        U_count=0
        list1=list(state.particle_dic.values())
        list2=list(state.particle_dic.values())
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        for p1 in list1:
            for p2 in list2:
                initial=True
                for d in range(0,dof-1):
                    initial=initial and (p1.dof_dic[d]==(p2.dof_dic[d]+1)%2)
                initial=initial and (p1.dof_dic[dof-1]==((p2.dof_dic[dof-1]+1)%2))
                if initial:
                    U_count+=1
                    list2.remove(p1)
                    list2.remove(p2)
                    list1.remove(p1)
                    list1.remove(p2)
                    position1=(state_list.index(state),state_list.index(state))
                    temp_H[position1]=U+temp_H[position1]
        H1=H1+temp_H
    return H1



def mu_N(state_list,mu):
    H0=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        n=len(list(state.particle_dic.keys()))
        position1=(state_list.index(state),state_list.index(state))
        H0[position1]=mu*n+H0[position1]
    return H0

def HK_band(state_list,U,kinetic_terms,kx,ky): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    H0=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for term in kinetic_terms:
        H0=H0+tpp(state_list=state_list,pauli_dic=term[2],prefactor=term[0]*term[1](kx,ky))
    eigvals0,eigvecs0=np.linalg.eig(H0)
    for state in state_list:
        U_count=0
        statewf=np.zeros(len(state_list),dtype=complex)
        statewf[state_list.index(state)]=1
        list1=list(state.particle_dic.values())
        list2=list(state.particle_dic.values())
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        for p1 in list1:
            for p2 in list2:
                initial=True
                for d in range(0,dof-1):
                    initial=initial and (p1.dof_dic[d]==p2.dof_dic[d])
                initial=initial and (p1.dof_dic[dof-1]==((p2.dof_dic[dof-1]+1)%2))#Checks spins are opposite
                if initial:
                    U_count+=1
                    list2.remove(p1)
                    list2.remove(p2)
                    list1.remove(p1)
                    list1.remove(p2)
                    position1=(state_list.index(state),state_list.index(state))
                    for k in range(0,len(eigvals0)):
                        temp_H[position1]=U*np.vdot(eigvecs0[:,k],statewf)+temp_H[position1]
        H1=H1+temp_H
    return H1




def HK_mu(state_list,Umu): #need reverse here too? #ONLY APPLICABLE FOR THREE PARTICLES
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        U_count=0
        list1=list(state.particle_dic.values())
        list2=list(state.particle_dic.values())
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        for p1 in list1:
            for p2 in list2:
                initial=True
                for d in range(1,2):#Hardcoded that dof=2!
                    initial=initial and (p1.dof_dic[d]==p2.dof_dic[d])
                initial=initial and (p1.dof_dic[0]==((p2.dof_dic[0]+1)%2))
                if initial:
                    U_count+=1
                    list2.remove(p1)
                    list2.remove(p2)
                    list1.remove(p1)
                    list1.remove(p2)
                    position1=(state_list.index(state),state_list.index(state))
                    temp_H[position1]=Umu+temp_H[position1]
        H1=H1+temp_H
    return H1








#######################################################DIAGONALIZATION############################################################



#Seem to have an issue with squaring generators.
def faction(state,pauli_dic,coeff,dof):
    result_list=[]
    dof_dic_temp={}
    particle_dic_temp={}
    for k in state.particle_dic.keys():
        for l in range(0,dof):
            dof_dic_temp[l]=pauli_dic[l](state.particle_dic[k].dof_dic[l])[1]
            coeff=coeff*pauli_dic[l](state.particle_dic[k].dof_dic[l])[0]
        particlen=particle(dof_dic=dict(dof_dic_temp))
        # print(vars(particlen))
        particle_dic_temp[k]=particlen
        for j in state.particle_dic.keys():
            if j!=k:
                particle_dic_temp[j]=state.particle_dic[j]
        new_state=basis(particle_dic=dict(particle_dic_temp))
        # print(eyepreservation(new_state))
        result=(coeff,new_state)
        result_list.append(result)

    return result_list
    
def tpptwice2(state_list,pauli_dic1,pauli_dic2,Uff,dof):
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        result_list=[]
        result_list1=faction(state=state,pauli_dic=pauli_dic1,coeff=1,dof=dof)
        #Which then needs to get acted on by V(G) again (really V^\dagger)
        for i in result_list1:
            #And the result of that into f2:
            result_list2=faction(state=i[1],pauli_dic=pauli_dic2,coeff=i[0],dof=dof)
            #And then again with V (^\dagger)

            for j in result_list2:
                result_list.append((j[0],j[1]))

        for result in result_list:
            for i in state_list:
                if basis.eqnoorder(i,result[1]):
                    temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
                    swapcount=basis.swaps(result[1],i)
                    position1=(state_list.index(i),state_list.index(state))
                    temp_H[position1]=((-1)**swapcount)*result[0]*Uff
                    H1=H1+temp_H
    return H1 


def HK_N_monolayer(state_list,pd1,Un):
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        U_count=0
        list1=list(state.particle_dic.values())
        list2=list(state.particle_dic.values())
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        for p1 in list1:
            for p2 in list1:
                initial=True
                coeff=1
                for d in range(0,2):
                    # print(p1.dof_dic[d])
                    # print(p2.dof_dic[d])
                    # print(pd1[d](p2.dof_dic[d]))
                    # exit()
                    initial=(initial and (p1.dof_dic[d]==pd1[d](p2.dof_dic[d])[1]))
                    coeff=coeff*pd1[d](p2.dof_dic[d])[0]
                if initial:
                    U_count+=coeff
                    # list2.remove(p1)
                    # list2.remove(p2)
                    # list1.remove(p1)
                    # list1.remove(p2)
                    position1=(state_list.index(state),state_list.index(state))
                    temp_H[position1]=U_count*Un+temp_H[position1]
        H1=H1+temp_H
    return H1

def HK_N_monolayer2(state_list,pd1,Un):
    H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for state in state_list:
        U_count=0
        list1=list(state.particle_dic.values())
        list2=[x.dof_dic for x in state.particle_dic.values()]
        temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
        for p1 in list1:
            coeff=1
            p2_dic={}
            for d in range(0,2):
                p2_dic[d]=pd1[d](p1.dof_dic[d])[1]
            print(f"p1:{p1.dof_dic} ,p2:{p2_dic}")
            if p2_dic in list2:
                # print(p2_dic)
                coeff=1
                for d2 in range(0,2):
                    coeff=coeff*pd1[d2](p1.dof_dic[d2])[0]
                U_count+=coeff
                    # list2.remove(p1)
                    # list2.remove(p2)
                    # list1.remove(p1)
                    # list1.remove(p2)
        position1=(state_list.index(state),state_list.index(state))
        temp_H[position1]=U_count*Un+temp_H[position1]
        H1=H1+temp_H
    return H1

def eyepreservation_mono(state):#Leave this at three states
    particle_list=[state.particle_dic[x] for x in state.particle_dic.keys()]
    symbol_map0={0:'Q0',1:'Q1',2:'Q2',3:'Q3'}
    symbol_map1={0:'A',1:'B'}
    symbol_map2={0:u'\u2191',1:u'\u2193'}
    helpful_string=''
    for i in particle_list:
        for k in i.dof_dic.keys():
            if k==0:
                helpful_string=helpful_string+symbol_map1[i.dof_dic[k]]
            elif k==1:
                helpful_string=helpful_string+symbol_map2[i.dof_dic[k]]
            # elif k==0:
            #     helpful_string=helpful_string+symbol_map0[i.dof_dic[k]]
        helpful_string=helpful_string+';'
    return helpful_string






def generate_H2(kx,ky,theta,phi,state_list,diag_terms,tun_terms,UHK,mu):
    H0=np.zeros((len(state_list),len(state_list)),dtype=complex)
    for term in diag_terms:
        H0=H0+diagtpp(state_list=state_list,theta=theta,k0=np.array([kx,ky]),layer_pauli_dic=term[1],prefactor=term[0])
    for term in tun_terms:
        H0=H0+tunnelling_blocks(theta=theta,phi=phi,k0=np.array([kx,ky]),state_list=state_list,tun_pauli_dic=term[2],prefactor=term[0],hopping=term[1])
    H0=H0+HK_sublattice(state_list=state_list,U=UHK)
    H0=H0-mu_N(state_list=state_list,mu=mu)
    
    return H0





if __name__ == "__main__":
    import pathdiag
    import plot
    import plot_save
    import os

    testpdd={'k':4,'sublattice':2,'spin':2}
    kd2=np.array([0,4*np.pi/3])
    k1=np.array([0,-1])
    k2=np.array([np.sqrt(3)/2,1/2])
    k3=np.array([-np.sqrt(3)/2,1/2])
    #########

    dof=4
    
    tqs=[np.array([-1,-1]),np.array([1,0]),np.array([0,1])]
    #test_linear_terms=[({0:p0,1:qkx,2:px,3:p0},gkx),({0:p0,1:qky,2:py,3:p0},gky)]
    #Can put in small angle correction, from TBG II, but it doesn't look like it makes much difference. Argh!
    test_linear_terms=[({0:p0,1:qkx,2:px,3:p0},g0),({0:p0,1:qky,2:py,3:p0},g0),({0:p0,1:t0,2:px,3:p0},gkx),({0:p0,1:t0,2:py,3:p0},gky)]
    test_linear_terms_nonzero=[({0:p0,1:qkx,2:px,3:p0},gw(np.cos(theta/2))),({0:p0,1:t0,2:px,3:p0},gkxw(np.cos(theta/2))),
                            ({0:pz,1:qkx,2:py,3:p0},gw(-np.sin(theta/2))),({0:pz,1:t0,2:py,3:p0},gkxw(-np.sin(theta/2))),
                            ({0:p0,1:qky,2:py,3:p0},gw(np.cos(theta/2))),({0:p0,1:t0,2:py,3:p0},gkyw(np.cos(theta/2))),
                            ({0:pz,1:qky,2:px,3:p0},gw(np.sin(theta/2))),({0:pz,1:t0,2:px,3:p0},gkyw(np.sin(theta/2))),
                            ]
    
    linear_terms_kx=[({0:p0,1:t0,2:px,3:p0},gw(np.cos(theta/2))),({0:pz,1:t0,2:py,3:p0},gw(-np.sin(theta/2)))]
    linear_terms_ky=[({0:p0,1:t0,2:py,3:p0},gw(np.cos(theta/2))),({0:pz,1:t0,2:px,3:p0},gw(np.sin(theta/2)))]
    linear_terms_constant=[({0:p0,1:qkx,2:px,3:p0},gw(np.cos(theta/2))),({0:pz,1:qkx,2:py,3:p0},gw(-np.sin(theta/2))),
                            ({0:p0,1:qky,2:py,3:p0},gw(np.cos(theta/2))),({0:pz,1:qky,2:px,3:p0},gw(np.sin(theta/2)))]



    shell_basis=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=particle_no,nonlayer=testnonlayer,center=center)
    print(f'Number of basis states {len(shell_basis)}')


    def make_matrices(state_list,pauli_dic,kfunction,variable_names,variable_factors,variable_functions,final_matrix_description):
        if ((qkx in pauli_dic.values()) or (qky in pauli_dic.values())):
            print('qkx or qky in pauli_dic')
            matrix=tpp(state_list=state_list,pauli_dic=pauli_dic,prefactor=1)/np.sin(theta/2)
        else:
            matrix=tpp(state_list=state_list,pauli_dic=pauli_dic,prefactor=1)
        matrix_object=saved_template_matrix(matrix=matrix,kfunction=kfunction,variable_functions=variable_functions,variable_names=variable_names,variable_factors=variable_factors,final_matrix_description=final_matrix_description)
        return matrix_object
    def make_matrices_U(state_list,type,pdic_n,U,final_matrix_description):
        if type=='HK_N':
            matrix=HK_N(state_list=state_list,pdic_n=pdic_n,U_N=U)
            nstring=''
            for key in sorted(list(pdic_n.keys())):
                nstring+=f'{pdic_n[key].__name__}'
            matrix_name='UHK_N_'+nstring
            print(f' matrix name: {matrix_name}')
            matrix_object=saved_template_matrix(matrix=matrix,kfunction=g0,variable_functions=[g00],variable_names=[matrix_name],variable_factors=[1],final_matrix_description=final_matrix_description)
        elif type=='HK_rot':
            matrix=HK_rot(state_list=state_list,U_rot=U)
            matrix_object=saved_template_matrix(matrix=matrix,kfunction=g0,variable_functions=[g00],variable_names=['UHK_rot'],variable_factors=[1],final_matrix_description=final_matrix_description)
            
        return matrix_object
    
    linear_terms_kx_save=[({0:p0,1:t0,2:px,3:p0},gkx,[np.cos,g00],['theta','v'],[1/2,1]),({0:pz,1:t0,2:py,3:p0},gkx,[np.sin,g00],['theta','v'],[-1/2,1])]
    linear_terms_ky_save=[({0:p0,1:t0,2:py,3:p0},gky,[np.cos,g00],['theta','v'],[1/2,1]),({0:pz,1:t0,2:px,3:p0},gky,[np.sin,g00],['theta','v'],[1/2,1])]
    linear_terms_constant_save=[({0:p0,1:qkx,2:px,3:p0},g00,[np.cos,g00,np.sin],['theta','v','theta'],[1/2,1,1/2]),({0:pz,1:qkx,2:py,3:p0},g00,[np.sin,g00,np.sin],['theta','v','theta'],[-1/2,1,1/2]),
                                ({0:p0,1:qky,2:py,3:p0},g00,[np.cos,g00,np.sin],['theta','v','theta'],[1/2,1,1/2]),({0:pz,1:qky,2:px,3:p0},g00,[np.sin,g00,np.sin],['theta','v','theta'],[1/2,1,1/2])]
    
    test_tun_terms=[({0:px,1:t1_plus,2:p0,3:p0},gw(w0)),({0:px,1:t1_minus,2:p0,3:p0},gw(w0)),({0:px,1:t1_plus,2:px,3:p0},gw(w1*np.cos(phi*(1-1)))),({0:px,1:t1_minus,2:px,3:p0},gw(w1*np.cos(phi*(1-1)))),
                    ({0:px,1:t2_plus,2:p0,3:p0},gw(w0)),({0:px,1:t2_plus,2:px,3:p0},gw(w1*np.cos(phi*(2-1)))),({0:px,1:t2_plus,2:py,3:p0},gw(w1*np.sin(phi*(2-1)))),
                    ({0:px,1:t2_minus,2:p0,3:p0},gw(w0)),({0:px,1:t2_minus,2:px,3:p0},gw(w1*np.cos(phi*(2-1)))),({0:px,1:t2_minus,2:py,3:p0},gw(w1*np.sin(phi*(2-1)))),
                    ({0:px,1:t3_plus,2:p0,3:p0},gw(w0)),({0:px,1:t3_plus,2:px,3:p0},gw(w1*np.cos(phi*(3-1)))),({0:px,1:t3_plus,2:py,3:p0},gw(w1*np.sin(phi*(3-1)))),
                    ({0:px,1:t3_minus,2:p0,3:p0},gw(w0)),({0:px,1:t3_minus,2:px,3:p0},gw(w1*np.cos(phi*(3-1)))),({0:px,1:t3_minus,2:py,3:p0},gw(w1*np.sin(phi*(3-1))))]
    test_tun_terms_save=[({0:px,1:t1_plus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t1_minus,2:p0,3:p0},g0,[g00],['w0'],[1]),
                        ({0:px,1:t1_plus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(1-1)]),({0:px,1:t1_minus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(1-1)]),
                        ({0:px,1:t2_plus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t2_plus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(2-1)]),({0:px,1:t2_plus,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(2-1)]),
                        ({0:px,1:t2_minus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t2_minus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(2-1)]),({0:px,1:t2_minus,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(2-1)]),
                       ({0:px,1:t3_plus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t3_plus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(3-1)]),({0:px,1:t3_plus,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(3-1)]), 
                       ({0:px,1:t3_minus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t3_minus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(3-1)]),({0:px,1:t3_minus,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(3-1)])
                       ]

    def make_each_matrix(term_list,state_list,dirname,matrix_name,type):
        os.makedirs(dirname, exist_ok=True)
        filename=f'{dirname}/{matrix_name}'

        for i in range(len(term_list)):
            print(f'i = {i}')
            start=time.time()
            if type=='nonint':
                test_matrix=make_matrices(state_list=state_list,pauli_dic=term_list[i][0],kfunction=term_list[i][1],
                variable_functions=term_list[i][2],variable_names=term_list[i][3],
                variable_factors=term_list[i][4],final_matrix_description='linear kx non-zero angle')
            elif (type=='HK_N' or type=='HK_rot'):
                test_matrix=make_matrices_U(state_list=state_list,type=type,pdic_n=term_list[i][0],U=1,
                                            final_matrix_description='linear kx non-zero angle')
                
            end=time.time()
            # print(f'Matrix construction time: {end-start}')
            # print(f'Object size: {sys.getsizeof(test_matrix)}')
            # print(test_matrix.kfunction.__name__)
            # print(np.cos.__name__)
            # print(globals()['theta']*180/np.pi)

            if len(term_list)>1:
                with open(filename+'_'+f'{i}'+'.dill', 'wb') as file:
                    pickle.dump(test_matrix, file)
                    print(filename+'_'+f'{i}'+'.dill')
            else:
                print(filename+'_'+f'{int(sys.argv[1])-1}'+'.dill')
                with open(filename+'_'+f'{int(sys.argv[1])-1}'+'.dill', 'wb') as file:
                    pickle.dump(test_matrix, file)

            reconstructed_matrix=test_matrix.form_matrix()
            
            
            # print(f'first 4x4 of reconstructed matrix')
            # print(reconstructed_matrix[:4,:4])
            #test=h_linear_diag(km=B,v=v,term_list=[linear_terms_kx[i]],state_list=state_list)
            #print(np.allclose(reconstructed_matrix,np.abs(test)))
    

    
    
    def load_matrices(filelist):
        first=True
        for matrix_file in filelist:
            print(matrix_file)
            if first:
                with open(matrix_file,'rb') as f:
                    combined_matrix_object=dill.load(f)
                    combined_matrix=combined_matrix_object.form_matrix()
                    del combined_matrix_object
                first=False
            else:
                with open(matrix_file,'rb') as f:
                    temp_matrix_object=dill.load(f)
                temp_matrix=temp_matrix_object.form_matrix()
                print(temp_matrix.shape)
                del temp_matrix_object
                combined_matrix=combined_matrix+temp_matrix
                gc.collect()#To remove the overwritten variable if not done already.

        return combined_matrix
    

    def make_template_files(particle_no,shell_count,target_terms,name,indiviudal,type):
        shells_particle=generate_shell_basis_gamma(shell_count=shell_count,q_vecs=tqs,number_of_particles=particle_no,nonlayer=testnonlayer,center=center)
        k0=B
        cluster_arg=int(sys.argv[1])-1
        dir_path=f'{particle_no}particles_{shells_used}shells_center{center}_matrices/{name}_particles{particle_no}_shells{shell_count}_center{center}'
        
        # for the cluster
        if indiviudal:
            make_each_matrix(term_list=[target_terms[cluster_arg]],state_list=shells_particle,dirname=dir_path,matrix_name=name,type=type)
        else:
            make_each_matrix(term_list=target_terms,state_list=shells_particle,dirname=dir_path,matrix_name=name,type=type)

    #make_template_files(particle_no=2,shell_count=shells_used,target_terms=[[{0:p0,1:t0,2:p0,3:px}]],name='HK',indiviudal=False,type='HK_N')
    

    
    
    def test_matrices_against_diag(matrix_dir,target_terms):
        loaded_matrix=load_matrices(matrix_dir)
        print(f'loaded matrix shape {loaded_matrix.shape}')
        shells_particle=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=particle_no,nonlayer=testnonlayer,center=center)
        H_linear=h_linear_diag(km=B,v=v,term_list=target_terms,state_list=shell_basis)
        print(f'target matrix shape {H_linear.shape}')

        
        
        return None



    # shells2=generate_shell_basis_gamma(shell_count=2,q_vecs=tqs,number_of_particles=1,nonlayer=testnonlayer,center=center)
    # k0=np.array([1,2])
    # cluster_arg=int(sys.argv[1])-1
    # dir_path=f'UHK_matrices_particles{particle_no}_shells{shells_used}_center{center}'
    # U_HK_nonint=np.zeros((16,16),dtype=complex)
    # U_HK_nonint_obj=saved_template_matrix(matrix=U_HK_nonint,kfunction=g00,variable_names=['UHK_rot'],variable_factors=[1],variable_functions=[g00],final_matrix_description='Non int UHK rot (all zeros)')
    # U_filename='1particles_2shells_centerK_matrices/UHK_particles_1_shells_2_center_K'
    # # with open(U_filename+'.dill', 'wb') as file:
    # #     pickle.dump(U_HK_nonint_obj, file)
    # with open(U_filename+'.dill', 'rb') as file:
    #     obj=pickle.load(file)
    # print(obj.variable_names)
    # obj.variable_names[0]='UHK'
    # with open(U_filename+'.dill', 'wb') as file:
    #     pickle.dump(obj, file)
    #make_each_matrix(term_list=[test_tun_terms_save[cluster_arg]],state_list=shell_basis,dirname=dir_path,matrix_name='U_HK',type='U_HK')
    
    #test new U
    #shells2=generate_shell_basis_gamma(shell_count=2,q_vecs=tqs,number_of_particles=2,nonlayer=testnonlayer,center=center)


    # for state in shells2[:10]:
    #     print(f'State: {eyepreservation(state)}, spin 1: {state.particle_dic[1].dof_dic[dof-1]},spin 2: {state.particle_dic[2].dof_dic[dof-1]}')
    
    # testU=HK_rot(state_list=shells2,U_rot=1)
    # inspect_elements(state_list=shells2,matrix=testU)
    # exit()


    #I think this is the check
    #make_each_matrix(term_list=linear_terms_kx_save,state_list=shell_basis,dirname=dir_path)
    # filelist=[dir_path+'/'+f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    # filelist2=[f for f in filelist if f'_{cluster_arg}' in f]
    # test_load=load_matrices(filelist)
    # #print('loaded')
    # #print(test_load[:16,:16])
    # kind_matrix=h_linear_diag2(km=k0,v=1,term_list=linear_terms_kx,state_list=shell_basis)
    # #print('reconstructed')
    # #print(kind_matrix)
    # print(f'Loaded matrix equal to reconstructed: {np.allclose(test_load,kind_matrix)}')
    # exit()

    ###############CURRENT CODE FOR CONSTRUCTING MATRICES!##########################################################


#     k0=B#shouldn't matter
#     cluster_arg=int(sys.argv[1])-1
#     #dir_path=f'{particle_no}particles_{shells_used}shells_center{center}_matrices_new/kindmatrices_particles{particle_no}_shells{shells_used}_center{center}'
#     dir_path=f'UHK_N_taux_matrix_particles{particle_no}_shells{shells_used}_center{center}'
#     #print(dir_path)
#     # for the cluster
#     print('start make')
#    #make_each_matrix(term_list=[linear_terms_constant_save[cluster_arg]],state_list=shell_basis,dirname=dir_path,matrix_name='kindmatrix',type='nonint')
#     make_each_matrix(term_list=[[{0:p0,1:t0,2:px,3:p0}]],state_list=shell_basis,dirname=dir_path,matrix_name='HK_N_taux',type='HK_N')
    
#     exit()


    ##############################################################################################################################################################################
    #os.makedirs(dir_path, exist_ok=True)
    # U_filename=f'UHK_particles_{particle_no}_shells_{shells_used}_center_{center}'
    # madeU=make_matrices_U(state_list=shell_basis,pdic_n={0:p0,1:t0,2:p0,3:px},type='HK_N',U=1,final_matrix_description=U_filename)
    # with open(U_filename+'.dill', 'wb') as file:
    #     pickle.dump(madeU, file)
    
    # #make_each_matrix(term_list=linear_terms_kx_save,state_list=shell_basis,dirname=dir_path)
    # # filelist=[dir_path+'/'+f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    # # filelist2=[f for f in filelist if f'_{cluster_arg}' in f]
    # #test_load=load_matrices(filelist)
    # with open(U_filename+'.dill', 'rb') as file:
    #     test_load=pickle.load(file).form_matrix()
    # print('loaded')
    # print(test_load[:6,:6])
    # kind_matrix=HK_N(state_list=shell_basis,pdic_n={0:p0,1:t0,2:p0,3:px},U_N=1)
    # print('reconstructed')
    # print(kind_matrix[:6,:6])
    # print(f'Loaded matrix equal to reconstructed: {np.allclose(test_load,kind_matrix)}')
    
    
    
    
    # print(f'shape of loaded matrix: {test_load.shape}')
    # kind_matrix=h_linear_diag2(km=k0,v=v,term_list=[linear_terms_kx[cluster_arg]],state_list=shell_basis)
    # print(f'Loaded matrix equal to reconstructed: {np.allclose(test_load,kind_matrix)}')
    
    #You now need to make the matrices that will get multplied by a given k

    
    
    
    # print(v)
    # print(k0)

    # print(len(shell_basis))

    # start=time.time()
    # kx_matrix=h_linear_diag2(km=k0,v=v,term_list=linear_terms_kx,state_list=shell_basis)
    # print(np.allclose(np.zeros((len(shell_basis),len(shell_basis)),dtype=complex),test_load))
    # end=time.time()
    # exit()


    

    # for state in shell_basis:
    #     print(eyepreservation(state))
    # print(len(shell_basis))
    # exit()

    # start=time.time()
    # test_tpp=tpp(state_list=shell_basis,pauli_dic={0:p0,1:qkx,2:px,3:p0},prefactor=1)
    # end=time.time()
    # print(start-end)
    # test_tun_terms=[({0:px,1:t1_plus,2:p0,3:p0},gw(w0)),({0:px,1:t1_minus,2:p0,3:p0},gw(w0)),({0:px,1:t1_plus,2:px,3:p0},gw(w1*np.cos(phi*(1-1)))),({0:px,1:t1_minus,2:px,3:p0},gw(w1*np.cos(phi*(1-1)))),
    #                 ({0:px,1:t2_plus,2:p0,3:p0},gw(w0)),({0:px,1:t2_plus,2:px,3:p0},gw(w1*np.cos(phi*(2-1)))),({0:px,1:t2_plus,2:py,3:p0},gw(w1*np.sin(phi*(2-1)))),
    #                 ({0:px,1:t2_minus,2:p0,3:p0},gw(w0)),({0:px,1:t2_minus,2:px,3:p0},gw(w1*np.cos(phi*(2-1)))),({0:px,1:t2_minus,2:py,3:p0},gw(w1*np.sin(phi*(2-1)))),
    #                 ({0:px,1:t3_plus,2:p0,3:p0},gw(w0)),({0:px,1:t3_plus,2:px,3:p0},gw(w1*np.cos(phi*(3-1)))),({0:px,1:t3_plus,2:py,3:p0},gw(w1*np.sin(phi*(3-1)))),
    #                 ({0:px,1:t3_minus,2:p0,3:p0},gw(w0)),({0:px,1:t3_minus,2:px,3:p0},gw(w1*np.cos(phi*(3-1)))),({0:px,1:t3_minus,2:py,3:p0},gw(w1*np.sin(phi*(3-1))))]
    
    
    
    #tlin=h_linear_diag(km=k0,v=1,term_list=test_linear_terms_nonzero,state_list=shell_basis)+h_linear_diag(km=k0,v=1,term_list=test_tun_terms,state_list=shell_basis)



    
    
    def gen_Hk(kx,ky,UHK):
        H_linear=h_linear_diag(km=np.array([kx,ky]),v=v,term_list=test_linear_terms_nonzero,state_list=shell_basis)
        H_tun=h_linear_diag(km=np.array([kx,ky]),state_list=shell_basis,term_list=test_tun_terms,v=1)
        H_HK=HK_sublattice(state_list=shell_basis,U=UHK)
        return (H_linear+H_tun)
    
    def gen_testH(kx,ky,UHK,state_list):
        H_linear=h_linear_diag(km=np.array([kx,ky]),v=v,term_list=test_linear_terms_nonzero,state_list=state_list)
        H_tun=h_linear_diag(km=np.array([kx,ky]),state_list=state_list,term_list=test_tun_terms,v=1)
        H_HK=HK_sublattice(state_list=shell_basis,U=UHK)
        return (H_linear+H_tun)

    #Code for constructing the template matrices


    #Naive tensor product



    def test_times(numberlist,tpp1):
        tpp_times=[]
        new_tpp_times=[]
        fig,axs=plt.subplots(1,len(numberlist))
        for shell in numberlist:
            shell_basis_temp=generate_shell_basis_gamma(shell_count=shell,q_vecs=tqs,number_of_particles=2,nonlayer=testnonlayer,center=center)
            print(len(shell_basis_temp))
            for term in linear_terms_kx[:1]:
                H0,times=tpp1(state_list=shell_basis_temp,pauli_dic=term[0],prefactor=term[1](k0[0],k0[1]))
            print(len(times))
            

            for i in range(len(times)):
                axs[numberlist.index(shell)].scatter(x=[range(len(times[0]))],y=[times[i]])
            axs[numberlist.index(shell)].set_xticks(range(len(times[0])))
            axs[numberlist.index(shell)].set_xticklabels(labels=['Step 1','If','Swapcount','Add'])
        plt.show()
            
        return tpp_times,new_tpp_times
    
    # def test_times2(numberlist,tpp1,tpp2):
    #     tpp_times=[]
    #     shells=[]
    #     new_tpp_times=[]
    #     fig,axs=plt.subplots(1,len(numberlist))
    #     for number in numberlist:
    #         shell_basis_temp=generate_shell_basis_gamma(shell_count=2,q_vecs=tqs,number_of_particles=number,nonlayer=testnonlayer,center=center)
    #         print(len(shell_basis_temp))
    #         for term in linear_terms_kx[:1]:
    #             start=time.time()
    #             H1=tpp1(state_list=shell_basis_temp,pauli_dic=term[0],prefactor=term[1](k0[0],k0[1]))
    #             end=time.time()
    #             time1=end-start
    #             start=time.time()
    #             H2=tpp2(state_list=shell_basis_temp,pauli_dic=term[0],prefactor=term[1](k0[0],k0[1]))
    #             end=time.time()
    #             time2=end-start
    #             print(np.allclose(np.real(H1),np.real(H2)))
    #             print(np.allclose(np.imag(H1),np.imag(H2)))
    #             tpp_times.append([time1,time2])
    #             shells.append(len(shell_basis_temp))

            
    #         axs[numberlist.index(number)].scatter(x=shells,y=[t[0] for t in tpp_times])
    #         axs[numberlist.index(number)].scatter(x=shells,y=[t[1] for t in tpp_times])
    #         #axs[shellslist.index(shell)].set_xticks(range(len(shells)))
    #         #axs[shellslist.index(shell)].set_xticklabels(labels=)
    #     plt.show()
            
    #     return tpp_times,new_tpp_times
    
    # # print(f"tpp times:")
    # test_times(numberlist=[1,2,3,4,5],tpp1=tpp_time)


    # exit()
    # start=time.time()
    # kx_matrix=h_linear_diag2(km=k0,v=v,term_list=linear_terms_kx,state_list=shell_basis)
    # end=time.time()
    # kx_time=end-start
    # print(f'kx matrix time: {end-start}')
    # with open(f'kxmatrix_particles{particle_no}_shells{shells_used}_center_{center}','wb') as handle:
    #     pickle.dump(kx_matrix,handle)
    # exit()
    # start=time.time()
    # ky_matrix=h_linear_diag(km=k0,v=v,term_list=linear_terms_ky,state_list=shell_basis)
    # end=time.time()
    # print(f'ky matrix time: {end-start}')
    # with open(f'kymatrix_particles{particle_no}_shells{shells_used}_center_{center}','wb') as handle:
    #     pickle.dump(ky_matrix,handle)
    # print(end-start)
    # start=time.time()
    # kind_matrix_linear=h_linear_diag(km=k0,v=v,term_list=linear_terms_constant,state_list=shell_basis)
    # end=time.time()
    # print(f'kind_matrix_linear matrix time: {end-start}')
    # with open(f'kindmatrix_particles{particle_no}_shells{shells_used}_center_{center}','wb') as handle:
    #     pickle.dump(kind_matrix_linear,handle)
    # print(f'kind_m: {end-start}')
    # start=time.time()
    # tun_matrix=h_linear_diag(km=k0,v=1,term_list=test_tun_terms,state_list=shell_basis)
    # end=time.time()
    # print(f'tun_matrix: {end-start}')
    # with open(f'tun_matrix_particles{particle_no}_shells{shells_used}_center_{center}','wb') as handle:
    #     pickle.dump(tun_matrix,handle)
    # start=time.time()
    # H_UHK=HK_sublattice(state_list=shell_basis,U=1)
    # print(H_UHK[:10,:10])
    # print(eyepreservation(shell_basis[0]))
    # end=time.time()
    # print(end-start)
    # start=time.time()
    # H_UHK=HK_N(state_list=shell_basis,pdic_n={0:p0,1:t0,2:p0,3:px},U_N=1)
    # print(H_UHK[:10,:10])
    # end=time.time()
    # print(end-start)
  
    # print(f'UHK construction time: {end-start}')
    # with open(f'H_UHK_particles{particle_no}_shells{shells_used}_center_{center}_UHK{UHK}','wb') as handle:
    #     pickle.dump(H_UHK,handle)

    # template_matrices_save=[f'data[1] is [kx_matrix,ky_matrix,kind_matrix_linear,tun_matrix, UHK_matrix] all at particle: {particle_no},shell: {shells_used}, center: {center}',[kx_matrix,ky_matrix,kind_matrix_linear,tun_matrix]]
    # template_name=f'template_matrices_particles{particle_no}_shells{shells_used}_center{center}'
    
    # with open(template_name,'wb') as handle:
    #     pickle.dump(template_matrices_save,handle)

    # template_matrices_save=[f'data[1] is [kx_matrix,ky_matrix,kind_matrix_linear,tun_matrix, UHK_matrix] all at particle: {particle_no},shell: {shells_used}, center: {center}',[kx_matrix,ky_matrix,kind_matrix_linear,tun_matrix,H_UHK]]
    #template_name=f'template_matrices_particles{particle_no}_shells{shells_used}_center{center}'
    
    # with open(template_name,'wb') as handle:
    # #     pickle.dump(template_matrices_save,handle)
    # with open(template_name,'rb') as f:
    #     templates_data=pickle.load(f)    


    # with open('cluster/H_UHK8_shells2_center_K_UHK1_old','rb') as f:
    #     test_hk=pickle.load(f)
    # # print(test_hk[:10,:10])
    # # print(eyepreservation(shell_basis[0]))

    
    #kx_matrix,ky_matrix,kind_matrix_linear,tun_matrix,H_UHK=templates_data[1][0],templates_data[1][1],templates_data[1][2],templates_data[1][3],templates_data[1][4]

    # for matrices in templates_data[1]:
    #     close_matrix=np.isclose(np.abs(matrices),np.zeros(matrices.shape))
    #     print(np.sum(close_matrix))

    #     print(f" Share non-zero: {1-np.sum(close_matrix)/(matrices.shape[0]*matrices.shape[1])}")
    
    # for matrix in templates_data[1]:
    #     print(f"Size of matrix is: {matrix.nbytes/(10**9)} GB, Size per entry is: {matrix.nbytes/(matrix.shape[0]*matrix.shape[1])}")
    
    # print("Size of template matrices:")
    # print(os.path.getsize(template_name)/(10**9))
    # exit()
    
    # L=4
    # test=np.zeros((L,L),dtype=complex)
    # t1=np.random.rand(L,L)+1j*np.random.rand(L,L)
    # print(f"Size of L={L}: {t1.nbytes}, size per byte={t1.nbytes/(t1.shape[0]*t1.shape[1])}")

    # sizes=[]
    # lengths=[10,100,1000,2000,5000,13000]
    # for l in lengths:
    #     t1=np.random.rand(l,l)+1j*np.random.rand(l,l)
    #     sizes.append(round(t1.nbytes/10**9,4))
    # plt.scatter(lengths,sizes)
    # plt.show()

    
    # #inspect_elements(H_UHK,shell_basis)



    # full_test=k0[0]*kx_matrix+k0[1]*ky_matrix+kind_matrix_linear+tun_matrix

    #non_int_templates=[(gkxw(w=1),kx_matrix),(gkyw(w=1),ky_matrix),(g0,kind_matrix_linear),(g0,tun_matrix)]
    
    
    # def h_linear_par(state_list,term_list,km):
    #     H0=np.zeros((len(state_list),len(state_list)),dtype=complex)
    #     for term in term_list:
    #         H0=H0+tpp_parallel(state_list=shell_basis,pauli_dic=term[0],prefactor=term[1](km[0],km[1]))
    #     H0=v*H0
    #     return H0
    # start=time.time()
    # kx_par=h_linear_par(state_list=shell_basis,term_list=linear_terms_kx,km=k0)
    # end=time.time()
    # print(f'parrallel same?: {np.allclose(kx_par,kx_matrix)}')
    
    # print(f'par time: {end-start}, regular time:250s')


    # dir_path_kx_1=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/1particles_2shells_centerK_matrices/kx_matrices_particles1_shells2_centerK'
    # filelist_kx_1=[dir_path_kx_1+'/'+f for f in os.listdir(dir_path_kx_1) if os.path.isfile(os.path.join(dir_path_kx_1, f))]
    # dir_path_ky_1=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/1particles_2shells_centerK_matrices/ky_matrices_particles1_shells2_centerK'
    # filelist_ky_1=[dir_path_ky_1+'/'+f for f in os.listdir(dir_path_ky_1) if os.path.isfile(os.path.join(dir_path_ky_1, f))]
    # dir_path_kind_1=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/1particles_2shells_centerK_matrices/kind_matrices_particles1_shells2_centerK'
    # filelist_kind_1=[dir_path_kind_1+'/'+f for f in os.listdir(dir_path_kind_1) if os.path.isfile(os.path.join(dir_path_kind_1, f))]
    # dir_path_tun_1=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/1particles_2shells_centerK_matrices/tun_matrices_particles1_shells2_centerK'
    # filelist_tun_1=[dir_path_tun_1+'/'+f for f in os.listdir(dir_path_tun_1) if os.path.isfile(os.path.join(dir_path_tun_1, f))]
    # HK_list_1=['/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/1particles_2shells_centerK_matrices/UHK_particles_1_shells_2_center_K.dill']

    # dir_path_kx=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/2particles_2shells_centerK_matrices/kxmatrix_particles2_shells2_centerK'
    # filelist_kx=[dir_path_kx+'/'+f for f in os.listdir(dir_path_kx) if os.path.isfile(os.path.join(dir_path_kx, f))]
    # dir_path_ky=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/2particles_2shells_centerK_matrices/kymatrix_particles2_shells2_centerK'
    # filelist_ky=[dir_path_ky+'/'+f for f in os.listdir(dir_path_ky) if os.path.isfile(os.path.join(dir_path_ky, f))]
    # dir_path_kind=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/2particles_2shells_centerK_matrices/kindmatrix_particles2_shells2_centerK'
    # filelist_kind=[dir_path_kind+'/'+f for f in os.listdir(dir_path_kind) if os.path.isfile(os.path.join(dir_path_kind, f))]
    # dir_path_tun=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/2particles_2shells_centerK_matrices/tun_matrices__particles2_shells2_centerK'
    # filelist_tun=[dir_path_tun+'/'+f for f in os.listdir(dir_path_tun) if os.path.isfile(os.path.join(dir_path_tun, f))]
    # HK_list=['/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/2particles_2shells_centerK_matrices/HK_particles2_shells2_centerK/HK_0.dill']

    dir_path_kx=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kxmatrices_particles4_shells2_centerK'
    filelist_kx=[dir_path_kx+'/'+f for f in os.listdir(dir_path_kx) if os.path.isfile(os.path.join(dir_path_kx, f))]
    dir_path_ky=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kymatrices_particles4_shells2_centerK'
    filelist_ky=[dir_path_ky+'/'+f for f in os.listdir(dir_path_ky) if os.path.isfile(os.path.join(dir_path_ky, f))]
    dir_path_kind=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kindmatrices_particles4_shells2_centerK'
    filelist_kind=[dir_path_kind+'/'+f for f in os.listdir(dir_path_kind) if os.path.isfile(os.path.join(dir_path_kind, f))]
    dir_path_tun=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/tunmatrices_particles4_shells2_centerK'
    filelist_tun=[dir_path_tun+'/'+f for f in os.listdir(dir_path_tun) if os.path.isfile(os.path.join(dir_path_tun, f))]
    HK_list=['/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCode/4particles_2shells_centerK_matrices/UHKrot_particles_4_shells_2_center_K.dill',
             '/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/UHK_N_taux_matrix_particles4_shells2_centerK/HK_N_taux_0.dill']
    

    

    def make_template_matrices(kx_list,ky_list,kind_list,tun_list,HK_list):
        kx_matrix=load_matrices(kx_list)
        ky_matrix=load_matrices(ky_list)
        kind_matrix=load_matrices(kind_list)
        kind_matrix=kind_matrix+load_matrices(tun_list)
        first=True
        for hkfile in HK_list:
            if first:
                with open(hkfile, 'rb') as file:
                    HK_matrix=pickle.load(file).form_matrix()
                first=False
            else:
                with open(hkfile, 'rb') as file:
                    HK_matrix=HK_matrix+pickle.load(file).form_matrix()
        non_int_templates=[(gkxw(w=1),kx_matrix),(gkyw(w=1),ky_matrix),(gw(w=1),kind_matrix)]
        return non_int_templates,HK_matrix
    
    
    
    
    def gen_Hk2(kx,ky):
        first=True
        for pair in non_int_templates:
            if first:
                H0=pair[0](kx,ky)*pair[1]
                first=False
            else:
                H0=H0+pair[0](kx,ky)*pair[1]
        H0=H0+HK_matrix
        return H0
    
    def gen_Hk_load(kx,ky,particles):
        shell_basis2=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=particles,nonlayer=testnonlayer,center=center)
        H0=np.zeros((len(shell_basis2),len(shell_basis2)),dtype=complex)
        for pair in non_int_templates:
            H0=H0+pair[0](kx,ky)*pair[1]
        H0=H0+HK_matrix
        return H0
    
    def gen_testH(kx,ky,UHK,state_list):
        H_linear=h_linear_diag(km=np.array([kx,ky]),v=v,term_list=test_linear_terms_nonzero,state_list=state_list)
        H_tun=h_linear_diag(km=np.array([kx,ky]),state_list=state_list,term_list=test_tun_terms,v=1)
        H_HK=HK_sublattice(state_list=shell_basis,U=UHK)
        return H_tun

    non_int_templates,HK_matrix=make_template_matrices(kx_list=filelist_kx,ky_list=filelist_ky,kind_list=filelist_kind,tun_list=filelist_tun,HK_list=HK_list)
    #non_int_templates,HK_matrix=make_template_matrices(kx_list=[],ky_list=[],kind_list=[],tun_list=filelist_tun,HK_list=HK_list)
    # basis2=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=2,nonlayer=testnonlayer,center=center)
    # test_load=gen_Hk_load(B[0],B[1],2)-HK_matrix
    # inspect_elements(test_load,basis2)
    # exit()

    def one_term(kx,ky,filelist,particles,kfunc):
        shell_basis2=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=particles,nonlayer=testnonlayer,center=center)
        
        print(f'load matrix')
        test_matrix=load_matrices(filelist)
        return kfunc(kx,ky)*test_matrix#gkxw(w=1)[kx,ky]*test_matrix


    # test_linear_terms=[({0:p0,1:qkx,2:px,3:p0},g0),({0:p0,1:qky,2:py,3:p0},g0),({0:p0,1:t0,2:px,3:p0},gkx),({0:p0,1:t0,2:py,3:p0},gky)]
    # test_linear_terms_nonzero=[({0:p0,1:qkx,2:px,3:p0},gw(np.cos(theta/2))),({0:p0,1:t0,2:px,3:p0},gkxw(np.cos(theta/2))),
    #                         ({0:pz,1:qkx,2:py,3:p0},gw(-np.sin(theta/2))),({0:pz,1:t0,2:py,3:p0},gkxw(-np.sin(theta/2))),
    #                         ({0:p0,1:qky,2:py,3:p0},gw(np.cos(theta/2))),({0:p0,1:t0,2:py,3:p0},gkyw(np.cos(theta/2))),
    #                         ({0:pz,1:qky,2:px,3:p0},gw(np.sin(theta/2))),({0:pz,1:t0,2:px,3:p0},gkyw(np.sin(theta/2))),
    #                         ]
    
    # linear_terms_kx=[({0:p0,1:t0,2:px,3:p0},gw(np.cos(theta/2))),({0:pz,1:t0,2:py,3:p0},gw(-np.sin(theta/2)))]
    # linear_terms_ky=[({0:p0,1:t0,2:py,3:p0},gw(np.cos(theta/2))),({0:pz,1:t0,2:px,3:p0},gw(np.sin(theta/2)))]
    # linear_terms_constant=[({0:p0,1:qkx,2:px,3:p0},gw(np.cos(theta/2))),({0:pz,1:qkx,2:py,3:p0},gw(-np.sin(theta/2))),
    #                         ({0:p0,1:qky,2:py,3:p0},gw(np.cos(theta/2))),({0:pz,1:qky,2:px,3:p0},gw(np.sin(theta/2)))]
    
    
    # linear_terms_kx_save=[({0:p0,1:t0,2:px,3:p0},gkx,[np.cos,g00],['theta','v'],[1/2,1]),({0:pz,1:t0,2:py,3:p0},gkx,[np.sin,g00],['theta','v'],[-1/2,1])]
    # linear_terms_ky_save=[({0:p0,1:t0,2:py,3:p0},gky,[np.cos,g00],['theta','v'],[1/2,1]),({0:pz,1:t0,2:px,3:p0},gky,[np.sin,g00],['theta','v'],[1/2,1])]
    # linear_terms_constant_save=[({0:p0,1:qkx,2:px,3:p0},g00,[np.cos,g00],['theta','v'],[1/2,1]),({0:pz,1:qkx,2:py,3:p0},g00,[np.sin,g00],['theta','v'],[-1/2,1]),
    #                             ({0:p0,1:qky,2:py,3:p0},g00,[np.cos,g00],['theta','v'],[1/2,1]),({0:pz,1:qky,2:px,3:p0},g00,[np.sin,g00],['theta','v'],[1/2,1])]
    
    # test_tun_terms=[({0:px,1:t1_plus,2:p0,3:p0},gw(w0)),({0:px,1:t1_minus,2:p0,3:p0},gw(w0)),({0:px,1:t1_plus,2:px,3:p0},gw(w1*np.cos(phi*(1-1)))),({0:px,1:t1_minus,2:px,3:p0},gw(w1*np.cos(phi*(1-1)))),
    #                 ({0:px,1:t2_plus,2:p0,3:p0},gw(w0)),({0:px,1:t2_plus,2:px,3:p0},gw(w1*np.cos(phi*(2-1)))),({0:px,1:t2_plus,2:py,3:p0},gw(w1*np.sin(phi*(2-1)))),
    #                 ({0:px,1:t2_minus,2:p0,3:p0},gw(w0)),({0:px,1:t2_minus,2:px,3:p0},gw(w1*np.cos(phi*(2-1)))),({0:px,1:t2_minus,2:py,3:p0},gw(w1*np.sin(phi*(2-1)))),
    #                 ({0:px,1:t3_plus,2:p0,3:p0},gw(w0)),({0:px,1:t3_plus,2:px,3:p0},gw(w1*np.cos(phi*(3-1)))),({0:px,1:t3_plus,2:py,3:p0},gw(w1*np.sin(phi*(3-1)))),
    #                 ({0:px,1:t3_minus,2:p0,3:p0},gw(w0)),({0:px,1:t3_minus,2:px,3:p0},gw(w1*np.cos(phi*(3-1)))),({0:px,1:t3_minus,2:py,3:p0},gw(w1*np.sin(phi*(3-1))))]
    # test_tun_terms_save=[({0:px,1:t1_plus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t1_minus,2:p0,3:p0},g0,[g00],['w0'],[1]),
    #                     ({0:px,1:t1_plus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(1-1)]),({0:px,1:t1_minus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(1-1)]),
    #                     ({0:px,1:t2_plus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t2_plus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(2-1)]),({0:px,1:t2_plus,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(2-1)]),
    #                     ({0:px,1:t2_minus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t2_minus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(2-1)]),({0:px,1:t2_minus,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(2-1)]),
    #                    ({0:px,1:t3_plus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t3_plus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(3-1)]),({0:px,1:t3_plus,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(3-1)]), 
    #                    ({0:px,1:t3_minus,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t3_minus,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(3-1)]),({0:px,1:t3_minus,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(3-1)])
    #                    ]
    
    # dir_path_kind=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCode/4particles_2shells_centerK_matrices/kindmatrices_particles4_shells2_centerK'
    # filelist_kind=[dir_path_kind+'/'+f for f in os.listdir(dir_path_kind) if os.path.isfile(os.path.join(dir_path_kind, f))]
    
    # testmat=load_matrices(filelist_kind)

    # print(testmat[:16,:16])
    # exit()

    ###############################TESTING TERM BY TERM !!!###############################

    ################################################################################
    # dir_ind='/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kindmatrices_particles4_shells2_centerK'
    # dir_kx='/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kxmatrices_particles4_shells2_centerK'
    # dir_ky='/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kymatrices_particles4_shells2_centerK'#'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCode/4particles_2shells_centerK_matrices/kymatrices_particles4_shells2_centerK'#
    # dir_tun='/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/tunmatrices_particles4_shells2_centerK'#'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/tunmatrices_particles4_shells2_centerK'#'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/2particles_2shells_centerK_matrices/tun_matrices__particles2_shells2_centerK''#'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCode/4particles_2shells_centerK_matrices/tun_matrices_particles4_shells2_centerK'#'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/tunmatrices_particles4_shells2_centerK'#'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/2particles_2shells_centerK_matrices/tun_matrices__particles2_shells2_centerK'
    # dir_rot='/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCode/4particles_2shells_centerK_matrices/UHKrot_particles_4_shells_2_center_K.dill'
    # k0=A
    # basis1=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=4,nonlayer=testnonlayer,center=center)
    # # dirpath_test=dir_tun
    from natsort import natsorted
    def load_mat(dirpath_test,matrixtype,k0):
        func_dic={'kx':gkxw(w=1),'ky':gkyw(w=1),'kind':gw(w=1),'tun':gw(w=1),'rot':gw(w=1)}
        if os.path.isfile(dirpath_test):
            filelist_test=[dirpath_test]
        elif os.path.isdir(dirpath_test):
            filelist_test=natsorted([dirpath_test+'/'+f for f in os.listdir(dirpath_test) if os.path.isfile(os.path.join(dirpath_test, f))])
        test_one_term=one_term(kx=k0[0],ky=k0[1],filelist=filelist_test,particles=4,kfunc=func_dic[matrixtype])
        
        return test_one_term
    # test_one_term=load_mat(dir_tun,'kx',k0)
    # test_one_term=test_one_term+load_mat(dir_ky,'ky',k0)
    # test_one_term=test_one_term+load_mat(dir_ind,'kind',k0)
    # test_one_term=test_one_term+load_mat(dir_tun,'tun',k0)
    # test_one_term=test_one_term+load_mat(dir_tun,'tun',k0)
    # test_one_term=test_one_term+load_mat(dir_rot,'rot',k0)
    # # print(inspect_elements(load_mat(dir_rot,'rot',k0),state_list=basis1))
    
    # # test_nonint=gen_Hk2(kx=B[0],ky=B[1])-HK_matrix
    # # print(np.allclose(test_one_term,test_nonint))
    

    
    # #test_diag=h_linear_diag(km=B,v=1,term_list=test_tun_terms,state_list=basis1)
    # #test_diag=h_linear_diag(km=B,v=v,term_list=list(np.array(test_linear_terms_nonzero)[np.array([1,3])]),state_list=basis1)
    # # test_diag=h_linear_diag(km=k0,v=v,term_list=test_linear_terms_nonzero,state_list=basis1)
    # # test_diag=test_diag+h_linear_diag(km=k0,v=1,term_list=test_tun_terms,state_list=basis1)
    # # print(np.allclose(test_one_term,test_diag))
    

    # def test_tensorprod(input_matrix,input_term_list,particles,k0):
    #     one_particle_basis=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=1,nonlayer=testnonlayer,center=center)
    #     #test_one_particle=h_linear_diag(km=k0,v=v,term_list=input_term_list,state_list=one_particle_basis)
    #     test_one_particle=h_linear_diag(km=k0,v=1,term_list=input_term_list,state_list=one_particle_basis)
        
    #     #test_one_particle2=gen_Hk(kx=k0[0],ky=k0[1],state_list=shell_basis_test,UHK=0)
    #     #print(np.allclose(test_one_particle,test_one_particle2))
    #     one_particle_energies=np.linalg.eigh(test_one_particle)[0]
    #     energy_combinations=list(itertools.combinations(one_particle_energies,particles))
    #     energy_combinations=np.sum(np.array(energy_combinations),axis=1)
    #     noninteigvals_1=np.linalg.eigh(input_matrix)[0]
        

    #     print(f'loaded matrix difference:{max(noninteigvals_1-np.sort(energy_combinations))}')
        


    #     return None
    #test_tensorprod(input_matrix=test_one_term,input_term_list=list(np.array(test_linear_terms_nonzero)[np.array([0,1,2,3,4,5,6,7])]),particles=4,k0=k0)
    
    # test_tensorprod(input_matrix=test_one_term,input_term_list=test_tun_terms,particles=4,k0=k0)
    # exit()


    #################################TESTING MATRICES VS EACH OTHER###############################

    # path_old=dir_path_kx_old
    # path_new=dir_path_kx
    # k0=A
    # old_mat=load_mat(dir_path_kind_old,'kind',k0)
    # new_mat=load_mat(dir_path_kind,'kind',k0)

    # print(f'matrices same? {np.allclose(old_mat,new_mat)}')

    # print('new mat')
    # print(old_mat[:4,:4])
    # print('old mat')
    # print(new_mat[:4,:4])
    # print('inspect old mat')
    # inspect_elements(old_mat,shell_basis,np.arange(0,5))
    # print(f'vksin(theta) = {v*kd*np.cos(theta/2)*(10.5/7)}')
    # exit()

    #################################TESTING MATRICES VS EACH OTHER###############################
    def test_rot_tensorprod(input_matrix,particles,k):
        one_particle_basis=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=1,nonlayer=testnonlayer,center=center)
        test_one_particle=h_linear_diag(km=k,v=v,term_list=test_linear_terms_nonzero,state_list=one_particle_basis)
        test_one_particle=test_one_particle+h_linear_diag(km=k,v=1,term_list=test_tun_terms,state_list=one_particle_basis)
        
        one_particle_energies=np.linalg.eigh(test_one_particle)[0]
        energy_combinations=list(itertools.combinations(one_particle_energies,particles))
        energy_combinations=np.sum(np.array(energy_combinations),axis=1)
        
        noninteigvals_1=np.linalg.eigh(input_matrix)[0]
        combined_eigvals=np.sort(energy_combinations)
        genhmatrix_eigvals=np.linalg.eigh(gen_Hk2(kx=k[0],ky=k[1]))[0]
        #print(max(noninteigvals_1-combined_eigvals))
        print(f' k is {k}')
        print(f'first 10 eigvals one term: {noninteigvals_1[:10]}')
        print(f'Energy combination {np.sort(energy_combinations)[:10]}')
        print(f'first 10 eigvals gen_Hk2: {genhmatrix_eigvals[:10]}')
        print(f'Single particle energies: {one_particle_energies}')
        print(f'sum of first four distinct single particle energies: {np.sum(one_particle_energies[:8])/2}')


        
        


        return None
    
    #test_rot_tensorprod(input_matrix=test_one_term,particles=4,k=k0)

    #exit()
    ###############################TESTING TERM BY TERM !!!###############################

    #############################################################################################
    # exit()

    # print(np.allclose(test_one_term,np.zeros(test_one_term.shape,dtype=complex)))
    
    # print('diag term')
    # print(test_diag[:16,:16])
    # print('Loaded term')
    # print(test_one_term[:16,:16])

    











    def check_nonint_tensorproduct(k0,particles):

        test_basis=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=particles,nonlayer=testnonlayer,center=center)
        test_load=gen_Hk2(B[0],B[1])-HK_matrix
        test_diag=h_linear_diag(km=B,v=v,term_list=test_linear_terms_nonzero,state_list=basis1)
        test_diag=test_diag+h_linear_diag(km=B,v=1,term_list=test_tun_terms,state_list=basis1)
        print(f'Loaded and constructed same: {np.allclose(test_load,test_diag)}')
        # print(np.sum(np.isclose(test_load,np.zeros(test_load.shape,dtype=complex))))
        # print(np.sum(np.isclose(test_diag,np.zeros(test_load.shape,dtype=complex))))
        # print(np.sum(np.isclose(test_diag-test_load,np.zeros(test_load.shape,dtype=complex))))
        # print(np.sum(np.isclose(test_diag-test_load,np.zeros(test_load.shape,dtype=complex)))-np.sum(np.isclose(test_diag,np.zeros(test_load.shape,dtype=complex))))

            
            
        
        one_particle_basis=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=1,nonlayer=testnonlayer,center=center)
        test_one_particle=h_linear_diag(km=B,v=v,term_list=test_linear_terms_nonzero,state_list=one_particle_basis)
        test_one_particle=test_one_particle+h_linear_diag(km=B,v=1,term_list=test_tun_terms,state_list=one_particle_basis)
        
        #test_one_particle2=gen_Hk(kx=k0[0],ky=k0[1],state_list=shell_basis_test,UHK=0)
        #print(np.allclose(test_one_particle,test_one_particle2))
        one_particle_energies=np.linalg.eigh(test_one_particle)[0]
        energy_combinations=list(itertools.combinations(one_particle_energies,particles))
        energy_combinations=np.sum(np.array(energy_combinations),axis=1)
        noninteigvals_1=np.linalg.eigh(test_load)[0]
        noninteigvals_2=np.linalg.eigh(test_diag)[0]

        print(f'loaded matrix difference:{max(noninteigvals_1-np.sort(energy_combinations))}')
        print(f'constructed matrix difference:{max(noninteigvals_2-np.sort(energy_combinations))}')
        
        exit()
        
        
        return None


    def gen_Hk3(kx,ky,U):
        H0=np.zeros((len(shell_basis),len(shell_basis)),dtype=complex)
        for pair in non_int_templates:
            H0=H0+pair[0](kx,ky)*pair[1]
        H0=H0+H_UHK*(U/(2*v0))
        return H0
    
    def check_equal(m1,m2):
        real=np.allclose(m1+np.conjugate(m1),m2+np.conjugate(m2))
        imag=np.allclose(m1-np.conjugate(m1),m2-np.conjugate(m2))
        return (real and imag)
    
    #Let's try to write a function which plots out the Hamiltonian for various U

    def one_k_plot(k0,U_values):

        def gen_Hk3(kx,ky,U):
            H0=np.zeros((len(shell_basis),len(shell_basis)),dtype=complex)
            for pair in non_int_templates:
                H0=H0+pair[0](kx,ky)*pair[1]
            H0=H0+(UHK/(2*v0))*U
            return H0
        energies=[]
        states=[]
        for U in U_values:
            H1=gen_Hk3(kx=k0[0],ky=k0[1],U=U)
            w,v=np.linalg.eigh(H1)
            energies.append(w[:6])
            states.append(v[:,:6])
        
        # Define a specific color for each possible number of repeats
        color_map = {
            0: 'violet',  # Unique scenario
            1: 'blue',    # One repeat or close
            2: 'green',   # Two repeats or close
            3: 'yellow',  # Three repeats or close
            4: 'orange',  # Four repeats or close
            5: 'red',     # Five repeats or close (all the same or close)
        }

        # Use np.isclose with a specified tolerance to find 'close' repeats
        def calculate_close_repeats(y_values, atol=0.5, rtol=0):
            close_counts = 0
            considered = np.zeros(len(y_values), dtype=bool)
            for i in range(len(y_values)):
                if not considered[i]:
                    # Find values close to y_values[i] that haven't been considered yet
                    close_to_i = np.isclose(y_values[i], y_values, atol=atol, rtol=rtol) & ~considered
                    if np.sum(close_to_i) > 1:  # If more than itself
                        close_counts += np.sum(close_to_i) - 1  # Count repeats, exclude the value itself
                        considered |= close_to_i  # Mark as considered
            return min(close_counts, 5)

        # Plotting
        plt.figure(figsize=(10, 6))
        for x, ys in zip(U_values, energies):
            num_repeats = calculate_close_repeats(ys)
            color = color_map[num_repeats]
            for y in ys:
                plt.scatter(x, y, color=color)
        plt.show()
    colors=pathdiag.colors

    def one_k_plot2(k0,U_values,state_list):
        deg_values=[]
        energies=[]
        int_energies=[]
        for U in U_values:
            H1=gen_Hk3(kx=k0[0],ky=k0[1],U=U)
            H2=gen_Hk3(kx=k0[0],ky=k0[1],U=1)
            Uproj=H2-gen_Hk3(kx=k0[0],ky=k0[1],U=0)
            eigvals,eigstates=np.linalg.eigh(H1)
            exp=np.diag(np.dot(eigstates.conj().T, np.dot(Uproj, eigstates)))
            deg_values.append(pathdiag.degcounter(eigvals,acc=8))
            energies.append(eigvals)
            int_energies.append(np.real(exp))
            print(f'exp shape{exp.shape}')
            print([i for i in range(len(exp)) if np.imag(i)>10**-8])
            
        
        fig, axs = plt.subplots(1,2)
        linesplotted=16
        linescolored=16
        linesinlabel=16
        eig_dict_loop={}
        print(f'len energies {len(energies[0])}')
        for k in range(0,len(energies[0])):
            eig_dict_loop[k]=([y[k] for y in energies],[j[k] for j in deg_values])
        print(f'eig_dict_loop keys: {list(eig_dict_loop.keys())[:10]}')
        
        for j in eig_dict_loop.keys():#The end-point sets how many lines you plot.
            #colorsmap=[colors[x] for x in eig_dict_loop[j][1]]
            colorsmap=[]
            for x in range(0,len(eig_dict_loop[j][1])):
                if j<linesplotted:#This is to set how many lines you colors #UHK!#1.4 for not subtracted GS
                    if x<linescolored:#eig_dict_loop[j][1][x]
                        colorsmap.append(colors[eig_dict_loop[j][1][x]])
                    else:
                        colorsmap.append('dimgray')
            axs[0].scatter(U_values,eig_dict_loop[j][0],s=30,c=colorsmap)
            axs[0].set_ylim([-2, 0])
            axs[0].set_title('First ten energies')
            axs[0].set_ylabel('E_4')
            axs[0].set_xlabel('Orbital HK U')
        eig_dict_loop1={}
        print(f'len energies {len(energies[0])}')
        for k in range(0,len(energies[0])):
            eig_dict_loop1[k]=([y[k] for y in int_energies],[j[k] for j in deg_values])
        print(f'eig_dict_loop keys: {list(eig_dict_loop.keys())[:10]}')
        for j in eig_dict_loop1.keys():#The end-point sets how many lines you plot.
            #colorsmap=[colors[x] for x in eig_dict_loop[j][1]]
            colorsmap=[]
            for x in range(0,len(eig_dict_loop1[j][1])):
                if j<linesplotted:#This is to set how many lines you colors #UHK!#1.4 for not subtracted GS
                    if x<linescolored:#eig_dict_loop[j][1][x]
                        colorsmap.append(colors[eig_dict_loop1[j][1][x]])
                    else:
                        colorsmap.append('dimgray')
            if j<linesplotted:
                axs[1].scatter(U_values,eig_dict_loop1[j][0],s=30,c=colorsmap)
                axs[1].set_title('Pair number expectation')
                axs[1].set_ylabel('Pair expectation')
                axs[1].set_xlabel('Orbital HK U')
        custom_lines=[]
        names=[]
        for i in colors.keys():
            if i<=linesinlabel:#This sets how many lines you have in the deg label
                custom_lines.append(Line2D([0], [0], color=colors[i], lw=4))
                names.append(i)
        leg = plt.figlegend(custom_lines,names, loc=(0.87, 0.11),title="Degeneracy",fontsize=20,title_fontsize=20)
        #leg = plt.figlegend(custom_lines,names, loc=(0.87, 0.11),title="Degeneracy",fontsize=20,title_fontsize=20)#0.464
        fig.subplots_adjust(right=0.85)
        plt.show()
        
    #one_k_plot2(k0=B,U_values=np.linspace(0,10,20),state_list=shell_basis)
    
    def one_particle_amp_plot(k0,state_list,number):
        H_linear=h_linear_diag(km=np.array([k0[0],k0[1]]),v=v,term_list=test_linear_terms_nonzero,state_list=state_list)
        H_tun=h_linear_diag(km=np.array([k0[0],k0[1]]),state_list=state_list,term_list=test_tun_terms,v=1)
        H=H_linear+H_tun
        eigvals,eigvecs=np.linalg.eigh(H)
        show_vecs=eigvecs[:,:number]
        labels=[eyepreservation(state) for state in state_list]
        fig,axs=plt.subplots(2,number//4)
        plots=0
        for e_ind in range(number):
            if e_ind%2==0:
                print(e_ind)
                print(e_ind//(number//2),(e_ind//2)%(number//4))
                axs[e_ind//(number//2),(e_ind//2)%(number//4)].scatter(np.square(np.abs(show_vecs[:,e_ind])),labels,color='b')
                axs[(e_ind)//(number//2),(e_ind//2)%(number//4)].set_title(f'E={round(eigvals[e_ind],3), round(eigvals[e_ind+1],3)}')
            else:
                axs[(e_ind-1)//(number//2),((e_ind-1)//2)%(number//4)].scatter(np.square(np.abs(show_vecs[:,e_ind])),labels,color='r')
            axs[e_ind//(number//2),e_ind%(number//4)].tick_params(axis='x', labelrotation=90)
                
                
        
        fig.tight_layout()
        plt.show()




    
    def one_particle_exp_plot(k0,state_list,number):
        H_linear=h_linear_diag(km=np.array([k0[0],k0[1]]),v=v,term_list=test_linear_terms_nonzero,state_list=state_list)
        H_tun=h_linear_diag(km=np.array([k0[0],k0[1]]),state_list=state_list,term_list=test_tun_terms,v=1)
        H=H_linear+H_tun
        eigvals,eigvecs=np.linalg.eigh(H)
        show_vecs=eigvecs[:,:number]
        labels=[eyepreservation(state)[:len(eyepreservation(state))-2] for state in state_list[::2]]
        spin_exp_operator=tpp(state_list=state_list,pauli_dic={0:p0,1:t0,2:p0,3:pz},prefactor=1)
        tau_exp_operator=tpp(state_list=state_list,pauli_dic={0:p0,1:t0,2:pz,3:p0},prefactor=1)
        momentum_exp_operator=tpp(state_list=state_list,pauli_dic={0:p0,1:tz,2:p0,3:p0},prefactor=1)
        operators=[spin_exp_operator,tau_exp_operator,momentum_exp_operator]
        names=['Spin Expectation','Tau Expectation','Momentum Expectation']
        expectations=[]
        for op in operators:
            exp=np.sum(np.conjugate(eigvecs)*np.dot(op,eigvecs),axis=0)
            expectations.append(np.abs(exp))
        grouped_expectations=[]
        group_spin=True
        if group_spin:
            for exp in expectations:
                exp=exp.reshape(-1,2)
                exp=exp.mean(axis=1)
                grouped_expectations.append(exp)
        else:
            grouped_expectations=expectations

        fig=make_subplots(rows=1,cols=len(operators),subplot_titles=names)
        for i in range(len(operators)):
            fig.add_trace(go.Scatter(x=grouped_expectations[i],y=eigvals[::2],mode='markers',marker=dict(size=10),name=names[i]),row=1,col=i+1)
            if i!=2:
                fig.update_xaxes(title_text='Expectation value',range=[0,1],row=1,col=i+1)
            else:
                fig.update_xaxes(title_text='Expectation value',range=[0,3],row=1,col=i+1)
        
                
                
        
        
        return fig
    
    # shells1=generate_shell_basis_gamma(shell_count=shells_used,q_vecs=tqs,number_of_particles=1,nonlayer=testnonlayer,center=center)
    # one_particle_amp_plot(k0=A,state_list=shells1,number=8)
    #one_particle_exp_plot(k0=A+1*(B-A)/10,state_list=shells1,number=8).show()
    # exit()

        
    
    #You should also take the non-interacting state and plot out the real and complex amplitudes for each of the sixteen basis states for the four lowest energy levels
            
    
    def test(U):
        htest=gen_Hk3(kx=B[0],ky=B[1],U=U)
        Uproj=HK_sublattice(state_list=shell_basis,U=1)
        eigvals,eigstates=np.linalg.eigh(htest)
        exp=np.diag(np.dot(eigstates.conj().T, np.dot(Uproj, eigstates)))
        return eigvals,exp
    
    # plt.scatter(range(5),exp[:5])
    # plt.show()

    # fig=make_subplots(rows=3,cols=2)
    # vals,pairs=test(U=0)
    # fig.add_trace(go.Scatter(x=np.linspace(0,10,10),y=np.real(vals[:10])),row=1,col=1)
    # fig.add_trace(go.Scatter(x=np.linspace(0,10,10),y=np.real(pairs[:10])),row=1,col=2)
    # vals,pairs=test(U=1)
    # fig.add_trace(go.Scatter(x=np.linspace(0,10,10),y=np.real(vals[:10])),row=2,col=1)
    # fig.add_trace(go.Scatter(x=np.linspace(0,10,10),y=np.real(pairs[:10])),row=2,col=2)
    # vals,pairs=test(U=10)
    # fig.add_trace(go.Scatter(x=np.linspace(0,10,10),y=np.real(vals[:10])),row=3,col=1)
    # fig.add_trace(go.Scatter(x=np.linspace(0,10,10),y=np.real(pairs[:10])),row=3,col=2)

    # fig.show()
    
    #one_k_plot(k0=B,U_values=np.linspace(0,10,5))
    


    # random_ks=[np.pi*np.random.rand(1,2)[0] for i in range(2)]
    # values=[check_equal(m1=gen_Hk(k[0],k[1]),m2=gen_Hk2(kx=k[0],ky=k[1],templates=templates)) for k in random_ks]
    # print(sum(values)/len(values))
    

    # test_h=gen_Hk2(kx=1.34,ky=0.004)
    # print(test_h.shape)
    # start=time.time()
    # eigvals,eigvecs=np.linalg.eigh(test_h)
    # end=time.time()
    # print(f'Diag time: {end-start}')
    

    
    #print(shell_basis[0].particle_dic[1].dof_dic.keys())
    # start=time.time()
    # testh0=h_linear_diag(km=np.array([0.13,0.222]),v=1,term_list=test_linear_terms_nonzero,state_list=shell_basis)
    # end=time.time()
    # print(f'Make linear H: {start-end}s')
    # start=time.time()
    # testh1=h_linear_diag(km=-A,v=1,term_list=test_tun_terms,state_list=shell_basis)
    # end=time.time()
    # print(f'Make tun H: {start-end}s')
    # start=time.time()
    # np.linalg.eigh(testh1+testh0)
    # end=time.time()
    # print(f'Diag: {start-end}s')
    # exit()
    # print(f'A {A/kd}')
    # inspect_elements(testh0,shell_basis)
    # print(np.linalg.eigh(testh0)[0])

    
    # print(testh0)
    start='KM'
    end='GammaM'

    k0=B
    t1=gen_Hk2(k0[0],k0[1])
    eigvals,eigvecs=np.linalg.eigh(t1)
    print(len(eigvecs))
    
    test12 = np.conjugate(eigvecs[:,:4]).T @ HK_matrix @ eigvecs[:,:4]
    print(test12.shape)
    print(np.array_str(test12, precision=5, suppress_small=True))
    print(np.linalg.eigh(test12)[0])

    hexp=np.conjugate(eigvecs)* HK_matrix @ eigvecs
    s0=hexp[np.abs(hexp)>10**-8]
    print(sum(np.abs(hexp[:,0])>10**-8))
    indices=np.where(np.abs(hexp[:,0])>10**-12)[0]
    for i in indices:
        print(eyepreservation(shell_basis[i]),f' amp: {np.abs(hexp[i,0])}')    
    print('EIGSTATES____________')
    indices2=np.where(np.abs(eigvecs[:,0])>10**-12)[0]
    # for i in indices2:
    #     print(eyepreservation(shell_basis[i]),f' amp: {np.abs(eigvecs[i,0])}')    
    
    for i in indices2:
        testwf=np.zeros(len(shell_basis),dtype=complex)
        testwf[i]=1
        reswf=np.dot(HK_matrix,testwf)
        #assumes state diagonal
        print('input state',eyepreservation(shell_basis[i]),f'HK value: {reswf[i]}', f' amp: {np.abs(eigvecs[i,0])}')
    
    
    print(hexp.shape)

    
    # # print(test12)
    # for i in range(len(eigvecs[:,0])):
    #     if np.abs(eigvecs[i,0])>10**-8:
            
            
    # exit()

    
    print(f'first ten eigvals: {eigvals[:10]}')
    print(f'non zero states in ground states')
    nzindices2=np.where(np.abs(eigvecs[:,0]) > 10**-8)[0]
    for i in nzindices2:
        print(eyepreservation(shell_basis[i]),f' amp: {np.abs(eigvecs[i,0])}')
    
    print(f'non zero states in second state')
    nzindices2=np.where(np.abs(eigvecs[:,1]) > 10**-8)[0]
    for i in nzindices2:
        print(eyepreservation(shell_basis[i]),f' amp: {np.abs(eigvecs[i,0])}')
    # print(np.conjugate(eigvecs[:,0]))
    
    # print(f'making sz')
    sz=tpp(state_list=shell_basis,pauli_dic={0:p0,1:t0,2:p0,3:pz},prefactor=1)
    tauz=tpp(state_list=shell_basis,pauli_dic={0:p0,1:t0,2:pz,3:p0},prefactor=1)
    spin_exp=np.conjugate(eigvecs)*np.dot(sz,eigvecs)
    spin2_exp=np.conjugate(np.dot(sz,eigvecs))*np.dot(sz,eigvecs)
    spin_exp=np.sum(spin_exp,axis=0)
    spin2_exp=np.sum(spin2_exp,axis=0)

    tau_exp=np.conjugate(eigvecs)*np.dot(tauz,eigvecs)
    tau2_exp=np.conjugate(np.dot(tauz,eigvecs))*np.dot(tauz,eigvecs)
    tau_exp=np.sum(tau_exp,axis=0)
    tau2_exp=np.sum(tau2_exp,axis=0)



    print(f'spin exp first 10 {spin_exp[:10]}')
    print(f'spin2 exp first 10 {spin2_exp[:10]}')
    print(f'tau exp first 10 {tau_exp[:10]}')
    print(f'tau2 exp first 10 {tau2_exp[:10]}')
    
    
    #sx=tpp(state_list=shell_basis,pauli_dic={0:p0,1:t0,2:p0,3:px},prefactor=1)
    #sy=tpp(state_list=shell_basis,pauli_dic={0:p0,1:t0,2:p0,3:px},prefactor=1)
    #s2=np.dot(sz,sz)+np.dot(sy,sy)+np.dot(sx,sx)
    #tauz=tpp(state_list=shell_basis,pauli_dic={0:p0,1:t0,2:pz,3:p0},prefactor=1)

    commut=np.dot(sz,t1)-np.dot(t1,sz)
    print(f' spin commutes H: {np.allclose(commut,np.zeros(commut[0].shape))}')
    commut_HK=np.dot(sz,HK_matrix)-np.dot(HK_matrix,sz)
    print(f' spin commutes H: {np.allclose(commut,np.zeros(commut[0].shape))}')
    # spin_exp=np.dot(sz,eigvecs)
    # spin2_exp=spin_exp**2
    # spin_exp=np.einsum('ij,ij->i',np.conjugate(eigvecs),spin_exp)
    
    # spin2_exp=np.einsum('ij,ij->i',np.conjugate(eigvecs),spin2_exp)

    # #tau_exp=np.dot(tauz,eigvecs)
    # #tau_exp=np.einsum('ij,ij->i',np.conjugate(eigvecs),tau_exp)
    # print(f'spin exp first 10 {spin_exp[:10]}')
    # print(f'spin2 exp first 10 {spin2_exp[:10]}')
    # #print(f'tau exp first 10 {tau_exp[:10]}')
    
    



    # print(var_file_reversed_Moire[tuple(KM)])
    # print(KM)
    # print(MM)
    # exit()
    # pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['KM'],vars_dic_Moire['GammaM']],kpoints=kpoints,generate_Hk=gen_Hk2,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    # pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['GammaM'],vars_dic_Moire['MM']],kpoints=kpoints,generate_Hk=gen_Hk2,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    # pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['MM'],vars_dic_Moire['KM']],kpoints=kpoints,generate_Hk=gen_Hk2,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)

    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['A'],vars_dic_Moire['B']],kpoints=kpoints,generate_Hk=gen_Hk2,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['B'],vars_dic_Moire['C']],kpoints=kpoints,generate_Hk=gen_Hk2,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['C'],vars_dic_Moire['D']],kpoints=kpoints,generate_Hk=gen_Hk2,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['D'],vars_dic_Moire['A']],kpoints=kpoints,generate_Hk=gen_Hk2,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)

    # pathdiag.chained_path_plot_link_doubleocc(path_list=[vars_dic_Moire['A'],vars_dic_Moire['B']],kpoints=kpoints,generate_Hk=gen_Hk,HKterm=HK_sublattice2(state_list=shell_basis,U=1),UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    # pathdiag.chained_path_plot_link_doubleocc(path_list=[vars_dic_Moire['B'],vars_dic_Moire['C']],kpoints=kpoints,generate_Hk=gen_Hk,HKterm=HK_sublattice2(state_list=shell_basis,U=1),UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    # pathdiag.chained_path_plot_link_doubleocc(path_list=[vars_dic_Moire['C'],vars_dic_Moire['D']],kpoints=kpoints,generate_Hk=gen_Hk,UHK=UHK,HKterm=HK_sublattice2(state_list=shell_basis,U=1),mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    # pathdiag.chained_path_plot_link_doubleocc(path_list=[vars_dic_Moire['D'],vars_dic_Moire['A']],kpoints=kpoints,generate_Hk=gen_Hk,UHK=UHK,HKterm=HK_sublattice2(state_list=shell_basis,U=1),mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)

    directory = os.fsencode("/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/pathdata5")
    dstr="/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/pathdata5"
    mu_shift1=UHK/2
    params=f'UHKrot{UHK_rot}UHKtau{UHK_N_p0t0pxp0}'
    #plot.chained_path_plot(path_lists=[[KM,GammaM,MM,KM]],kpoints=str(kpoints),directory=directory,dstr=dstr,mu_shift=mu_shift1,params=params,variable='kcent',theta=f'theta{thetadeg}')
    plot_save.chained_path_plot(path_lists=[[A,B,C,D,A]],kpoints=str(kpoints),directory=directory,dstr=dstr,mu_shift=mu_shift1,params=params,variable=f'{particle_no}particle',theta=f'theta{thetadeg}',linesplotted=16,plotcutoff=16)
    plot.chained_path_plot(path_lists=[[A,B,C,D,A]],kpoints=str(kpoints),directory=directory,dstr=dstr,mu_shift=mu_shift1,params=params,variable=f'{particle_no}particle',theta=f'theta{thetadeg}')    
    