import sys
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire')
import numpy as np

import Code.make_templates as makeH
import dill

class symmetry():
    def __init__(self,space_group_generators_dic,non_unitary_generators_dic):
        self.space_group_generators=space_group_generators_dic
        self.space_group_generators=non_unitary_generators_dic

class time_reversal():
	def __init__(self,state_list,pauli_dic,coeff,element_list):
		self.state_list=state_list
		self.pauli_dic=pauli_dic
		self.coeff=coeff
		self.element_list=element_list
		# self.conjugate=len(state_list[0].particle_dic.keys())%2
		self.unitary_matrix=tpp_tensor_tr(state_list=state_list,pauli_dic=pauli_dic,prefactor=coeff)
	def vector_act(self,vector):
		if (len(self.state_list[0].particle_dic.keys())%2)==0:
			return np.dot(self.unitary_matrix,vector)
		elif (len(self.state_list[0].particle_dic.keys())%2)==1:
			return np.dot(self.unitary_matrix,np.conjugate(vector))

class symmetry_operator():
	def __init__(self,state_list,pauli_dic,coeff,TR):
		self.state_list=state_list
		self.pauli_dic=pauli_dic
		self.coeff=coeff
		self.TR=TR
		# self.conjugate=len(state_list[0].particle_dic.keys())%2
		self.unitary_matrix=tpp_tensor_tr(state_list=state_list,pauli_dic=pauli_dic,prefactor=coeff)
	def linear_act(self,linear):#i.e could be vector or matrix
		if self.TR:
			if (len(self.state_list[0].particle_dic.keys())%2)==0:
				return np.dot(self.unitary_matrix,linear)
			elif (len(self.state_list[0].particle_dic.keys())%2)==1:
				return np.dot(self.unitary_matrix,np.conjugate(linear))
		else:
			return np.dot(self.unitary_matrix,linear)

class symmetry_operator2():
	def __init__(self,state_list,generator_list):
		self.state_list=state_list
		self.generator_list=generator_list

	def unitary_matrix(self):
		matrix=np.identity(n=len(self.state_list),dtype=complex)
		#Assumes generator list is in the form (pauli_dic,prefactor)
		
		for generator in list(reversed(self.generator_list)):#the reverse so that the first elements act first!
			if generator[2]:
				matrix=np.conjugate(matrix)
				generator_unitary=tpp_tensor_tr(state_list=self.state_list,pauli_dic=generator[0],prefactor=generator[1])
				matrix=np.dot(generator_unitary,matrix)
			else:
				generator_unitary=tpp_tensor(state_list=self.state_list,pauli_dic=generator[0],prefactor=generator[1])
				matrix=np.dot(generator_unitary,matrix)
		return matrix
	def is_tr(self):
		TR_list=[generators[2] for generators in self.generator_list]
		return bool(sum(TR_list)%2)
	
class symmetry_operator_saved():
	def __init__(self,state_list,generator_list):
		self.state_list=state_list
		self.generator_list=generator_list
		self.saved_unitary_matrix=None

	def unitary_matrix(self):
		if self.saved_unitary_matrix is not None:
			return self.saved_unitary_matrix
		else:
			matrix=np.identity(n=len(self.state_list),dtype=complex)
			#Assumes generator list is in the form (pauli_dic,prefactor)
			for generator in list(reversed(self.generator_list)):#the reverse so that the first elements act first!
				if generator[2]:
					matrix=np.conjugate(matrix)
					generator_unitary=tpp_tensor_tr(state_list=self.state_list,pauli_dic=generator[0],prefactor=generator[1])
					matrix=np.dot(generator_unitary,matrix)
				else:
					generator_unitary=tpp_tensor(state_list=self.state_list,pauli_dic=generator[0],prefactor=generator[1])
					matrix=np.dot(generator_unitary,matrix)
			self.saved_unitary_matrix=matrix
			return matrix
			
	def is_tr(self):
		TR_list=[generators[2] for generators in self.generator_list]
		return bool(sum(TR_list)%2)
		

	def linear_act(self,linear):#i.e could be vector or matrix
		if self.is_tr():
			if (len(self.state_list[0].particle_dic.keys())%2)==0:
				return np.dot(self.unitary_matrix(),linear)
			elif (len(self.state_list[0].particle_dic.keys())%2)==1:
				return np.dot(self.unitary_matrix(),np.conjugate(linear))
		else:
			return np.dot(self.unitary_matrix(),linear)
		
		
		




def tpp_tensor(state_list,pauli_dic,prefactor):
	H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
	particle_dic_temp1={}#Idea is to constantly overwrite
	dof_dic_temp={}
	for state in state_list:
		result_list=[]
		coeff=1
		for k in state.particle_dic.keys():
			for l in range(0,len(pauli_dic.keys())):
				dof_dic_temp[l]=pauli_dic[l](state.particle_dic[k].dof_dic[l])[1]
				coeff=coeff*pauli_dic[l](state.particle_dic[k].dof_dic[l])[0]
			coeff=prefactor*coeff
			particlen=makeH.particle(dof_dic=dict(dof_dic_temp))
			particle_dic_temp1[k]=particlen
		new_state=makeH.basis(particle_dic=dict(particle_dic_temp1))
		result=(coeff,new_state)
		result_list.append(result)
	#2.6 Now I need to map the states into some standard Hamiltonian ordering.
	#I need to do an unordered match of the state to a state in the basis, and then I need to count the number of swaps
#Have to do this because, since I've introduced the basis as a class - seperate instances of the class count as different objects...    
		for result in result_list:
			for i in state_list:
				if makeH.basis.eqnoorder(i,result[1]):
					temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
					swapcount=makeH.basis.swaps(result[1],i)
					position1=(state_list.index(i),state_list.index(state))
					temp_H[position1]=((-1)**swapcount)*1*result[0]
					H1=H1+temp_H
            # position1=(ordered_basis.index(basis1),ordered_basis.index(state)) #in the matrix, row index is final state, col. index is final state
            #the SO coupling brings imaginary terms in, unfortunately...
            # bit of a hack but I'm just using the pair to seperate out the coefficent from the state label
            
	return H1
            
def tpp_tensor_tr(state_list,pauli_dic,prefactor):
	H1=np.zeros((len(state_list),len(state_list)),dtype=complex)
	particle_dic_temp1={}#Idea is to constantly overwrite
	dof_dic_temp={}
	for state in state_list:
		result_list=[]
		coeff=1
		for k in state.particle_dic.keys():
			coeff=np.conjugate(coeff)
			for l in range(len(pauli_dic.keys())):
				dof_dic_temp[l]=pauli_dic[l](state.particle_dic[k].dof_dic[l])[1]
				coeff=coeff*pauli_dic[l](state.particle_dic[k].dof_dic[l])[0]
			coeff=coeff*prefactor
			particlen=makeH.particle(dof_dic=dict(dof_dic_temp))
			particle_dic_temp1[k]=particlen
		new_state=makeH.basis(particle_dic=dict(particle_dic_temp1))
		result=(coeff,new_state)
		result_list.append(result)
	#2.6 Now I need to map the states into some standard Hamiltonian ordering.
	#I need to do an unordered match of the state to a state in the basis, and then I need to count the number of swaps
#Have to do this because, since I've introduced the basis as a class - seperate instances of the class count as different objects...    
		for result in result_list:
			for i in state_list:
				if makeH.basis.eqnoorder(i,result[1]):
					temp_H=np.zeros((len(state_list),len(state_list)),dtype=complex)
					swapcount=makeH.basis.swaps(result[1],i)
					position1=(state_list.index(i),state_list.index(state))
					temp_H[position1]=((-1)**swapcount)*result[0]
					H1=H1+temp_H
            # position1=(ordered_basis.index(basis1),ordered_basis.index(state)) #in the matrix, row index is final state, col. index is final state
            #the SO coupling brings imaginary terms in, unfortunately...
            # bit of a hack but I'm just using the pair to seperate out the coefficent from the state label
	return H1

# def pexpz3(sigma):
#     sigma_exp=(-1)**(sigma)
#     return (np.exp(1j*np.pi*sigma_exp/3),sigma)


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Code import make_templates as makeH
from Code.variables import *

testnonlayer={'sublattice':2,'spin':2}
#tqs=[np.array([-1,-1]),np.array([1,0]),np.array([0,1])]
#shell_basis=makeH.generate_shell_basis(shell_count=shells_used,q_vecs=tqs,number_of_particles=particle_no,nonlayer=testnonlayer)
shell_basis=makeH.shell_basis
#Let's try to define the single particle generators
def qc3(qpair):
	q2=qpair[0]
	q3=qpair[1]
	return (1,(-q3,q2-q3))
def qc0(qpair):
	q2=qpair[0]
	q3=qpair[1]
	return (1,(q2,q3))

c3z=({0:makeH.p0,1:qc3,2:makeH.p0,3:makeH.pexpz3},1,False)
c2zT=({0:makeH.p0,1:qc0,2:makeH.px,3:makeH.py},1j,True)
ctest2=({0:makeH.p0,1:qc3,2:makeH.pexpz3,3:makeH.pexpz6},1,False)
ctest3=({0:makeH.p0,1:qc3,2:makeH.pexpz3,3:makeH.p0},1,False)
ctest3vg=({0:makeH.p0,1:qc0,2:makeH.pz,3:makeH.p0},1,False)
ctauz=({0:makeH.p0,1:qc0,2:makeH.p0,3:makeH.pz},1,False)
six_prime_unitary={'C3z':[c3z],'C3z^2':2*[c3z],'C3z^3':3*[c3z],'C3z^4':4*[c3z],'C3z^5':5*[c3z],'1':6*[c3z]}
six_prime_antiunitary={'C2zT':[c2zT],'C2zTC3z':[c2zT]+[c3z],'C2zTC3z^2':[c2zT]+2*[c3z],'C2zTC3z^3':[c2zT]+3*[c3z],'C2zTC3z^4':[c2zT]+4*[c3z],'C2zTC3z^5':[c2zT]+5*[c3z]}
six_prime_unit_irreps={'C3z':{'GM1':np.array([1]),'GM2GM3':np.array([[np.exp(1j*2*np.pi/3),0],[0,np.exp(-1j*2*np.pi/3)]]),'GM4bar':np.array([-1]),'GM5barGM6bar':np.array([[np.exp(-1j*np.pi/3),0],[0,np.exp(1j*np.pi/3)]])},
				'C3z^2':{'GM1':np.array([1]),'GM2GM3':np.array([[np.exp(-1j*2*np.pi/3),0],[0,np.exp(1j*2*np.pi/3)]]),'GM4bar':np.array([-1]),'GM5barGM6bar':np.array([[np.exp(1j*np.pi/3),0],[0,np.exp(-1j*np.pi/3)]])},
				'C3z^3':{'GM1':np.array([1]),'GM2GM3':np.array([[1,0],[0,1]]),'GM4bar':np.array([-1]),'GM5barGM6bar':np.array([[-1,0],[0,-1]])},
				'C3z^4':{'GM1':np.array([1]),'GM2GM3':np.array([[np.exp(1j*2*np.pi/3),0],[0,np.exp(-1j*2*np.pi/3)]]),'GM4bar':np.array([1]),'GM5barGM6bar':np.array([[np.exp(1j*2*np.pi/3),0],[0,np.exp(-1j*2*np.pi/3)]])},#I think this is d3+
				'C3z^5':{'GM1':np.array([1]),'GM2GM3':np.array([[np.exp(-1j*2*np.pi/3),0],[0,np.exp(1j*2*np.pi/3)]]),'GM4bar':np.array([1]),'GM5barGM6bar':np.array([[np.exp(-1j*2*np.pi/3),0],[0,np.exp(1j*2*np.pi/3)]])},#I think this is d3-
				'1':{'GM1':np.array([1]),'GM2GM3':np.array([[1,0],[0,1]]),'GM4bar':np.array([1]),'GM5barGM6bar':np.array([[1,0],[0,1]])}
}

def save_symmetry_operators(sym_dir,relative_filename,operator_object):
		full_filename=sym_dir+'/'+relative_filename
		os.makedirs(os.path.dirname(full_filename), exist_ok=True)
		with open(full_filename+'.dill', 'wb') as file:
			dill.dump(operator_object, file)
		print(f'Saved the symmetry operator object to:\n {full_filename}')
			
	

#Make and saved symmetry operators
# test_c3=symmetry_operator_saved(state_list=shell_basis,generator_list=[ctest2])
# test_c3.unitary_matrix()
# c3_spinless=symmetry_operator_saved(state_list=shell_basis,generator_list=[ctest3])
# c3_spinless.unitary_matrix()

# sym_dir='/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/matrix_templates/4particles_2shells_centerK_matrices_new/symmetry_objects'
# print(f'started making symmetry objects')
# save_symmetry_operators(sym_dir=sym_dir,relative_filename='c3_spinless',operator_object=c3_spinless)
# save_symmetry_operators(sym_dir=sym_dir,relative_filename='c3_spinfull',operator_object=test_c3)
# exit()

###########################################################################

# c3vg=symmetry_operator2(state_list=shell_basis,generator_list=[ctest3vg])
#tauz=symmetry_operator2(state_list=shell_basis,generator_list=[({0:makeH.p0,1:qc0,2:makeH.pz,3:makeH.p0},1,False)])
if __name__ == "__main__":

	#testsym=makeH.inspect_elements(matrix=test_c3.unitary_matrix(),state_list=shell_basis)
	#makeH.inspect_elements(matrix=test_c3.unitary_matrix(),state_list=shell_basis)


	
	#makeH.inspect_elements(matrix=test_c3,state_list=shell_basis)
	c2zTR=symmetry_operator2(state_list=shell_basis,generator_list=[c2zT])
	# testwf=np.zeros(len(shell_basis),dtype=complex)
	# testind=1
	# testwf[testind]=1
	# print(testwf.shape)
	# #test_res=test_c3.linear_act(linear=testwf)
	# test_res=np.dot(test_c3.unitary_matrix(),testwf)
	# for i in range(len(test_res)):
	# 	if np.isclose(np.abs(test_res[i]),0)==False:
	# 		print(f'Input state: {makeH.eyepreservation(shell_basis[testind])}, Output {makeH.eyepreservation(shell_basis[i])}, coeff: {test_res[i]}')
	
	def first_projector_unit(irrep,state_list):
		d=six_prime_unit_irreps['1'][irrep].shape[0]
		group_order=len(six_prime_unit_irreps.keys())
		print(d,group_order)
		projector=np.zeros((len(state_list),len(state_list)),dtype=complex)
		for element_key in six_prime_unit_irreps.keys():
			char=np.conjugate(six_prime_unit_irreps[element_key][irrep])
			if char.shape==(1,):
				char=char[0]
			else:
				char=np.trace(char)
			
			rep_matrix=symmetry_operator2(state_list=state_list,generator_list=six_prime_unitary[element_key]).unitary_matrix()
			projector=projector+char*rep_matrix
		matrix=(d/group_order)*projector
		return matrix
	
	print('proj')
	
	testwf=np.zeros(len(shell_basis),dtype=complex)
	test_id_proj=first_projector_unit(irrep='GM1',state_list=shell_basis)
	print(np.allclose(test_id_proj,np.zeros(test_id_proj.shape)))
	exit()
	testind=1
	testwf[testind]=1
	testrs=np.dot(test_id_proj,testwf)
	#testrs=np.dot(test_id_proj,testrs)
	for i in range(len(testrs)):
		if np.isclose(np.abs(testrs[i]),0)==False:
			print(f'Input state: {makeH.eyepreservation(shell_basis[testind])}, Output {makeH.eyepreservation(shell_basis[i])}, coeff: {testrs[i]}')

	exit()
	# testwf=np.zeros(len(shell_basis),dtype=complex)
	# count=0
	# for i in shell_basis:
	# 	count+=1
	# 	print(f'State {count}: {makeH.eyepreservation(i)}')
        
    

    #########