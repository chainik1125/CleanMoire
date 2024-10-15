

import os
import sys
import yaml
import argparse
import dill as pickle
import time
from variables import *
from setup import *
from base_functions import *
from base_classes import saved_template_matrix
import make_templates
import load_templates
import pathdiag
import plot_save


sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire')








start='KM'
end ='GammaM'
UHK=UHK_N_p0t0p0px


#pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire[start],vars_dic_Moire[end]],kpoints=10,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
#Save path files
UHK=UHK_N_p0t0p0px
U_tau=UHK_N_p0t0pxp0
kp=kpoints

#excited_particles=7
#gs_particles=8



get_path=True
if get_path:
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['A'],vars_dic_Moire['B']],kpoints=kpoints,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['B'],vars_dic_Moire['C']],kpoints=kpoints,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['C'],vars_dic_Moire['D']],kpoints=kpoints,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['D'],vars_dic_Moire['A']],kpoints=kpoints,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)

plot_path=False
if plot_path:
    dstr=pathdata_folder_exc_minus+f"/mu{mu}UHK{UHK}UHKrot{UHK_rot}Utau{Utau}kp{kp}theta{round(thetadeg,2)}"
    directory = os.fsencode(dstr)
    params=f'UHK{UHK}'
    plot_save.chained_path_plot(path_lists=[[A,B,C,D,A]],kpoints=str(kpoints),directory=directory,dstr=dstr,mu_shift=mu,params=params,variable=f'{particle_no}particle',theta=f'theta{thetadeg}',linesplotted=16,plotcutoff=16)

exit()
#print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load YAML files

# print('herro')
# print(make_templates.saved_template_matrix)
# exit()
# Update config with command-line arguments if provided
for key, value in vars(args).items():
    if value is not None:
        config[key] = value

dir_path=load_templates.template_matrix_dir
shell_basis=make_templates.shell_basis

#Let's start working at one particle
HkA=load_templates.gen_Hk2(kx=A[0],ky=A[1],particles_used=1)

print(HkA.shape)

print(f'eigenvalues of last block diagonal:\n{np.linalg.eigh(HkA[12:16,12:16])[0]}')
print(f'Tunnelling term w_1: {w1}')

def vg(k0,G):
    Hk0=load_templates.gen_Hk2(kx=k0[0],ky=k0[1],particles_used=1)
    eigvals_init,eigvecs_init=np.linalg.eigh(Hk0)
    Hkg=load_templates.gen_Hk2(kx=(k0+G)[0],ky=(k0+G)[1],particles_used=1)
    eigvals_translated,eigvecs_translated=np.linalg.eigh(Hkg)

    vg_eigbasis=np.conjugate(np.transpose(eigvecs_init))@eigvecs_translated
    vg_compbasis=eigvecs_init@vg_eigbasis@np.conjugate(np.transpose(eigvecs_init))


    
    

    print(np.allclose(vg_compbasis@eigvecs_init,eigvecs_translated))


vg(A,-qvecs[0]-2*qvecs[1])

exit()


#term_number=int(sys.argv[1])

#template=make_templates.construct_templates(dir_path,make_templates.term_list_dic,term_number=None,make_all=True)
#q0proj=make_templates.tpp(state_list=shell_basis,pauli_dic={0:make_templates.p0,1:make_templates.tq0,2:make_templates.p0,3:make_templates.p0},prefactor=1)
#make_templates.inspect_elements(q0proj,shell_basis)

#from Code.oldbase import symmetries2





state_indices_1=[4,5]
state_indices_2=[10,11]

b1=-qvecs[0]-2*qvecs[0]
print(f'b1: \n {b1}')

b2=qvecs[0]-qvecs[1]
HkA_2=load_templates.gen_Hk2(kx=A[0]+b2[0],ky=A[1]+b2[1],particles_used=1)
# print(np.linalg.eigh(HkA_1)[0])
# print(np.linalg.eigh(HkA_2)[0])
# exit()
print(f'same under RLV translation: {np.allclose((np.linalg.eigh(HkA_1)[0]),(np.linalg.eigh(HkA_2)[0]))}')
exit()
print(HkA_1.shape)
print(np.linalg.eigh(HkA_1)[0])
eigvals1,eigvecs1=np.linalg.eigh(HkA_1)
coeffmatrix=eigvecs1[:,state_indices_1[0]]
print(coeffmatrix.shape)

#2. OK the more difficult part is to write a function that takes an arbitrary four particle states and returns it as a 16x4 matrix
print(len(shell_basis))
exit()




HkA=load_templates.gen_Hk2(kx=A[0],ky=A[1])
print(np.linalg.eigh(HkA)[0][:16])




# HkB=load_templates.gen_Hk2(kx=B[0],ky=B[1])
# HkgB=load_templates.gen_Hk2(kx=-qvecs[0][0],ky=-qvecs[0][1])
# HkD=load_templates.gen_Hk2(kx=D[0],ky=D[1])
# HkD=load_templates.gen_Hk2(kx=D[0],ky=D[1])
# HkgD=load_templates.gen_Hk2(kx=qvecs[1][0],ky=qvecs[1][1])

# randpoint=np.array([B[0]/32,B[1]/32])
# rotrandpoint=np.array([randpoint[0]*np.cos(phi)-randpoint[1]*np.sin(phi),randpoint[0]*np.sin(phi)+randpoint[1]*np.cos(phi)])
# Hkrand=load_templates.gen_Hk2(kx=randpoint[0],ky=randpoint[1])
# Hkrotrand=load_templates.gen_Hk2(kx=rotrandpoint[0],ky=rotrandpoint[1])


ham=HkA
rotham=HkA
print(f'start import')
from Code.oldbase import symmetries2
def qc3(qpair):
    q2=qpair[0]
    q3=qpair[1]
    return (1,(-q3,q2-q3))


###########################band hk##########################

#1. Get single particle states.
#2. Convert four particle states to matrix of coefficients.
#3. Contract with the single particle states to get the projection.
#4. From the projections, get the HK term




# c3z=({0:make_templates.p0,1:qc3,2:make_templates.p0,3:make_templates.pexpz3},1,False)
# c3z=({0:make_templates.p0,1:qc3,2:make_templates.p0,3:make_templates.pexpz3},1,False)
# c3z2=({0:make_templates.p0,1:qc3,2:make_templates.pexpz3,3:make_templates.pexpz6},1,False)
# #c3perm=({0:make_templates.p0,1:qc3,2:make_templates.p0,3:make_templates.p0},1,False)
# test_c3=symmetries2.symmetry_operator2(state_list=shell_basis,generator_list=[c3z2])
# #test_c3_perm=symmetries2.symmetry_operator2(state_list=shell_basis,generator_list=[c3perm])
# symop=test_c3.unitary_matrix()
# c3vg=symmetries2.c3vg.unitary_matrix()

# ctest=({0:make_templates.p0,1:symmetries2.qc0,2:make_templates.p0,3:make_templates.p0},1,False)
# c3zop=symmetries2.test_c3

#make_templates.inspect_elements(matrix=c3zop.unitary_matrix(),state_list=shell_basis)
print(f'load expectations')
#c3zopmat=symmetries2.test_c3.unitary_matrix()
#c3spinlessmat=symmetries2.c3_spinless.unitary_matrix()

# spin_exp_term=[({0:make_templates.p0,1:make_templates.t0,2:make_templates.p0,3:make_templates.pz},g00,[],[],[])]
# other_matrices_dic={'expectation_matrices':(spin_exp_term,'spin_tauz','nonint')}
# template=make_templates.construct_templates(spin_dir,other_matrices_dic,term_number=None,make_all=True)
# exit()


print(f'Load expectation operators')
exp_dir=f"/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/matrix_templates/4particles_2shells_centerK_matrices_new/expectation_operators"
load_spin=load_templates.load_matrices([f'{exp_dir}/spin_tauz_None.dill'])


print(f'Load symmetry operators')
sym_dir='/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/matrix_templates/4particles_2shells_centerK_matrices_new/symmetry_objects'

with open(sym_dir+'/c3_spinless.dill', 'rb') as file:
    c3spinless_op=pickle.load(file)
with open(sym_dir+'/c3_spinfull.dill', 'rb') as file:
    c3spinful_op=pickle.load(file)


c3spinlessmat=c3spinless_op.unitary_matrix()
c3spinfulmat=c3spinful_op.unitary_matrix()

print(f'unitary part done')

#permop=test_c3_perm.unitary_matrix()
#print(f'symmetry test: {np.allclose(np.conjugate(symop).T@Hkrand@symop,np.conjugate(permop).T@Hkrotrand@permop)}')
#print(f'symmetry test: {np.allclose(np.conjugate(c3spinlessmat.dot(c3vg)).T@HkB@(c3spinlessmat.dot(c3vg)),HkgB)}')
print(f'start symmetry test')
print(f'symmetry test done: {np.allclose(np.conjugate(c3spinlessmat).T@rotham@(c3spinlessmat),ham)}')

print(f'eigenvalue expectations')

eigvals,eigvecs=np.linalg.eigh(ham)
c3spinless_exp=np.conjugate(np.transpose(eigvecs))@c3spinlessmat@eigvecs
spintauz_exp=np.conjugate(np.transpose(eigvecs))@load_spin@eigvecs

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.2f}'.format}, linewidth=100):
    print(f'spin tauz expectation value, first six')
    print(spintauz_exp[:6,:6])



with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.2f}'.format}, linewidth=100):
    print(f'c3spinless expectation value, first four')
    print(c3spinless_exp[:4,:4])
    reducedeigvals,reducedeigvecs=np.linalg.eig(c3spinless_exp[:4,:4])
    print(reducedeigvals)
    projspin=np.conjugate(np.transpose(reducedeigvecs))@spintauz_exp[:4,:4]@reducedeigvecs
    print(f'projected spin tauz expectation value, first four')
    print(projspin)
    

exit()


irrep_dic={'1':np.ones(1,dtype=complex),
           '2pi/3':np.array([1,np.exp(2*np.pi*1j/3),np.exp(-2*np.pi*1j/3)]),
           '-2pi/3':np.array([1,np.exp(-2*np.pi*1j/3),np.exp(2*np.pi*1j/3)]),
           'matrices':np.array([np.identity(n=c3zopmat.shape[0],dtype=complex),c3zopmat,c3zopmat.dot(c3zopmat)])}



def projector(irrep_dic,irrep):
    matrices=irrep_dic['matrices']
    characters=irrep_dic[irrep]

    return (1/(len(characters)))*np.sum(characters*matrices,axis=0)

test_proj=projector(irrep_dic,'1')

#print('inspect element c3spinless')
#make_templates.inspect_elements(matrix=c3spinlessmat,state_list=shell_basis)
#print('inspect element c3spinless^2')
#make_templates.inspect_elements(matrix=c3spinlessmat.dot(c3spinlessmat),state_list=shell_basis)
print(f'square = inverse? {np.allclose(c3spinlessmat@c3spinlessmat,np.conjugate(np.transpose(c3spinlessmat)))}')


test_proj1=(1/3)*(np.identity(16,dtype=complex)+np.conjugate(np.exp(0*np.pi*1j/3))*c3spinlessmat+np.conjugate(np.exp(0*np.pi*1j/3))*np.conjugate(np.transpose(c3spinlessmat)))
test_proj2=(1/3)*(np.identity(16,dtype=complex)+np.conjugate(np.exp(2*np.pi*1j/3))*c3spinlessmat+np.conjugate(np.exp(-2*np.pi*1j/3))*np.conjugate(np.transpose(c3spinlessmat)))
test_proj3=(1/3)*(np.identity(16,dtype=complex)+np.conjugate(np.exp(-2*np.pi*1j/3))*c3spinlessmat+np.conjugate(np.exp(+2*np.pi*1j/3))*np.conjugate(np.transpose(c3spinlessmat)))

print(f'sum proj identity? {np.allclose(np.identity(16,complex),test_proj1+test_proj2+test_proj3)}')

commute_test=np.allclose(c3spinlessmat@ham,ham@c3spinlessmat)
print(f'commute test: {commute_test}')
print(ham.shape)

exit()
print(f'P^2=P?  {np.allclose(test_proj1.dot(test_proj1),test_proj1)}')





print(f'Second one: P^2=P?  {np.allclose(test_proj2.dot(test_proj2),test_proj2)}')

eigvals,eigvecs=np.linalg.eigh(ham)
print(eigvecs.shape)

symmetry_charges=np.diag(np.conjugate(eigvecs).T@((test_proj1+test_proj2+test_proj3)@eigvecs))
tau_exp=np.diag(np.conjugate(eigvecs).T@((c3vg)@eigvecs))
#symmetry_charges=np.diag(np.conjugate(eigvecs).T@eigvecs)
print('symmetry charges:')
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(symmetry_charges)
print(f'eigenvalues:')
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(eigvals)
print(f'tau expectation value:')
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
    print(tau_exp)
exit()

#symop=symmetries2.test_c3.unitary_matrix()

#make_templates.inspect_elements(HkA,shell_basis)

#make_templates.inspect_elements(symop,shell_basis)

#print(f'symmetry elements same? :{np.allclose(np.conjugate(symop).T@HkA@symop,HkA)}')

print('new')





diff=((np.conjugate(symop).T)@HkA@symop+HkA)
make_templates.inspect_elements(diff,shell_basis)
exit()
print(diff)
#print(np.allclose(np.linalg.eigh(diff)[0],np.linalg.eigh(HkA)[0]))

exit()


exit()

def symmetry_charges(HSP,proj,charge):
    ham=load_templates.gen_Hk2(kx=HSP[0],ky=HSP[1])
    a_eigvals,a_eigvecs=np.linalg.eigh(ham)
    a_gindices=np.where(a_eigvals==min(a_eigvals))[0]
    a_geigvecs=a_eigvecs[:,a_gindices]
    qexp=np.conjugate(a_geigvecs).T@(proj@a_geigvecs)
    print(f'qexp shape {qexp.shape}')
    
    print(f'Charge {charge} expectation value')
    print(qexp)
    
    return None


q_allequal=make_templates.tpp(state_list=make_templates.shell_basis,pauli_dic={0:make_templates.p0,1:make_templates.tqproj1,2:make_templates.p0,3:make_templates.p0},prefactor=1)
q_allequal=q_allequal+make_templates.tpp(state_list=make_templates.shell_basis,pauli_dic={0:make_templates.p0,1:make_templates.tqproj2,2:make_templates.p0,3:make_templates.p0},prefactor=1)
q_allequal=q_allequal+make_templates.tpp(state_list=make_templates.shell_basis,pauli_dic={0:make_templates.p0,1:make_templates.tqproj3,2:make_templates.p0,3:make_templates.p0},prefactor=1)
#make_templates.inspect_elements(q_allequal,make_templates.shell_basis)

symmetry_charges(A,q_allequal,'all equal')
symmetry_charges(A,q0proj,'q0')

exit()
a_geigvecs=a_eigvecs[:,np.argsort(a_eigvals)]

HkB=load_templates.gen_Hk2(kx=B[0],ky=B[1])



print(np.linalg.eigh(HkA)[0])
print(np.linalg.eigh(HkB)[0])





