import numpy as np
import sys
import yaml
import argparse
import torch
#sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire')

with open('default_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# This allows you to modify any of the yaml configs from the command line.
parser = argparse.ArgumentParser(description="Update YAML configuration values.")
for key in config:
    parser.add_argument(f'--{key}', type=str, help=f'Override value for {key}')

args = parser.parse_args()

for key, value in vars(args).items():
    if value is not None:
        config[key] = value

#import all args
thetadeg=float(config['thetadeg'])
UHK_N_p0t0p0px=float(config['Uorb'])
UHK_rot=float(config['Urot'])
UHK_N_p0t0pxp0=float(config['Utau'])
particle_no=int(config['particles'])

precision=torch.complex64
# t=1
# v=1
# v=3*t/2
# w0=2*t
# w1=2*t
dof=4
testpdd={'k':4,'sublattice':2,'spin':2}
thetadeg=thetadeg#float(sys.argv[2])
theta=(np.pi/180)*thetadeg
phi=2*np.pi/3
center='K'
shells_used=2
kpoints=3
particle_no=particle_no

#Bernevig params
v=5.944#t=v=5.944?
w1=0.11#0.11
w0=0.7*w1
Kmag=1.703
UHK_N_p0t0p0px=UHK_N_p0t0p0px
Utau=0
Uff=0
Umu=0
mu=(UHK_N_p0t0p0px+Utau)/2
UHK_rot=UHK_rot
UHK_N_p0t0pxp0=UHK_N_p0t0pxp0

#Mcdonald param
# t=1
# v=5.944
# w1=0.11
# w0=w1
# Kmag=1.703
# shells_used=6
# kpoints=100

# t=1
# v=1
# w1=2*t
# w0=w1
# Kmag=1.703
# shells_used=2
# kpoints=100



# Gamma=np.array([0,0])
# K=np.array([4*np.pi/3,0])
# K1=np.array([-4*np.pi/3,0])
# M=np.array([2*np.pi/3,0])
#A=2*np.sqrt(np.vdot(K,K))*np.sin(theta/2)*np.array([0,-1])
#A=np.array([1,2])
# B=Gamma
# C=2*np.sqrt(np.vdot(K,K))*np.sin(theta/2)*np.array([0,1])
# D=2*np.sqrt(np.vdot(K,K))*np.sin(theta/2)*np.array([np.sqrt(3)/2,-1/2])
# QTR=2*np.sqrt(np.vdot(K,K))*np.sin(theta/2)*np.array([np.sqrt(3)/2,1/2])
# QTL=2*np.sqrt(np.vdot(K,K))*np.sin(theta/2)*np.array([-np.sqrt(3)/2,1/2])

kd=2*(Kmag)*np.sin(theta/2)#2*np.sqrt(np.vdot(K,K))*np.sin(theta/2)
# k1=np.array([0,-1])
# k2=np.array([np.sqrt(3)/2,1/2])
# k3=np.array([-np.sqrt(3)/2,1/2])


# b1=8*qvecs[1]+qvecs[0]
# b2=5*qvecs[1]-3*qvecs[0]
#K-centered
# KM=np.array([0,0])#Because you are in the K-centered model.
# GammaM=-qvecs[0]
# MM=-qvecs[0]-(1/2)*(qvecs[0]+qvecs[1])
#Gamma-Centerd
if center=='Gamma':
    qvecs=[kd*np.array([-np.sqrt(3)/2,-1/2]),kd*np.array([np.sqrt(3)/2,-1/2])]#q2,q3
    GammaM=np.array([0,0])
    KM=qvecs[1]
    MM=qvecs[1]+(1/2)*(qvecs[0]+qvecs[1])
elif center=='K':
    #Bernevig setup

    #Parameters I used for overleaf plots
    # qvecs=[-kd*np.array([np.sqrt(3)/2,1/2]),-kd*np.array([-np.sqrt(3)/2,1/2])]#q2,q3
    # KM=np.array([0,0])#Because you are in the K-centered model.
    # GammaM=-qvecs[0]
    # MM=-qvecs[0]+(-qvecs[0]-qvecs[1])/2
    #Parameters I should have used for overleaf plots
    # qvecs=[-kd*np.array([np.sqrt(3)/2,1/2]),-kd*np.array([-np.sqrt(3)/2,1/2])]#q2,q3
    # KM=np.array([0,0])#Because you are in the K-centered model.
    # GammaM=-qvecs[0]
    # MM=-(qvecs[0]+qvecs[1])/2
    #Mcdonald setup
    qvecs=[kd*np.array([np.sqrt(3)/2,1/2]),kd*np.array([-np.sqrt(3)/2,1/2])]#q2,q3
    # A=-qvecs[0]-qvecs[1]
    # B=np.array([0,0]) #i.e. Bernevigs KM in the K centered model
    # C=qvecs[0]+qvecs[1]
    # D=-qvecs[1] #This is what Mcdonald calls D
    A=np.array([0,0])
    B=qvecs[0]+qvecs[1]
    C=2*(qvecs[0]+qvecs[1])#Is this correct? I think so - it's twice the distance of B
    D=qvecs[0]

    KM=np.array([4,5])#np.array([0,0])
    GammaM=np.array([6,7])#-qvecs[0]
    MM=np.array([7,8])#-qvecs[0]-(qvecs[0]+qvecs[1])/2
   
    




    
# KM=(1/2)*(b1+b2)
# MM=(1/4)*(b1)

# names_dic={"Gamma":str(Gamma),"K":str(K),"K1":str(K1),"M":str(M),"A":str(A),"B":str(B),"C":str(C),"D":str(D),'KM':str((0,0)),'GammaM':str(GammaM),'MM':str(MM)}
# names_reversed=dict(zip(names_dic.values(),names_dic.keys()))
# vars_dic={"Gamma":Gamma,"K":K,"K1":K1,"M":M,"A":A,"B":B,"C":C,"D":D,'KM':KM,'GammaM':GammaM,'MM':MM}


names_dic_Moire={'KM':str(KM),'GammaM':str(GammaM),'MM':str(MM),'A':str(A),'B':str(B),'C':str(C),'D':str(D)}
names_reversed_Moire=dict(zip(names_dic_Moire.values(),names_dic_Moire.keys()))
vars_dic_Moire={'KM':KM,'GammaM':GammaM,'MM':MM,'A':A,'B':B,'C':C,'D':D}

# var_name_dict={r'$A$':tuple(A),r'$B$':tuple(B),r'$C$':tuple(C),r'$D$':tuple(D)}
# var_dict_reversed=dict(zip(var_name_dict.values(),var_name_dict.keys()))
# var_greek_dict={tuple(B):r'$(0,0)$',tuple(A):r'2$|k_{D}|\sin\theta(0,-1)$',tuple(C):r'$2|k_{D}|\sin(\theta)(0,-1)$',tuple(D):r'$2|k_{D}|\sin(\theta)(\sqrt{3}/2,-1/2)$'}
# var_file={'A':tuple(A),'B':tuple(B),'C':tuple(C),'D':tuple(D)}
# var_file_reversed=dict(zip(var_file.values(),var_file.keys()))

var_name_dict_Moire={r'$K_M$':tuple(KM),r'$\Gamma_M$':tuple(GammaM),r'$M_M$':tuple(MM),r'$A$':tuple(A),r'$B$':tuple(B),r'$C$':tuple(C),r'$D$':tuple(D)}
var_dict_reversed_Moire=dict(zip(var_name_dict_Moire.values(),var_name_dict_Moire.keys()))
var_greek_dict_Moire={tuple(KM):r'$(0,0)$',tuple(GammaM):r'2$|k_{D}|\sin\theta(-\sqrt{3}/2,-1/2)$',tuple(MM):r'$2|k_{D}|\sin(\theta)(0,-1/2)$',tuple(A):r'$2|k_{D}|\sin(\theta)(0,-1)$',tuple(B):r'$2|k_{D}|\sin(\theta)(0,0)$',tuple(C):r'$2|k_{D}|\sin(\theta)(0,1)$',tuple(D):r'$2|k_{D}|\sin(\theta)(\sqrt{3}/2,-1/2)$'}
var_file_Moire={'KM':tuple(KM),'GammaM':tuple(GammaM),'MM':tuple(MM),'A':tuple(A),'B':tuple(B),'C':tuple(C),'D':tuple(D)}
var_file_reversed_Moire=dict(zip(var_file_Moire.values(),var_file_Moire.keys()))






colors={1:'black',2:'darkorange',3:'gold',4:'darkgreen',5:'lime',6:'blue',7:'cornflowerblue',8:'red',9:'olive',10:'rebeccapurple',11:'darkslategrey',12:'maroon',13:'lightcoral',14:'chocolate',15:'navy',16:'magenta',17:'gold',18:'navy',19:'orange',20:'magenta',21:'darkblue',22:'maroon',23:'yellow',24:'red',25:'cyan',30:'brown',32:'maroon',33:'slateblue',34:'brown',35:'peru',36:'brown',48:'crimson',49:'purple',68:'yellow',70:'red',np.nan:'blue'}

#make_templates
testpdd={'k':4,'sublattice':2,'spin':2}
tqs=[np.array([-1,-1]),np.array([1,0]),np.array([0,1])]

