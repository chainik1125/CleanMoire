import numpy as np
import matplotlib.pyplot as plt
import json
from oldbase.variables_cluster2 import *
from tqdm import tqdm
import dill

acc0=8
def convert_to_json(data_dic):
    json_dic={}
    for i in data_dic.keys():
        json_dic[str(i)]=data_dic[i]
    return json_dic

#save data
def json_save(outputfilename,inputdic):
    with open(outputfilename, 'w') as fp:
        json.dump(inputdic, fp)


# read data
def json_open(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def HSP_paths(HSP_list,point_no):
    k_points=[]
    for i in range(0,len(HSP_list)-1):
            k_interval=[]
            interval=(HSP_list[i+1]-HSP_list[i])/(point_no)
            for k in range(1,point_no+1):
                k_interval.append(HSP_list[i]+(k-1)*(interval))
            k_points.append(((HSP_list[i+1],HSP_list[i]),k_interval))
    return k_points

# print(HSP_paths(HSP_list=[Gamma,X],point_no=10))
def samogon_counter(list1,accuracy):
    counter_dic={}
    list2=[round(k,accuracy) for k in list1]
    for j in list2:
        if (j in list(counter_dic.keys()))==False:
            counter_dic[j]=1
        elif j in list(counter_dic.keys()):
            counter_dic[j]=counter_dic[j]+1
    return counter_dic

def degcounter(yv,acc):
    deglist=[]
    degdict=samogon_counter(yv,acc)
    for y in yv:
        deglist.append(degdict[round(y,acc)])
    return deglist

colors={1:'black',2:'darkorange',3:'limegreen',4:'darkgreen',5:'pink',6:'blue',7:'darkblue',8:'cyan',9:'yellow',10:'rebeccapurple',11:'green',12:'maroon',15:'violet',13:'violet',14:'brown',16:'indigo',17:'gold',18:'navy',19:'orange',20:'magenta',21:'darkblue',22:'maroon',23:'yellow',24:'red',25:'cyan',30:'brown',32:'maroon',33:'slateblue',34:'brown',35:'peru',36:'brown',48:'crimson',49:'purple',68:'yellow',70:'red'}


def chained_path_plot_link(path_list,kpoints,generate_Hk,UHK,mu,Umu,Utau,Uff,names_reversed_var):
    #Note: here generate_Hk is the hamiltonian, but which only accepts the k-coordinates as arguments
    #the U's are just there so that I can feed them into the string which describes the output - probably a more efficient way to do it
    #
    #Also note: remember the idea here is that the `link` is an unbroken path, G-X-M-G-Z, say. In the cluster makes more sense to only take on start and end HSP.
    deg_dict={}
    kgrid_dic={}
    # Calculate total_steps more safely
    total_steps = 0
    for i in range(len(path_list) - 1):
        list_temp = [path_list[i], path_list[i + 1]]
        # Assuming HSP_paths function returns an iterable where each element represents a step
        for _ in HSP_paths(list_temp, point_no=kpoints):
            total_steps += len(_[1])  # Adjust according to how HSP_paths structure is defined

    progress_bar = tqdm(total=total_steps, desc='Processing k-points')
    for i in range(0,len(path_list)-1):
        list_temp=[path_list[i],path_list[i+1]]
        x_values=[]
        for j in HSP_paths(list_temp,point_no=kpoints):
            for k in j[1]:
                x_values.append(k)
        y_values=[]
        deg_values=[]
        
        for j in x_values:
            twoparticleenergies=np.linalg.eigh(generate_Hk(kx=j[0],ky=j[1]))[0]
            
            gs=twoparticleenergies[0]
            
            deg_dict[(j[0],j[1])]=samogon_counter(twoparticleenergies,accuracy=acc0)
            deg_values.append(degcounter(twoparticleenergies,acc=acc0))
            

            kgrid_dic[(j[0],j[1])]=(list(twoparticleenergies),degcounter(twoparticleenergies,acc=acc0))
            progress_bar.update(1)
    progress_bar.close()
    filename=f"kcent_{particle_no}particle_shells{shells_used}_{names_reversed_var[str(path_list[0])]}to{names_reversed_var[str(path_list[1])]}UHK{UHK}UHKrot{UHK_rot}UHKtau{UHK_N_p0t0pxp0}mu{int(mu)}kp{kpoints}theta{round(theta*180/np.pi,2)}"
    print(filename)
    json_save('pathdata5/'+filename+'datadic.json',convert_to_json(kgrid_dic))

def chained_path_plot_link_doubleocc(path_list,kpoints,generate_Hk,HKterm,UHK,mu,Umu,Utau,Uff,names_reversed_var):
    #Note: here generate_Hk is the hamiltonian, but which only accepts the k-coordinates as arguments
    #the U's are just there so that I can feed them into the string which describes the output - probably a more efficient way to do it
    #
    #Also note: remember the idea here is that the `link` is an unbroken path, G-X-M-G-Z, say. In the cluster makes more sense to only take on start and end HSP.
    deg_dict={}
    kgrid_dic={}
    # Calculate total_steps more safely
    total_steps = 0
    for i in range(len(path_list) - 1):
        list_temp = [path_list[i], path_list[i + 1]]
        # Assuming HSP_paths function returns an iterable where each element represents a step
        for _ in HSP_paths(list_temp, point_no=kpoints):
            total_steps += len(_[1])  # Adjust according to how HSP_paths structure is defined

    progress_bar = tqdm(total=total_steps, desc='Processing k-points')
    for i in range(0,len(path_list)-1):
        list_temp=[path_list[i],path_list[i+1]]
        x_values=[]
        for j in HSP_paths(list_temp,point_no=kpoints):
            for k in j[1]:
                x_values.append(k)
        y_values=[]
        deg_values=[]
        
        for j in x_values:
            twoparticleenergies,eigstates=np.linalg.eigh(generate_Hk(kx=j[0],ky=j[1],UHK=UHK))
            Uproj=np.dot(eigstates.conj().T, np.dot(HKterm, eigstates))
            gs=twoparticleenergies[0]
            
            deg_dict[(j[0],j[1])]=samogon_counter(twoparticleenergies,accuracy=acc0)
            deg_values.append(degcounter(twoparticleenergies,acc=acc0))
            

            kgrid_dic[(j[0],j[1])]=(list(Uproj),degcounter(twoparticleenergies,acc=acc0))
            progress_bar.update(1)
    progress_bar.close()
    print(Uproj.shape)
    filename=f"kcent_{particle_no}particle_shells{shells_used}_{names_reversed_var[str(path_list[0])]}to{names_reversed_var[str(path_list[1])]}UHK{UHK}mu{int(mu)}Umu{Umu}Utau{Utau}Uff{Uff}kp{kpoints}theta{round(theta*180/np.pi,2)}"
    print(filename)
    json_save('pathdata5/'+filename+'datadic.json',convert_to_json(kgrid_dic))