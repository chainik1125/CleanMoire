

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






if __name__ == "__main__":
    



    start='KM'
    end ='GammaM'
    UHK=UHK_N_p0t0p0px


    #pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire[start],vars_dic_Moire[end]],kpoints=10,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    #Save path files

    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['A'],vars_dic_Moire['B']],kpoints=kpoints,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['B'],vars_dic_Moire['C']],kpoints=kpoints,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['C'],vars_dic_Moire['D']],kpoints=kpoints,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)
    pathdiag.chained_path_plot_link(path_list=[vars_dic_Moire['D'],vars_dic_Moire['A']],kpoints=kpoints,generate_Hk=load_templates.gen_Hk2_tensor,UHK=UHK,mu=mu,Utau=Utau,Umu=Umu,Uff=Uff,names_reversed_var=names_reversed_Moire)

    #plot path
    dstr=pathdata_folder_cluster+f"/mu{mu}UHK{UHK}UHKrot{UHK_rot}Utau{Utau}kp{kpoints}theta{thetadeg}"
    directory = os.fsencode(dstr)
    params=f'UHK{UHK}'
    plot_save.chained_path_plot(path_lists=[[A,B,C,D,A]],kpoints=str(kpoints),directory=directory,dstr=dstr,mu_shift=mu,params=params,variable=f'{particle_no}particle',theta=f'theta{thetadeg}',linesplotted=16,plotcutoff=16)

    exit()