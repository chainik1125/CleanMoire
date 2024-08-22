from variables import *
from setup import *
from base_functions import *
from base_classes import saved_template_matrix
from make_templates import load_matrices
import pickle
import os



# dir_path_kx=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kxmatrices_particles4_shells2_centerK'
# filelist_kx=[dir_path_kx+'/'+f for f in os.listdir(dir_path_kx) if os.path.isfile(os.path.join(dir_path_kx, f))]
# dir_path_ky=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kymatrices_particles4_shells2_centerK'
# filelist_ky=[dir_path_ky+'/'+f for f in os.listdir(dir_path_ky) if os.path.isfile(os.path.join(dir_path_ky, f))]
# dir_path_kind=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/kindmatrices_particles4_shells2_centerK'
# filelist_kind=[dir_path_kind+'/'+f for f in os.listdir(dir_path_kind) if os.path.isfile(os.path.join(dir_path_kind, f))]
# dir_path_tun=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/tunmatrices_particles4_shells2_centerK'
# filelist_tun=[dir_path_tun+'/'+f for f in os.listdir(dir_path_tun) if os.path.isfile(os.path.join(dir_path_tun, f))]
# HK_list=['/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCode/4particles_2shells_centerK_matrices/UHKrot_particles_4_shells_2_center_K.dill',
#             '/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/Numerics/MoireCluster/4particles_2shells_centerK_matrices_new/UHK_N_taux_matrix_particles4_shells2_centerK/HK_N_taux_0.dill']


def find_template_dirs(parent_dir,particle_no,shells_used,center):
    dir_path_kx=parent_dir+f'/kx'
    filelist_kx=[dir_path_kx+'/'+f for f in os.listdir(dir_path_kx) if os.path.isfile(os.path.join(dir_path_kx, f))]
    dir_path_ky=parent_dir+f'/ky'
    filelist_ky=[dir_path_ky+'/'+f for f in os.listdir(dir_path_ky) if os.path.isfile(os.path.join(dir_path_ky, f))]
    dir_path_kind=parent_dir+f'/qadjust'
    filelist_kind=[dir_path_kind+'/'+f for f in os.listdir(dir_path_kind) if os.path.isfile(os.path.join(dir_path_kind, f))]
    dir_path_tun=parent_dir+f'/tun'
    filelist_tun=[dir_path_tun+'/'+f for f in os.listdir(dir_path_tun) if os.path.isfile(os.path.join(dir_path_tun, f))]
    HK_list=[parent_dir+f'/HK_rot/HK_rot_None.dill',
                parent_dir+f'/HK_orb/HK_orb_None.dill',
                parent_dir+f'/HK_N_taux/HK_N_taux_None.dill']

    return filelist_kx,filelist_ky,filelist_kind,filelist_tun,HK_list



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


template_matrix_dir=f'../large_files/matrix_templates/{particle_no}particles_{shells_used}shells_center{center}_matrices_new/ham_terms'
def gen_Hk2(kx,ky):
    filelist_kx,filelist_ky,filelist_kind,filelist_tun,HK_list=find_template_dirs(template_matrix_dir,particle_no,shells_used,center)
    non_int_templates,HK_matrix=make_template_matrices(kx_list=filelist_kx,ky_list=filelist_ky,kind_list=filelist_kind,tun_list=filelist_tun,HK_list=HK_list)
    
    first=True
    for pair in non_int_templates:
        if first:
            H0=pair[0](kx,ky)*pair[1]
            first=False
        else:
            H0=H0+pair[0](kx,ky)*pair[1]
    H0=H0+HK_matrix
    return H0


if __name__ == "__main__":
    pass