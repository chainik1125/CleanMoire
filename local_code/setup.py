from variables import *

template_matrix_dir=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/matrix_templates'
#dir_path=f'../Large_files/matrix_templates/{particle_no}particles_{shells_used}shells_center{center}_matrices'
pathdata_folder_local=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/pathdata/{particle_no}particles_{shells_used}shells_center{center}'
pathdata_folder_exc_plus=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/pathdata/exc/{particle_no}particles_excparticles_{particle_no+1}_{shells_used}shells_center{center}'
pathdata_folder_exc_minus=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/pathdata/exc/{particle_no}particles_excparticles_{particle_no-1}_{shells_used}shells_center{center}'
saved_spectra_folder=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/band_plots/{particle_no}particles_{shells_used}shells_center{center}'
saved_exc_plus_folder=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/exc_plots/{particle_no}particles_{particle_no+1}particles_{shells_used}shells_center{center}'
saved_exc_minus_folder=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/exc_plots/{particle_no}particles_{particle_no-1}particles_{shells_used}shells_center{center}'


#cluster_paths
template_matrix_dir_cluster=f'/projects/illinois/eng/physics/bbradlyn/dmitry2/Moire/large_files/matrix_templates/tensor/{particle_no}particles_shells{shells_used}_center{center}'
template_matrix_dir_cluster=f'/projects/illinois/eng/physics/bbradlyn/dmitry2/Moire/large_files/matrix_templates/tensor/8particles_2shells_centerK_matrices/ham_terms'
#dir_path=f'../Large_files/matrix_templates/{particle_no}particles_{shells_used}shells_center{center}_matrices'
pathdata_folder_cluster=f'/projects/illinois/eng/physics/bbradlyn/dmitry2/Moire/large_files/spectra/pathdata/{particle_no}particles_shells{shells_used}_center{center}'
spectra_plots_folder_cluster=f'/projects/illinois/eng/physics/bbradlyn/dmitry2/Moire/large_files/spectra/band_plots/{particle_no}particles_shells{shells_used}_center{center}'
saved_spectra_folder_cluster=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/band_plots/{particle_no}particles_{shells_used}shells_center{center}'
saved_exc_plus_folder_cluster=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/exc_plots/{particle_no}particles_{particle_no+1}particles_{shells_used}shells_center{center}'
saved_exc_minus_folder_cluster=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files/spectra/exc_plots/{particle_no}particles_{particle_no-1}particles_{shells_used}shells_center{center}'


#function to generate the right path
cluster_root=f'/projects/illinois/eng/physics/bbradlyn/dmitry2/Moire/large_files'
local_root=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/large_files'

def path_string(pathtype:str,clusterarg:bool=False,particles_exc:int=8,particles_gs:int=8)->str:
    if pathtype=='pathdata':
        first_level=f'/spectra/pathdata/{particles_exc}_particles_{shells_used}_shells_center_{center}'
    elif pathtype=='band_plots':
        first_level=f'/spectra/band_plots/{particles_exc}_exc_{particles_gs}_gs_{shells_used}_shells_center_{center}'
    second_level=f'/mu{mu}UHK{UHK_N_p0t0p0px}UHKrot{UHK_rot}Utau{Utau}kp{kpoints}theta{round(theta*180/np.pi,2)}'
    if clusterarg:
        return cluster_root+first_level+second_level
    else:
        return local_root+first_level+second_level
    
        
    
# def spectra_path(pathtype:str,cluster:bool=False,particles_exc:int=8,particles_gs:int=8)->str:
#     par_string=local_root+path_string(pathtype,particles_exc,particles_gs)
#     if cluster:
#         return cluster_root+'/'+pathtype+par_string
#     else:
#         return local_root+'/'+pathtype+par_string

