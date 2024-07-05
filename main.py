import make_templates
from variables import *
from setup import *
import load_templates


dir_path=f'../Large_files/matrix_templates/{particle_no}particles_{shells_used}shells_center{center}_matrices'

#term_number=int(sys.argv[1])
#template=make_templates.construct_templates(dir_path,make_templates.term_list_dic,term_number=term_number,make_all=True)

HkA=load_templates.gen_Hk2(kx=A[0],ky=A[1])
HkB=load_templates.gen_Hk2(kx=B[0],ky=B[1])



print(np.linalg.eigh(HkA)[0])
print(np.linalg.eigh(HkB)[0])





