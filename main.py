import make_templates
from variables import *



dir_path=f'../Large_files/matrix_templates/{particle_no}particles_{shells_used}shells_center{center}_matrices'

term_number=int(sys.argv[1])
template=make_templates.construct_templates(dir_path,make_templates.term_list_dic,term_number=term_number,make_all=True)