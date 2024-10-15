from setup import *
from variables import *
import make_templates
import load_templates
import pickle
import os

symmetry_matrix_templates_dir=f'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/Large_files/symmetry_matrix_templates'



c6prime=make_templates.construct_templates(symmetry_matrix_templates_dir,make_templates.term_list_dic,term_number=6,make_all=True)



