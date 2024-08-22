from variables import *

class saved_template_matrix():
    def __init__(self,matrix,kfunction,variable_names,variable_factors,variable_functions,final_matrix_description):
        self.matrix=matrix
        self.kfunction=kfunction#the g0 the gkx and so on
        self.variable_names=variable_names#The things that go inside the functions
        self.variable_factors=variable_factors #i.e if you divide the variable by 2...
        self.variable_functions=variable_functions
        self.final_matrix_description=final_matrix_description
        self.parameterdic=None
        self.term=None
        
    def form_matrix(self):
        #The idea here is that this function will allow me to construct a matrix for arbitrary parameters once I have the template matrix saved
        coeff=1
        for i in range(len(self.variable_functions)):
            #print(self.variable_names,[globals()[x] for x in self.variable_names],self.variable_factors)
            arg=globals()[self.variable_names[i]]*self.variable_factors[i]
            coeff=coeff*self.variable_functions[i](arg)

        return self.matrix*coeff