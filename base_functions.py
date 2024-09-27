import numpy as np
import torch
from variables import *

def gkx(kx,ky):
    return kx
def gky(kx,ky):
    return ky
def g0(kx,ky):
    return 1
def g00(x):
    return x
def gw(w):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * g0(kx, ky)
    return multiplied_function
def gkxw(w):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * gkx(kx, ky)
    return multiplied_function
def gkyw(w):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * gky(kx, ky)
    return multiplied_function

def gcos(w,theta):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * np.cos(theta)* g0(kx, ky)
    return multiplied_function
def gsin(w,theta):
    # Define a new function that takes a and b, and uses the captured c
    def multiplied_function(kx, ky):
        return w * np.sin(theta)* g0(kx, ky)
    return multiplied_function
def gtx(phi):
    return np.cos(phi)
def gty(phi):
    return np.sin(phi)


#Now let's build up the tunnelling matrices
def t1_plus(sigma):
    newsigma=(sigma[0]-1,sigma[1]-1)#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)
def t1_plus_tensor(sigma_tensor):
    res_tensor=sigma_tensor.clone()
    res_tensor=sigma_tensor-1
    return (1,res_tensor)


def px(sigma):
    sigma_pair=(1,(sigma+1)%2)
    return sigma_pair
def py(sigma):
    sigma_pair=((1j)*((-1)**(sigma)),(sigma+1)%2)
    return sigma_pair
def pz(sigma):
    sigma_pair=((-1)**(sigma),sigma)
    return sigma_pair
def p0(sigma):
    return (1,sigma)
def pexpz(sigma):
    sigma_exp=(-1)**(sigma)
    return (np.exp(1j*np.pi*sigma_exp/4),sigma)

def pexpz3(sigma):
    sigma_exp=(-1)**(sigma)
    return (np.exp(1j*2*np.pi*sigma_exp/3),sigma)

def pexpz6(sigma):
    sigma_exp=(-1)**(sigma)
    return (np.exp(1j*2*np.pi*sigma_exp/6),sigma)

def qkx(sigma):
    q=sigma[0]*qvecs[0]+sigma[1]*qvecs[1]
    return (q[0],sigma)
def qkx_tensor(sigma_tensor,qs=qvecs):

    res_tensor=torch.complex(sigma_tensor.float(),torch.zeros_like(sigma_tensor).float())
    qs=torch.complex(torch.tensor(qs).float(),torch.zeros_like(torch.tensor(qs).float()))
    res_tensor=res_tensor@torch.tensor(qs)
    
    
    #Because you're goinf to take the product of the coeffs over both dof entries
    qx_tensor=torch.ones_like(res_tensor)
    qx_tensor[:,0]=res_tensor[:,0]#NOTE! The minus sign is applied in the g function so you don't have it here.

    
    return (qx_tensor,sigma_tensor)

def qky(sigma):
    q=sigma[0]*qvecs[0]+sigma[1]*qvecs[1]
    return (q[1],sigma)

def qky_tensor(sigma_tensor,qs=qvecs):
    
    res_tensor=torch.complex(sigma_tensor.float(),torch.zeros_like(sigma_tensor).float())
    qs=torch.complex(torch.tensor(qs).float(),torch.zeros_like(torch.tensor(qs).float()))
    res_tensor=res_tensor@torch.tensor(qs)
    
    
    #Because you're goinf to take the product of the coeffs over both dof entries
    qy_tensor=torch.ones_like(res_tensor)
    qy_tensor[:,1]=res_tensor[:,1]

    
    return (qy_tensor,sigma_tensor)
#Now let's build up the tunnelling matrices
def t1_minus(sigma):
    newsigma=(sigma[0]+1,sigma[1]+1)#i.e this is equivalent to subtracting q1 from the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)

def t1_minus_tensor(sigma_tensor):
    res_tensor=sigma_tensor.clone()
    res_tensor=sigma_tensor+1
    return (1,res_tensor)

def t2_plus(sigma):
    newsigma=(sigma[0]+1,sigma[1])#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)

def t2_plus_tensor(sigma_tensor):
    res_tensor=sigma_tensor.clone()
    res_tensor[:,0]+=res_tensor[:,0]+1
    return (1,res_tensor)

def t2_minus(sigma):
    newsigma=(sigma[0]-1,sigma[1])#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)

def t2_minus_tensor(sigma_tensor):
    res_tensor=sigma_tensor.clone()
    res_tensor[:,0]+=res_tensor[:,0]-1
    return (1,res_tensor)

def t3_plus(sigma):
    newsigma=(sigma[0],sigma[1]+1)#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)

def t3_plus_tensor(sigma_tensor):
    res_tensor=sigma_tensor.clone()
    res_tensor[:,1]+=res_tensor[:,1]+1
    return (1,res_tensor)

def t3_minus(sigma):
    newsigma=(sigma[0],sigma[1]-1)#i.e this is equivalent to adding q1 to the state. - I guess this is not good, because you're not tying it to your definition of the qs!
    return (1,newsigma)

def t3_minus_tensor(sigma_tensor):
    res_tensor=sigma_tensor.clone()
    res_tensor[:,1]+=res_tensor[:,1]-1
    return (1,res_tensor)

    


#sigma is here understood to be the pair of q states.
def t0(sigma):
    return (1,sigma)

def tz(sigma):
    return ((2*sigma[0]+3*sigma[1])%6,sigma)

def tqx(sigma,qs=qvecs):
    q=sigma[0]*qs[0]+sigma[1]*qs[1]
    qx=q[0]
    
    return (-qx,sigma)#Note - sign because it's k-Q

def tqx_tensor(sigma_tensor,qs=qvecs):
    
    res_tensor=torch.complex(sigma_tensor.float(),torch.zeros_like(sigma_tensor).float())
    qs=torch.complex(torch.tensor(qs).float(),torch.zeros_like(torch.tensor(qs).float()))
    res_tensor=res_tensor@torch.tensor(qs)
    
    
    #Because you're goinf to take the product of the coeffs over both dof entries
    qx_tensor=torch.ones_like(res_tensor)
    qx_tensor[:,0]=-res_tensor[:,0]

    
    return (qx_tensor,sigma_tensor)


def tqy(sigma,qs=qvecs):
    q=sigma[0]*qs[0]+sigma[1]*qs[1]
    qy=q[1]
    
    return (-qy,sigma)#Note - sign because it's k-Q

def tqy_tensor(sigma_tensor,qs=qvecs):
    res_tensor=torch.complex(sigma_tensor.float(),torch.zeros_like(sigma_tensor).float())
    qs=torch.complex(torch.tensor(qs).float(),torch.zeros_like(torch.tensor(qs).float()))
    res_tensor=res_tensor@torch.tensor(qs)
    
    
    #Because you're goinf to take the product of the coeffs over both dof entries
    qy_tensor=torch.ones_like(res_tensor)
    qy_tensor[:,1]=-res_tensor[:,1]

    
    return (qy_tensor,sigma_tensor)





linear_terms_kx_save=[({0:p0,1:t0,2:px,3:p0},gkx,[np.cos,g00],['theta','v'],[1/2,1]),({0:pz,1:t0,2:py,3:p0},gkx,[np.sin,g00],['theta','v'],[-1/2,1])]
linear_terms_ky_save=[({0:p0,1:t0,2:py,3:p0},gky,[np.cos,g00],['theta','v'],[1/2,1]),({0:pz,1:t0,2:px,3:p0},gky,[np.sin,g00],['theta','v'],[1/2,1])]
linear_terms_constant_save=[({0:p0,1:qkx_tensor,2:px,3:p0},g00,[np.cos,g00,np.sin],['theta','v','theta'],[1/2,1,1/2]),({0:pz,1:qkx_tensor,2:py,3:p0},g00,[np.sin,g00,np.sin],['theta','v','theta'],[-1/2,1,1/2]),
                            ({0:p0,1:qky_tensor,2:py,3:p0},g00,[np.cos,g00,np.sin],['theta','v','theta'],[1/2,1,1/2]),({0:pz,1:qky_tensor,2:px,3:p0},g00,[np.sin,g00,np.sin],['theta','v','theta'],[1/2,1,1/2])]

test_tun_terms=[({0:px,1:t1_plus_tensor,2:p0,3:p0},gw(w0)),({0:px,1:t1_minus_tensor,2:p0,3:p0},gw(w0)),({0:px,1:t1_plus_tensor,2:px,3:p0},gw(w1*np.cos(phi*(1-1)))),({0:px,1:t1_minus_tensor,2:px,3:p0},gw(w1*np.cos(phi*(1-1)))),
                ({0:px,1:t2_plus_tensor,2:p0,3:p0},gw(w0)),({0:px,1:t2_plus_tensor,2:px,3:p0},gw(w1*np.cos(phi*(2-1)))),({0:px,1:t2_plus_tensor,2:py,3:p0},gw(w1*np.sin(phi*(2-1)))),
                ({0:px,1:t2_minus_tensor,2:p0,3:p0},gw(w0)),({0:px,1:t2_minus_tensor,2:px,3:p0},gw(w1*np.cos(phi*(2-1)))),({0:px,1:t2_minus_tensor,2:py,3:p0},gw(w1*np.sin(phi*(2-1)))),
                ({0:px,1:t3_plus_tensor,2:p0,3:p0},gw(w0)),({0:px,1:t3_plus_tensor,2:px,3:p0},gw(w1*np.cos(phi*(3-1)))),({0:px,1:t3_plus_tensor,2:py,3:p0},gw(w1*np.sin(phi*(3-1)))),
                ({0:px,1:t3_minus_tensor,2:p0,3:p0},gw(w0)),({0:px,1:t3_minus_tensor,2:px,3:p0},gw(w1*np.cos(phi*(3-1)))),({0:px,1:t3_minus_tensor,2:py,3:p0},gw(w1*np.sin(phi*(3-1))))]

test_tun_terms_save=[({0:px,1:t1_plus_tensor,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t1_minus_tensor,2:p0,3:p0},g0,[g00],['w0'],[1]),
                    ({0:px,1:t1_plus_tensor,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(1-1)]),({0:px,1:t1_minus_tensor,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(1-1)]),
                    ({0:px,1:t2_plus_tensor,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t2_plus_tensor,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(2-1)]),({0:px,1:t2_plus_tensor,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(2-1)]),
                    ({0:px,1:t2_minus_tensor,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t2_minus_tensor,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(2-1)]),({0:px,1:t2_minus_tensor,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(2-1)]),
                    ({0:px,1:t3_plus_tensor,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t3_plus_tensor,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(3-1)]),({0:px,1:t3_plus_tensor,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(3-1)]), 
                    ({0:px,1:t3_minus_tensor,2:p0,3:p0},g0,[g00],['w0'],[1]),({0:px,1:t3_minus_tensor,2:px,3:p0},g0,[g00,np.cos],['w1','phi'],[1,(3-1)]),({0:px,1:t3_minus_tensor,2:py,3:p0},g0,[g00,np.sin],['w1','phi'],[1,(3-1)])
                    ]


term_list_dic={'kx_matrix':(linear_terms_kx_save,'kx','nonint'),
                'ky_matrix':(linear_terms_ky_save,'ky','nonint'),
                'q_matrices':(linear_terms_constant_save,'qadjust','nonint'),
                'tun_matrix':(test_tun_terms_save,'tun','nonint'),
                'HK_orb':([[{0:p0,1:t0,2:p0,3:px}]],'HK_orb','HK_N'),
                'HK_rot':([[{0:p0,1:t0,2:p0,3:p0}]],'HK_rot','HK_rot'),#Note that the term list in rot is inert.
                'HK_taux':([[{0:p0,1:t0,2:px,3:p0}]],'HK_N_taux','HK_N')}