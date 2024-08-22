import numpy as np

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