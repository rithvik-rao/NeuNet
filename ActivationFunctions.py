import numpy as np

#defining activation functions

def ReLU(vec, deriv = False):
    
    # p = np.sign(np.sign(vec+1))

    if deriv:
        return (vec > 0).astype(float)
    else:
        return np.maximum(0,vec)

def Sigmoid(vec, deriv = False):

    p = 1/(1+np.exp(-vec))

    if deriv:
        return p*(1-p)
    else:
        return p
    
def tanh(vec, deriv = False):

    if deriv:
        return  np.cosh(vec)**(-2)
    
    else:
        return np.tanh(vec)
    
def linear(vec, deriv = False):

    if deriv:
        return np.ones_like(vec)
    else:
        return vec

def Softmax(vec):

    vec_shifted = vec - np.max(vec)
    exp_vec = np.exp(vec_shifted)
    return exp_vec / np.sum(exp_vec)