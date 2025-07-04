"""

NeuNet Module for various Activation functions

Functions:
- ReLu
- Sigmoid
- TanH
- Linear
- Softmax

More to come...
"""

import numpy as np

#defining activation functions

def ReLU(vec, deriv = False):

    """
    
    Rectified Linear Unit (ReLU) function

    Parameters
    ----------

    vec : input (Scalar Array)

    deriv : derivative flag (default False)

    ----------
    
    """
    
    # p = np.sign(np.sign(vec+1))

    if deriv:
        return (vec > 0).astype(float)
    else:
        return np.maximum(0,vec)

def Sigmoid(vec, deriv = False):

    """
    
    Sigmoid function

    Parameters
    ----------

    vec : input (Scalar Array)

    deriv : derivative flag (default False)

    ----------
    
    """

    p = 1/(1+np.exp(-vec))

    if deriv:
        return p*(1-p)
    else:
        return p
    
def TanH(vec, deriv = False):
    """
    
    Tan Hyperbolic (TanH) function

    Parameters
    ----------

    vec : input (Scalar Array)

    deriv : derivative flag (default False)

    ----------
    
    """
    if deriv:
        return  np.cosh(vec)**(-2)
    
    else:
        return np.tanh(vec)
    
def Linear(vec, deriv = False):

    """
    
    Linear function

    Parameters
    ----------

    vec : input (Scalar Array)

    deriv : derivative flag (default False)

    ----------
    
    """

    if deriv:
        return np.ones_like(vec)
    else:
        return vec

def Softmax(vec):

    """
    
    Softmax function

    Parameters
    ----------

    vec : input (Scalar Array)

    ----------
    
    derivative depends on loss function used

    """

    vec_shifted = vec - np.max(vec)
    exp_vec = np.exp(vec_shifted)
    return exp_vec / np.sum(exp_vec)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    funx = [item for item in globals().values() if callable(item)]

    xs = np.linspace(-5,5,100)

    fig, ax = plt.subplots(int(0.5*len(funx)+1) , 2 , figsize = (5,6) ,sharex=True, facecolor = '#111111')
    i,j =0,0

    for func in funx:

        ax[j][i].set_facecolor('#000000')
        ax[j][i].set_yticks([-10,0,10])
        ax[j][i].set_xticks([-10,0,10])
        ax[j][i].plot(xs,func(xs),c='w')
        ax[j][i].set_title(func.__name__ , c='w')
        ax[j][i].grid(True,which = 'major')

        i += 1

        if i == 2:
            j += 1
            i = 0
    
    ax[-1][-1].set_facecolor('#111111')

    plt.savefig('ActivationFunctionsPlotted.png')