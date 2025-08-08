"""

NeuNet Module for various Loss/Cost functions

Functions:
- Mean Squared Error (MSE)
- Cross Entrpy Loss (CrossEntropyLoss)

More to come...
"""

import numpy as np

#Cost Functions

def MSE(vec : np.ndarray, goal : np.ndarray, deriv = False):

    """
    
    Rectified Linear Unit (ReLU) function

    Parameters
    ----------

    vec : input (Scalar Array)

    goal : target (Scalar Array)

    deriv : derivative flag (default False)

    ----------
    
    """

    p = goal - vec

    if deriv:
        return -2*p/len(p)
    
    else:
        return np.sum(p**2) / len(p)
    
def CrossEntropyLoss(vec : np.ndarray, goal : np.ndarray, deriv = False, pfunc = None):

    """
    
    Rectified Linear Unit (ReLU) function

    Parameters
    ----------

    vec : input (Scalar Array)

    goal : target (Scalar Array)

    deriv : derivative flag (default False)

    pfunc : Activation function used for output layer

    ----------
    
    """

    epsilon = 1e-15
    output = np.clip(vec, epsilon, 1 - epsilon)

    if deriv:

        if pfunc.__name__ == 'Softmax':

            return output - goal
        
        else:

            quit('Derivative not defined yet .!..!...!....')

    else:
        return -np.sum(goal * np.log(output))
    
if __name__ == '__main__':
    pass