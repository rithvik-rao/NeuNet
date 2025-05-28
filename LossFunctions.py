import numpy as np

#Cost Functions

def MSE(vec : np.ndarray, goal : np.ndarray, deriv = False):

    p = goal - vec

    if deriv:
        return -2*p/len(p)
    
    else:
        return np.sum(p**2) / len(p)
    
def CrossEntroyLoss(vec : np.ndarray, goal : np.ndarray, deriv = False, pfunc = None):

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