import numpy as np

def Label(_x, _shape):
    """
    Returns _x in the output shape

    Used for regression problems

    Parameters
    ----------
    _x : numpy ndarray of shape (1,)
        The input value
    _shape : tuple
        The shape of the output vector

    Returns
    -------
    np.ndarray
        _x
    """
    
    return _x.reshape(_shape)

def OneHot(_x, _shape):

    """
    Generates a one-hot encoded vector of size `size` with a 1 at the index `x`.

    Parameters
    ----------
    x : numpy ndarray of shape (1,)
        The index where the 1 should be placed.
    shape : tuple
        The shape of the output vector.

    Returns
    -------
    np.ndarray
        A one-hot encoded vector of shape (size, 1).
    """
    
    _vec = np.zeros(_shape)
    _vec[int(_x[0])] = 1
    return _vec