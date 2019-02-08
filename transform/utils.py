import numpy as np

def reduce_cvector(v):
    """ remove the last element of column vectors

    Parameter:
        v (numpy.array): array of n x m
    
    Return:
        return new array with (n-1) x m
    """

    shape = np.shape(v)
    assert len(shape) == 2, "should be 2 dimensional array"

    return v[:-1]

def extend_cvector(v):
    """ extend the column vectors v by 1 element with value 1

    Usage: for matrix transformation

    Parameter:
        v (numpy.array): array of n x m
    
    Return:
        return new array with (n+1) x m
    """

    shape = list(np.shape(v)) # to list , so we can edit the shape
    assert len(shape) == 2, "should be 2 dimensional array"

    shape[0] = shape[0] + 1
    copy = np.ones(shape, dtype=v.dtype)
    copy[:-1] = v

    return copy

