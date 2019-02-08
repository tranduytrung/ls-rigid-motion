import numpy as np

def get_rigid_transform(p, q, w):
    """ get the rigid transformation applied on q

    A transformation like Rp + T = q
    Find R and T by SVD method

    Parameter:
        p (matrix): columns vectors, d x n matrix: vectors before applied Q and T
        q (matrix): columns vectors, d x n matrix: vectors after applied Q and T
        w (vectors): n elements: weight for the vectors
    
    Return:
        r (matrix): d x d roration matrix
        t (vector): d x 1 translation vector

    Referenece: Least-Squares Rigid Motion Using SVD, 
    Olga Sorkine-Hornung, Michael Rabinovich
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    """

    # ensure the same amount of points
    assert np.shape(p) == np.shape(q)

    # number of points and dim
    n = np.shape(p)[1]
    d = np.shape(p)[0]

    # ensure the length of weight vector
    assert len(w) == n

    # 1. compute weighted centroids
    sum_w = np.sum(w)
    # calc centroids
    p_mean = np.dot(p, w) / sum_w
    q_mean = np.dot(q, w) / sum_w
    p_mean.shape = (d, 1)
    q_mean.shape = (d, 1)

    # 2. compute centered vectors
    x = p - p_mean
    y = q - q_mean

    # 3. Compute Covariance matrix
    w_m = np.diag(w)
    s = x.dot(w_m).dot(y.T)

    # 4. Compute SVD of s => roration matrix r
    u, _, vh = np.linalg.svd(s)
    uh = u.T
    v = vh.T
    det_vuh = np.linalg.det(v.dot(uh)) # det(V.UT)
    m_m = np.eye(u.shape[1]) # matrix M in the note
    m_m[-1, -1] = det_vuh
    r = v.dot(m_m).dot(uh)

    # 5. Compute optimal translation
    t = q_mean - r.dot(p_mean)

    return r, t


def create_translate2d(x, y):
    """ create translation matrix
    
    Parameter:
        x (number): translate in x-axis
        y (number): translate in y-axis
    
    Return:
        return a 3x3 matrix
    """

    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float)


def create_translate3d(x, y, z):
    """ create translation matrix
    
    Parameter:
        x (number): translate in x-axis
        y (number): translate in y-axis
        z (number): translate in z-axis
    
    Return:
        return a 4x4 matrix
    """

    return np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float
    )


def create_rotate2d(d):
    """ create rotation (counter clockwise) transformation in 2D world
    
    Parameter:
        d (number): degree of rotation
    
    Return:
        return a 3x3 matrix
    """
    return np.array(
        [[np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1]], dtype=float
    )


def create_rotate3d(x=0, y=0, z=0):
    """ create rotation (counter clockwise) transformation in 3D world
    
    Parameters:
        x (number): degree of rotation on x-axis
        y (number): degree of rotation on y-axis
        z (number): degree of rotation on z-axis
    
    Return:
        return a 4x4 matrix
    """

    rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x), -np.sin(x), 0],
            [0, np.sin(x), np.cos(x), 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    ry = np.array(
        [
            [np.cos(y), 0, np.sin(y), 0],
            [0, 1, 0, 0],
            [-np.sin(y), 0, np.cos(y), 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    rz = np.array(
        [
            [np.cos(z), -np.sin(z), 0, 0],
            [np.sin(z), np.cos(z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    return np.dot(rx, ry).dot(rz)
