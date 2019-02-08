import numpy as np
import transform

def main():
    # test_2d()
    test_3d()

def test_3d():
    # generate n points in 3D
    n = 10
    p = np.random.uniform(size=(3, n))
    
    # convert to transformable 4d
    p_ = transform.utils.extend_cvector(p)

    # apply roration and translation by random numbers
    rx, ry, rz = np.random.uniform(-180, 180, size=3)
    r_mat = transform.rigid.create_rotate3d(rx, ry, rz)
    tx, ty, tz = np.random.uniform(size=3)
    t_mat = transform.rigid.create_translate3d(tx, ty, tz)
    trans_mat = t_mat.dot(r_mat) # rorate then translate
    q_ = trans_mat.dot(p_) # apply transform

    # convert back to 3d
    q = transform.utils.reduce_cvector(q_)

    # deduce the applied transformation
    r, t = transform.rigid.get_rigid_transform(p, q, np.random.uniform(size=n))

    print("p = ", p)
    print("q = ", q)
    print("rotation matrix = ", r_mat)
    print("translation matrix = ", t_mat)
    print("combined transform matrix = ", trans_mat)
    print("predicted rotation = ", r)
    print("predicted translation = ", t)

def test_2d():
    # generate 5 points in 2D
    p = np.random.uniform(size=(2, 10))

    # convert to transformable
    p_ = transform.utils.extend_cvector(p)

    # apply rorate 30d and translate by 0.5, 0.1
    r_mat = transform.rigid.create_rotate2d(-1)
    t_mat = transform.rigid.create_translate2d(0.01, 0.1)
    trans_mat = t_mat.dot(r_mat) # rorate then translate
    q_ = trans_mat.dot(p_)

    # convert back
    q = transform.utils.reduce_cvector(q_)

    # deduce the applied transformation
    r, t = transform.rigid.get_rigid_transform(p, q, np.random.uniform(size=10))

    print("p = ", p)
    print("q = ", q)
    print("rotation matrix = ", r_mat)
    print("translation matrix = ", t_mat)
    print("combined transform matrix = ", trans_mat)
    print("predicted rotation = ", r)
    print("predicted translation = ", t)

if __name__ == "__main__":
    main()