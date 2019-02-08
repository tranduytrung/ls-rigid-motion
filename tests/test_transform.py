import unittest
import transform
import numpy as np

class TestTransform(unittest.TestCase):
    """ Test for regressing roration and translation using SVD

    The test first simulates the rorate and translation by creating
    transformation matrix and comparing with the result from the method
    """
    def test_2d(self):
        # generate 5 points in 2D
        n = 5
        p = np.random.uniform(size=(2, n))
        
        # convert to transformable
        p_ = transform.utils.extend_cvector(p)

        # apply roration and translation by random numbers
        r_mat = transform.rigid.create_rotate2d(np.random.uniform(-180, 180))
        t_mat = transform.rigid.create_translate2d(np.random.uniform(), np.random.uniform())
        trans_mat = t_mat.dot(r_mat) # rorate then translate
        q_ = trans_mat.dot(p_)

        # convert back to 2d
        q = transform.utils.reduce_cvector(q_)

        # deduce the applied transformation
        r_reg, t_reg = transform.rigid.get_rigid_transform(p, q, np.random.uniform(size=n))

        # expect small SSE
        sse_t = np.sum((t_mat[:2, 2:3] - t_reg)**2)
        sse_r = np.sum((r_mat[:2, :2] - r_reg)**2)
        self.assertLessEqual(sse_t, 10e-3)
        self.assertLessEqual(sse_r, 10e-3)

    def test_3d(self):
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

        # expect small SSE
        sse_t = np.sum((t_mat[:3, 3:4] - t)**2)
        sse_r = np.sum((r_mat[:3, :3] - r)**2)
        self.assertLessEqual(sse_t, 10e-3)
        self.assertLessEqual(sse_r, 10e-3)


    def test_3d_noise(self):
        # This test may fail due to noise unless relaxing the strict error 

        # generate n points in 3D
        n = 20
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

        # add gaussian noise to the transformed
        q = q + np.random.normal(scale=0.1, size=(3, n))

        # deduce the applied transformation
        r, t = transform.rigid.get_rigid_transform(p, q, np.random.uniform(size=n))

        # expect small SSE
        sse_t = np.sum((t_mat[:3, 3:4] - t)**2)
        sse_r = np.sum((r_mat[:3, :3] - r)**2)
        self.assertLessEqual(sse_t, 10e-3)
        self.assertLessEqual(sse_r, 10e-3)


if __name__ == "__main__":
    unittest.main()