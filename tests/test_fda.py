import numpy as np
import unittest

import finite_depth_analysis as fda

c0 = 299792458
n0 = 1.5
c = c0/n0


def h_old(lambdas, lambda_prime, Z, r=-1, mode=1, k0=0):

    omega = 2*np.pi*c/lambdas
    omega_prime = 2*np.pi*c/lambda_prime

    if mode==2:
        if Z == 'inf' or Z == 'infinite':
            Z = 100E-6
        s1 = fda.s_z_tilde_dev(Z, omega_prime-omega, k0=k0)
        s3 = fda.s_z_tilde_dev(Z, omega_prime+omega, k0=k0)
        return r/2*s1 + np.conj(r)/2*s3
    elif mode==3:
        s2 = fda.s_z_tilde_dev(Z, omega_prime, k0=k0)
        return (1+np.abs(r)**2)/2*s2
    else:
        s2 = fda.s_z_tilde_dev(Z, omega_prime, k0=k0)
        if Z == 'inf' or Z == 'infinite':
            Z = 100E-6
        s1 = fda.s_z_tilde_dev(Z, omega_prime-omega, k0=k0)
        s3 = fda.s_z_tilde_dev(Z, omega_prime+omega, k0=k0)
        return r/2*s1 + (1+np.abs(r)**2)/2*s2 + np.conj(r)/2*s3


class TestSZTildeDev(unittest.TestCase):
    def test_vectorisation(self):
        self.assertEqual(True, True)


class TestH(unittest.TestCase):
    lambdas_over = np.linspace(0.1, 1)
    lambdas = np.linspace(0.1, 1, 10)
    Z = 0.3
    r = 0.2
    k0 = 1

    def test_dimensions(self):
        A = fda.h(self.lambdas_over, self.lambdas, self.Z, r=self.r, mode=2, k0=self.k0)
        A0 = fda.h(self.lambdas_over, self.lambdas, self.Z, r=self.r, mode=3, k0=self.k0)
        self.assertEqual(A.shape[0], len(self.lambdas))
        self.assertEqual(A.shape[1], len(self.lambdas_over))
        self.assertEqual(A0.shape[0], len(self.lambdas))
        self.assertEqual(A0.shape[1], len(self.lambdas_over))

    def test_old(self):
        A_new = fda.h(self.lambdas_over, self.lambdas, self.Z, r=self.r, mode=2, k0=self.k0)
        A0_new = fda.h(self.lambdas_over, self.lambdas, self.Z, r=self.r, mode=3, k0=self.k0)
        A = np.zeros((len(self.lambdas), len(self.lambdas_over)), dtype=complex)
        A0 = np.zeros((len(self.lambdas), len(self.lambdas_over)), dtype=complex)
        for i, lambda_prime in enumerate(self.lambdas):
            A[i, :] = h_old(self.lambdas_over, lambda_prime, self.Z, r=self.r, mode=2, k0=self.k0)
            A0[i, :] = h_old(self.lambdas_over, lambda_prime, self.Z, r=self.r, mode=3, k0=self.k0)
        np.testing.assert_array_almost_equal(A_new, A)
        np.testing.assert_array_almost_equal(A0_new, A0)


if __name__ == '__main__':
    unittest.main()
