import unittest

import numpy as np

from sketched import sketched

RTOL = 0.01
ATOL = 0.01
N_TRIALS = 10
EPSILON = 1e-8

class IsCloseMixin(object):

    def assertIsClose(self, a, b, rtol=RTOL, atol=ATOL, equal_nan=False):

        if not np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan):
            rel_err = abs(b - a) / abs(a)
            self.fail(f'{a} !~= {b} (relative error: {rel_err})')


class TestBasicEquivalence(IsCloseMixin, unittest.TestCase):

    def test_zeta_0_z_0(self):

        p = 200
        rng = np.random.default_rng(42)

        for _ in range(20):

            p_n = rng.uniform(0.5, 1.2)
            d_p = rng.uniform(0.7, 1.0)

            d = int(d_p * p)
            n = int(p / p_n)

            cov_spectrum = sketched.DiscreteSpectrum([0, 1], [1 - d / p, d / p])
            basic_equiv = sketched.BasicEquivalence(cov_spectrum, p / n)

            zs = np.linspace(-1, basic_equiv.z_0 / 2)
            zetas = [basic_equiv.get_zeta_from_z(z) for z in zs]

            equiv_stieljes = [zeta * cov_spectrum.stieltjes_transform(zeta) for zeta in zetas]

            data_min_nz_eigvals = []
            data_stieltjes = []
            data_zetas = []

            for _ in range(N_TRIALS):

                X = rng.standard_normal((n, p))
                X[:, d:] = 0

                data_spectrum = np.linalg.eigvalsh(X.T @ X / n)
                data_min_nz_eigvals.append(np.min(data_spectrum[data_spectrum > EPSILON]))
                data_stieltjes.append([z * np.mean(1 / (data_spectrum - z)) for z in zs])

                companion_spectrum = np.linalg.eigvalsh(X @ X.T / n)
                companion_stieltjes = np.asarray([np.mean(1 / (companion_spectrum - z)) for z in zs])
                data_zetas.append(-1 / companion_stieltjes)

            self.assertIsClose(np.mean(data_min_nz_eigvals), basic_equiv.z_0, rtol=0.1)
            self.assertTrue((d / n  > 1) == (basic_equiv.zeta_0 < 0))

            data_stieltjes_mean = np.mean(np.asarray(data_stieltjes), 0)
            for data_s, equiv_s in zip(data_stieltjes_mean, equiv_stieljes):
                self.assertIsClose(data_s, equiv_s, rtol=0.01)

            data_zetas_mean = np.mean(np.asarray(data_zetas), 0)
            for data_zeta, zeta in zip(data_zetas_mean, zetas):
                self.assertIsClose(data_zeta, zeta, rtol=0.01)




            
