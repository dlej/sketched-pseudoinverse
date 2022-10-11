import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

# uses fork at https://github.com/dnbaker/FFHT
from ffht import fht


""" All sketches are LinearOperators designed to be used as follows:

    `y = A @ x`

    where `y` is the sketched vector, `A` is the sketching operator, and `x` is
    the original vector. This is the transpose of the sketching matrices 
    considered in the text of our paper.
"""


def get_count_sketch_matrix(b, t, m, normed=False, rng=None, dtype=None):
    """ notation from Charikar, Moses, Kevin Chen, and Martin Farach-Colton.
    "Finding frequent items in data streams." International Colloquium on
    Automata, Languages, and Programming. 2002.
    
    build the sketching matrix for Count Sketch for `m` items with `t` hash
    functions mapping onto `{1, ..., b}`.
    """

    if rng is None:
        rng = np.random.default_rng()

    data = rng.choice(np.asarray([-1, 1]), size=t * m, replace=True)
    if normed:
        data = data / np.sqrt(t)

    hash = rng.choice(b, size=(m, t), replace=True)
    row_ind = (hash + b * np.arange(t)[None, :]).ravel()
    col_ind = np.repeat(np.arange(m), t)

    return sparse.csc_matrix((data, (row_ind, col_ind)), shape=(b * t, m), dtype=dtype)


def get_subsample_matrix(q, p, normed=False, rng=None, dtype=None):

    if rng is None:
        rng = np.random.default_rng()
    
    data = np.ones(q)
    if normed:
        data = data * np.sqrt(p / q)

    row_ind = np.arange(q)
    col_ind = rng.choice(p, q, replace=False)

    return sparse.csc_matrix((data, (row_ind, col_ind)), shape=(q, p), dtype=dtype)


def get_fjl_P_matrix(k, d, q, normed=False, rng=None, dtype=None):
    """ notation as described in Ailon, Nir and Bernard Chazelle. "Approximate
    nearest neighbors and the fast Johnson-Lindenstrauss transform." STOC '06.
    doi:10.1137/060673096
    """

    if rng is None:
        rng = np.random.default_rng()

    nP = rng.binomial(k * d, q)
    
    data = rng.normal(loc=0.0, scale=1/np.sqrt(q), size=nP)
    if normed:
        data /= np.sqrt(k)

    ind = rng.choice(k * d, size=nP, replace=False)
    row_ind = ind // d
    col_ind = ind % d

    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(k, d), dtype=dtype)


def get_permutation_matrix(p, rng=None, dtype=None):

    if rng is None:
        rng = np.random.default_rng()

    data = np.ones(p, dtype=dtype)

    row_ind = np.arange(p)
    col_ind = rng.permutation(p)

    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(p,  p), dtype=dtype)


class CountSketch(LinearOperator):

    def __init__(self, b, t, m, normed=False, rng=None, dtype=float):

        self.A = get_count_sketch_matrix(b, t, m, normed=normed, rng=rng, dtype=dtype)
        super().__init__(dtype, self.A.shape)
    
    def _matmat(self, X):
        return self.A @ X
    
    def _adjoint(self):
        return self.A.transpose()


class SubsampleSketch(LinearOperator):

    def __init__(self, q, p, normed=False, rng=None, dtype=float):

        self.A = get_subsample_matrix(q, p, normed=normed, rng=rng, dtype=dtype)
        super().__init__(dtype, (q, p))
    
    def _matmat(self, X):
        return self.A @ X
    
    def _adjoint(self):
        return self.A.transpose()


class FastJohnsonLindenstraussSketch(LinearOperator):
    """ Implementation and notation as described in Ailon, Nir and Bernard 
    Chazelle. "Approximate nearest neighbors and the fast Johnson-Lindenstrauss 
    transform." STOC '06. doi:10.1137/060673096
    """

    def __init__(self, k, d, q=None, ortho=False, perm=False, normed=False, rng=None, dtype=float):

        if q is None and not ortho:
            raise ValueError('Must specify `q` for non-orthogonal sketch')

        if rng is None:
            rng = np.random.default_rng()
        
        self.d = d
        self.d_pow2 = get_pow2_ceil(d) 

        if ortho:
            self.P = get_subsample_matrix(k, self.d_pow2, normed=normed, rng=rng, dtype=dtype)
        else:
            self.P = get_fjl_P_matrix(k, self.d_pow2, q, normed=normed, rng=rng, dtype=dtype)

        self.D = sparse.spdiags(np.random.choice([-1, 1], size=(1, d), replace=True), [0], d, d)

        if perm:
            self.D = get_permutation_matrix(d) @ self.D

        super().__init__(dtype, (k, d))
    
    def _matmat(self, X):

        Z = self.D @ X
        Z = np.vstack((Z, np.zeros((self.d_pow2 - self.d, Z.shape[1]), dtype=Z.dtype)))
        # fht doesn't work on transpose without copy! beware!
        Z = fht(Z.T.copy()).T / np.sqrt(self.d_pow2)
        Z = self.P @ Z

        return Z
    
    def _adjoint(self):
        return FJLTranspose(self)


class FJLTranspose(LinearOperator):

    def __init__(self, fjl):

        self.fjl = fjl

        super().__init__(fjl.dtype, fjl.shape[::-1])
    
    def _matmat(self, X):

        Z = self.fjl.P.T @ X
        # fht doesn't work on transpose without copy! beware!
        Z = fht(Z.T.copy()).T / np.sqrt(self.fjl.d_pow2)
        Z = Z[:self.fjl.d, :]
        Z = self.fjl.D @ Z

        return Z
    
    def _adjoint(self):
        return self.fjl


def get_pow2_ceil(n):

    m = 1
    while m < n:
        m *= 2

    return m