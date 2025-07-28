import numpy as np

# E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
def davidson(a, b, k, max_iter=500, conv_thrd=1E-7, deflation_min_size=2, deflation_max_size=30, iprint=False, mpi=False):
    """
    Davidson diagonalization.

    Args:
        a : Matrix
            The matrix to diagonalize.
        b : list(Vector)
            The initial guesses for eigenvectors.

    Kwargs:
        max_iter : int
            Maximal number of davidson iteration.
        conv_thrd : float
            Convergence threshold for squared norm of eigenvector.
        deflation_min_size : int
            Sub-space size after deflation.
        deflation_max_size : int
            Maximal sub-space size before deflation.
        iprint : bool
            Indicate whether davidson iteration information should be printed.
        mpi : bool
            Indicate whether mpi is used.

    Returns:
        ld : list(float)
            List of eigenvalues.
        b : list(Vector)
            List of eigenvectors.
    """

    if mpi:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        comm = MPI.COMM_WORLD
    else:
        rank = 0

    assert len(b) == k
    if deflation_min_size < k:
        deflation_min_size = k
    aa = a.diag() if hasattr(a, "diag") else None
    for i in range(k):
        for j in range(i):
            b[i] += -np.dot(b[j].conj().T, b[i]) * b[j]
        b[i] /= np.linalg.norm(b[i])
    sigma = [None] * k
    q = b[0]
    l = k
    ck = 0
    msig = 0
    m = l
    xiter = 0
    while xiter < max_iter:
        xiter += 1
        if mpi and xiter != 1:
            for i in range(msig, m):
                b[i] = comm.bcast(b[i], root=0)
        for i in range(msig, m):
            sigma[i] = a @ b[i]
            msig += 1
        if not mpi or rank == 0:
            atilde = np.zeros((m, m), dtype=sigma[0].dtype)
            for i in range(m):
                for j in range(i + 1):
                    atilde[i, j] = np.dot(b[i].conj().T, sigma[j])
                    atilde[j, i] = atilde[i, j].conj()
            ld, alpha = np.linalg.eigh(atilde)
            # b[1:m] = np.dot(b[:], alpha[:, 1:m])
            tmp = [ib.copy() for ib in b[:m]]
            for j in range(m):
                b[j] = b[j] * alpha[j, j]
            for j in range(m):
                for i in range(m):
                    if i != j:
                        b[j] += alpha[i, j] * tmp[i]
            # sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
            for i in range(m):
                tmp[i] = sigma[i].copy()
            for j in range(m):
                sigma[j] *= alpha[j, j]
            for j in range(m):
                for i in range(m):
                    if i != j:
                        sigma[j] += alpha[i, j] * tmp[i]
            for i in range(ck):
                q = sigma[i].copy()
                q += (-ld[i]) * b[i]
                qq = np.dot(q.conj().T, q)
                if np.sqrt(qq) >= conv_thrd:
                    ck = i
                    break
            # q = sigma[ck] - b[ck] * ld[ck]
            q = sigma[ck].copy()
            q += (-ld[ck]) * b[ck]
            qq = np.abs(np.dot(q.conj().T, q))
            if iprint:
                print("%5d %5d %5d %15.8f %9.2E" % (xiter, m, ck, ld[ck], qq))

            if aa is not None:
                _olsen_precondition(q, b[ck], ld[ck], aa)
        if mpi:
            qq = comm.bcast(qq if rank == 0 else None, root=0)
            ck = comm.bcast(ck if rank == 0 else None, root=0)

        if qq < 0 or np.sqrt(qq) < conv_thrd:
            ck += 1
            if ck == k:
                break
        else:
            if m >= deflation_max_size:
                m = deflation_min_size
                msig = deflation_min_size
            if not mpi or rank == 0:
                for j in range(m):
                    q += (-np.dot(b[j].conj().T, q)) * b[j]
                q /= np.linalg.norm(q)
            if m >= len(b):
                b.append(None)
                sigma.append(None)
            if not mpi or rank == 0:
                b[m] = q
            m += 1

        if xiter == max_iter:
            import warnings
            warnings.warn("Only %d converged!" % ck)
            ck = k

    if mpi:
        ld = comm.bcast(ld if rank == 0 else None, root=0)
        for i in range(0, ck):
            b[i] = comm.bcast(b[i], root=0)

    return ld[:ck], b[:ck]