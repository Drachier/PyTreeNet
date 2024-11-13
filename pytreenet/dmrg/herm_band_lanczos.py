import numpy as np
from typing import Dict, Any, Union, Optional, Tuple

def save_result(aflag: int, V: np.ndarray, Vh: np.ndarray, mc: int, 
               Iv: Dict, T: np.ndarray, rho: np.ndarray, tol: Dict,
               D: Optional[np.ndarray] = None, 
               W: Optional[np.ndarray] = None,
               Wh: Optional[np.ndarray] = None,
               pc: Optional[int] = None,
               Iw: Optional[Dict] = None,
               Tt: Optional[np.ndarray] = None,
               eta: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Python translation of save_result.m
    Saves algorithm results in a structured dictionary
    """
    result = {}
    
    if aflag == 0:
        result.update({
            'V': V,
            'Vh_defl': Vh,
            'mc': mc,
            'Iv': Iv,
            'rho': rho,
            'H': T,
            'Iv': Iv
        })
    elif aflag == 1:
        result.update({
            'V': V,
            'Vh_defl': Vh,
            'mc': mc,
            'Iv': Iv,
            'rho': rho,
            'T': T,
            'W': W,
            'Wh_defl': Wh,
            'pc': pc,
            'Iw': Iw,
            'eta': eta,
            'Tt': Tt,
            'D': D
        })
    elif aflag == 2:
        result.update({
            'V': V,
            'Vh_defl': Vh,
            'mc': mc,
            'Iv': Iv,
            'rho': rho,
            'T': T
        })
    elif aflag == 3:
        result.update({
            'V': V,
            'Vh_defl': Vh,
            'mc': mc,
            'Iv': Iv,
            'rho': rho,
            'T': T,
            'D': D
        })
        
    result['tol'] = tol
    return result

def deflation(vw_f: int, n: int, m: int, mc: int, fv: int, 
             Vh: np.ndarray, R: np.ndarray, I: Dict, 
             d_f: int, d_t: float, nA: float) -> Tuple[int, int, np.ndarray, Dict, float]:
    """
    Python translation of deflation.m with consistent 0-based indexing
    """
    itmp = I['ph'][0]  # Already 0-based
    nv = np.linalg.norm(Vh[:, itmp])
    
    if n < mc:
        if (d_f == 1) or (d_f == 2):
            def_fac = np.linalg.norm(R[:, n+m-mc])
        else:
            def_fac = 1
    else:
        if (d_f == 1) or (d_f == 3):
            def_fac = nA
        else:
            def_fac = 1
            
    act_defl_tol = def_fac * d_t
    
    if nv > act_defl_tol:
        fv = 1
    else:
        print("\n**--------------------------------------**")
        
        if vw_f == 1:
            print("** Deflation of next candidate v vector **")
        elif vw_f == -1:
            print("** Deflation of next candidate w vector **")
        else:
            print("** Deflation of next candidate vector **")
            
        print("**--------------------------------------**")
        print(f"  Iteration index (n)    : {n}")
        print(f"  Deflation Tolerance    : {act_defl_tol}")
        print(f"  Norm of deflated vector: {nv}")
        
        if vw_f == 1:
            print(f"  New right block size   : {mc-1}")
        elif vw_f == -1:
            print(f"  New left block size    : {mc-1}")
        else:
            print(f"  New block size         : {mc-1}")
            
        print("**--------------------------------------**\n")
        
        # Shift the pointers of the candidate vectors
        for k in range(mc - 1):
            I['ph'][k] = I['ph'][k + 1]
            
        # Update I.nd, I.I, and I.pd
        if nv > 0:
            I['nd'] += 1
            I['I'] = np.append(I['I'], n-mc)
            I['pd'] = np.append(I['pd'], itmp)
        else:
            Vh[:, itmp] = np.zeros(Vh[:, itmp].shape)
            
        # Update current block size and resize I.ph
        mc -= 1
        I['ph'] = I['ph'][:mc]
        
    return mc, fv, Vh, I, nv


def check_tolerances(tol: Dict, n0: int) -> Tuple[float, int, int, float]:
    """
    Python translation of check_tolerances.m
    Handles tolerance checking and norm estimation for deflation
    """
    d_t = tol['defl_tol']
    eps = np.finfo(float).eps
    
    if (d_t < eps) or (d_t >= 1):
        print("\n**-------------------------------------**")
        print("** tol.defl_tol should be in the range **")
        print("**                                     **")
        print("**       eps <= tol.defl_tol < 1       **")
        print("**                                     **")
        print("**-------------------------------------**\n")
        
    d_f = tol['defl_flag']
    nA_f = tol['normA_flag']
    
    if (nA_f == 1) and (n0 == 0):
        nA = 0
    else:
        nA = tol['normA']
        
    return d_t, d_f, nA_f, nA


def herm_band_lanczos(A: Union[np.ndarray, callable], 
                      R: np.ndarray,
                      nmax: int,
                      sflag: Optional[int] = None,
                      tol: Optional[Dict] = None,
                      n: int = 0,
                      result: Optional[Dict] = None) -> Dict[str, Any]:
    
    if nmax is None:
        raise ValueError("** Not enough input arguments! **")
        
    if sflag is None:
        sflag = 0
    elif not isinstance(sflag, int):
        raise ValueError("** sflag needs to be an integer **")
        
    if tol is None:
        tol = {
            'defl_flag': 1,
            'defl_tol': np.sqrt(np.finfo(float).eps),
            'normA_flag': 1
        }
        
    if n < 0 or not isinstance(n, int):
        raise ValueError("** n needs to be a nonnegative integer **")
        
    if nmax <= n:
        raise ValueError("** nmax is not large enough; we need to have nmax > n **")
        
    N, m = R.shape
    
    if isinstance(A, np.ndarray):
        mvec_A = 1
        Nt1, Nt2 = A.shape
        if Nt1 != Nt2:
            raise ValueError("** The matrix A is not square **")
        if not np.allclose(A, A.conj().T):
            raise ValueError("** The matrix A is not Hermitian **")
        if Nt1 != N:
            raise ValueError("** The matrices A and R need to have the same number of rows **")
    else:
        mvec_A = 0
        
    if n > 0:
        if result is None:
            raise ValueError("** n > 0, but there is no input 'result' **")
            
        V = result['V']
        Vh_defl = result['Vh_defl']
        T = result['T']
        rho = result['rho']
        mc = result['mc']
        Iv = result['Iv']
        n_check = result['n']
        sflag = result['sflag']
        tol = result['tol']
        exh_flag = result['exh_flag']
        
        if exh_flag > 0:
            print("\n**---------------------------------**")
            print("** Previous run ended due to an    **")
            print("** exhausted block Krylov subspace **")
            print("**---------------------------------**\n")
            return result
            
        if n != n_check:
            raise ValueError("** n does not match the value of n in result **")
            
        n1 = n
        
    else:
        V = np.zeros((N, 0), dtype=complex)
        Vh_defl = np.zeros((N, m), dtype=complex)
        Vh_defl[:, :m] = R
        T = np.zeros((0, 0), dtype=complex)
        rho = np.zeros((0, 0), dtype=complex)
        mc = m
        Iv = {'v': np.array([], dtype=int)}
        
        if sflag == 0:
            Iv['av'] = np.arange(m+1)  # 0-based
            
        Iv['ph'] = np.arange(m)  # 0-based
        Iv['I'] = np.array([], dtype=int)
        Iv['pd'] = np.array([], dtype=int)
        Iv['nd'] = 0
        n1 = 0
        exh_flag = 0
        
    result = {'m': m}
    
    defl_tol, defl_flag, normA_flag, normA = check_tolerances(tol, n)
    
    for n in range(n1, nmax):
        foundvn = 0
        
        while foundvn == 0:
            mc, foundvn, Vh_defl, Iv, normv = deflation(0, n, m, mc, foundvn,
                                                       Vh_defl, R, Iv, defl_flag, 
                                                       defl_tol, normA)
            
            if mc == 0:
                print("**------------------------------------------**")
                print("** There are no more Krylov vectors, and so **")
                print("** the algorithm has to terminate: STOP     **")
                print("**------------------------------------------**")
                print(f"  Number of Lanczos steps performed: {n}")
                
                tol['normA'] = normA
                result = save_result(2, V, Vh_defl, mc, Iv, T, rho, tol)
                
                result['n'] = n
                result['sflag'] = sflag
                result['exh_flag'] = 1
                return result
                
        # Normalize v_n
        if sflag == 0:
            nvi = np.min(Iv['av'])  # Already 0-based
            Iv['av'] = np.setdiff1d(Iv['av'], [nvi])
        else:
            nvi = n  # 0-based
            
        # Grow V matrix as needed
        if V.shape[1] <= nvi:
            V = np.column_stack((V, np.zeros((N, nvi - V.shape[1] + 1), dtype=complex)))
            
        V[:, nvi] = Vh_defl[:, Iv['ph'][0]] / normv
        Iv['v'] = np.append(Iv['v'], nvi)
        
        # Grow rho matrix as needed
        if rho.shape[0] <= n:
            new_rows = n + 1
            rho_new = np.zeros((new_rows, m), dtype=complex)
            if rho.size > 0:
                rho_new[:rho.shape[0], :rho.shape[1]] = rho
            rho = rho_new
            
        rho[n, 0] = 0
        
        # Grow T matrix as needed
        if T.shape[0] <= n:
            new_size = n + 1
            T_new = np.zeros((new_size, new_size), dtype=complex)
            if T.size > 0:
                T_new[:T.shape[0], :T.shape[1]] = T
            T = T_new

        if n >= mc:
            T[n, n-mc] = normv
        else:
            rho[n, n-mc+m] = normv
            
        # First orthogonalization
        ivph1 = Iv['ph'][0]
        Itmp = Iv['ph'][1:mc].copy()
        Iv['ph'][:mc-1] = Itmp
        Iv['ph'][mc-1] = ivph1
        
        tmp = V[:, nvi].conj().T @ Vh_defl[:, Itmp]
        for j, idx in enumerate(Itmp):
            Vh_defl[:, idx] = Vh_defl[:, idx] - V[:, nvi] * tmp[j]
            
        Ktmp = np.flatnonzero(np.arange(mc-1) >= mc-n-1)
        if len(Ktmp) > 0:
            T[n, Ktmp-mc+n+1] = tmp[Ktmp]
        
        Ktmp = np.flatnonzero(np.arange(mc-1) < mc-n-1)
        if len(Ktmp) > 0:
            rho[n, Ktmp-mc+n+m+1] = tmp[Ktmp]
            
        # Advance block Krylov subspace
        if mvec_A == 1:
            tmpv = A @ V[:, nvi]
        else:
            tmpv = A(V[:, nvi])
            
        if normA_flag == 1:
            normA = max(normA, np.linalg.norm(tmpv))
            
        # Second orthogonalization (against deflated vectors)
        nd = Iv['nd']
        if nd > 0:
            pd_indices = Iv['pd'][:nd]
            tmp = V[:, nvi].conj().T @ Vh_defl[:, pd_indices]
            
            Itmp = Iv['I'][:nd]
            Ktmp = np.flatnonzero(Itmp >= 0)
            if len(Ktmp) > 0:
                IKtmp = Itmp[Ktmp].astype(int)
                T[n, IKtmp] = tmp[Ktmp]
                T[IKtmp, n] = tmp[Ktmp].conj()
                for idx in IKtmp:
                    tmpv = tmpv - V[:, Iv['v'][idx]] * T[idx, n]
            
            Ktmp = np.flatnonzero(Itmp < 0)
            if len(Ktmp) > 0:
                rho[n, Itmp[Ktmp]+m] = tmp[Ktmp]

        # Third orthogonalization (against previous vectors)
        Ktmp = np.arange(max(0, n-mc), n)
        if len(Ktmp) > 0:
            T[Ktmp, n] = T[n, Ktmp].conj()
            for idx in Ktmp:
                tmpv = tmpv - V[:, Iv['v'][idx]] * T[idx, n]

        T[n, n] = np.real(V[:, nvi].conj().T @ tmpv)
        Vh_defl[:, Iv['ph'][mc-1]] = tmpv - V[:, nvi] * T[n, n]
        
        # Update available slots for Lanczos vectors
        if sflag == 0:
            if n >= mc:
                if n-mc not in Iv['I']:
                    Iv['av'] = np.union1d(Iv['av'], [Iv['v'][n-mc]])
                    
    tol['normA'] = normA
    result = save_result(2, V, Vh_defl, mc, Iv, T, rho, tol)
    result.update({
        'n': n,
        'sflag': sflag,
        'exh_flag': 0
    })
    
    return result


# np.random.seed(24)

# N = 20
# A = np.random.random((N, N))
# A = (A + A.conj().T) / 2  

# m = 5  # number of starting vectors
# R = np.random.random((N, m))

# nmax = 20

# print("Running Hermitian band Lanczos method...")
# result = herm_band_lanczos(A, R, nmax)
# Afunc = lambda x: A@x
# result2 = herm_band_lanczos(Afunc, R, nmax)

# e1 = np.linalg.eigvalsh(result['T'])
# e2 = np.linalg.eigvalsh(result2['T'])
# e_exact = np.linalg.eigvalsh(A)
# print("Error in eigenvalues:", np.linalg.norm(e1-e2))
# print(e_exact,'\n', e1, '\n', e2)
# print("Orthogonality of deflated vectors:", np.allclose(result['V'].T@result['V'], np.eye(m+1)))
# print("Orthogonality of deflated vectors:", np.allclose(result2['V'].T@result2['V'], np.eye(m+1)))

# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))

# plt.imshow(np.abs(result2['T']), cmap='binary', interpolation='nearest')

# plt.colorbar(label='Absolute Value')

# plt.title("Heatmap of Sparse Matrix")

# plt.show()