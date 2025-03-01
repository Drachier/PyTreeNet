import qutip as qt
import numpy as np
import itertools

def create_tfi_2d(Lx, Ly, J, g, periodic=False):
    """
    Create 2D Transverse-Field Ising Hamiltonian for spin-1/2 on an Lx x Ly lattice.

    Hamiltonian: H = -J * sum_{<i,j>} sz_i sz_j  - g * sum_i sx_i

    Parameters
    ----------
    Lx, Ly : int
        Dimensions of the 2D lattice.
    J : float
        Ising coupling constant.
    g : float
        Transverse field strength (coefficient of sigma_x).
    periodic : bool, optional
        If True, enforces periodic boundary conditions along x and y.

    Returns
    -------
    H : qutip.Qobj
        The Hamiltonian operator.
    sx_ops, sy_ops, sz_ops : list of qutip.Qobj
        Spin-1/2 operators for each site (tensor-product form).
    """

    # Number of sites
    n_sites = Lx * Ly

    # Single-site Pauli operators
    sx_single = qt.sigmax()
    sy_single = qt.sigmay()
    sz_single = qt.sigmaz()

    # Identity for a single site
    identity = qt.qeye(2)

    # Construct site-index helpers
    def get_site_index(x, y):
        return x * Ly + y

    # Build sx_ops, sy_ops, sz_ops as lists of operators for each site
    sx_ops = []
    sy_ops = []
    sz_ops = []

    for i in range(n_sites):
        # Initialize a list of identities
        op_list_x = [identity] * n_sites
        op_list_y = [identity] * n_sites
        op_list_z = [identity] * n_sites

        # Replace the i-th position with the Pauli operator
        op_list_x[i] = sx_single
        op_list_y[i] = sy_single
        op_list_z[i] = sz_single

        sx_ops.append(qt.tensor(op_list_x))
        sy_ops.append(qt.tensor(op_list_y))
        sz_ops.append(qt.tensor(op_list_z))

    # Initialize Hamiltonian
    H = qt.qzero([2] * n_sites, [2] * n_sites)

    # Nearest-neighbor pairs
    neighbor_pairs = []
    for x in range(Lx):
        for y in range(Ly):
            current = get_site_index(x, y)
            # Right neighbor
            if x < Lx - 1:
                neighbor_pairs.append((current, get_site_index(x+1, y)))
            elif periodic:
                neighbor_pairs.append((current, get_site_index(0, y)))
            # Up neighbor
            if y < Ly - 1:
                neighbor_pairs.append((current, get_site_index(x, y+1)))
            elif periodic:
                neighbor_pairs.append((current, get_site_index(x, 0)))

    # Build the Ising term: -J sum_{<i,j>} sz_i sz_j
    # Each pair is counted once here.
    for (i, j) in neighbor_pairs:
        # s_z^i * s_z^j
        H += -J * sz_ops[i] * sz_ops[j]

    # Add the transverse field term: -g sum_i sigma_x^i
    for i in range(n_sites):
        H += -g * sx_ops[i]

    return H, sx_ops, sy_ops, sz_ops


class TransverseFieldIsing2D_Qutip:
    """
    Transverse Field Ising model on a 2D lattice with optional periodic boundary conditions.
    Evolved via Lindblad master equation (mesolve).
    """

    def __init__(self, Lx, Ly, J, g, periodic=False):
        """
        Initialize the 2D Transverse Field Ising model.

        Parameters
        ----------
        Lx, Ly : int
            Lattice dimensions in x and y directions.
        J : float
            Ising coupling constant (sz_i sz_j).
        g : float
            Transverse field strength (sigma_x).
        periodic : bool, optional
            If True, applies periodic boundary conditions.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.J = J
        self.g = g
        self.periodic = periodic
        self.n_sites = Lx * Ly

        # Build Hamiltonian and spin operators
        self.H, self.sx_ops, self.sy_ops, self.sz_ops = create_tfi_2d(
            Lx, Ly, J, g, periodic
        )

    def uniform_product_state(self, local_state):
        """
        Create a uniform product state across all lattice sites.

        Parameters
        ----------
        local_state : array_like
            Single-site state vector, e.g. [1, 0] for |up>,
            or [1/sqrt(2), 1/sqrt(2)] for an X-eigenstate.

        Returns
        -------
        state : qutip.Qobj
            The tensor product state of all sites in the same local_state.
        """
        single_site = qt.Qobj(local_state, dims=[[2], [1]])
        # Create the product state
        state = qt.tensor([single_site] * self.n_sites)
        return state

    def alternating_product_state(self, stateA, stateB, pattern="checkerboard"):
        """
        Create an alternating product state (e.g., checkerboard up/down).

        Parameters
        ----------
        stateA, stateB : array_like
            State vectors for two different sublattices or patterns.
        pattern : str
            - "checkerboard": (x+y even) -> A, (x+y odd) -> B
            - "half_random": randomly pick half the sites as A, half as B, etc.

        Returns
        -------
        product_state : qutip.Qobj
            The combined tensor product state.
        """
        # Convert states
        stateA_qobj = qt.Qobj(stateA, dims=[[2], [1]])
        stateB_qobj = qt.Qobj(stateB, dims=[[2], [1]])

        if pattern == "checkerboard":
            # checkerboard distribution
            site_states = []
            for x in range(self.Lx):
                for y in range(self.Ly):
                    if (x + y) % 2 == 0:
                        site_states.append(stateA_qobj)
                    else:
                        site_states.append(stateB_qobj)
            product_state = qt.tensor(site_states)

        elif pattern == "half_random":
            # or any random half pattern
            import random
            all_sites = list(range(self.n_sites))
            random.shuffle(all_sites)
            half = self.n_sites // 2
            # first half is A, second half is B
            site_assignment = ["A"] * half + ["B"] * (self.n_sites - half)
            # shuffle back to site order
            unshuffled = [None]*self.n_sites
            for i, site_idx in enumerate(all_sites):
                unshuffled[site_idx] = site_assignment[i]

            site_states = []
            for assignment in unshuffled:
                if assignment == "A":
                    site_states.append(stateA_qobj)
                else:
                    site_states.append(stateB_qobj)

            product_state = qt.tensor(site_states)

        else:
            raise ValueError("Invalid pattern. Use 'checkerboard' or 'half_random'.")

        return product_state

    def build_c_ops(self, gamma_relax=0.0, gamma_deph=0.0):
        """
        Build collapse operators for Lindblad dynamics.

        Parameters
        ----------
        gamma_relax : float
            Relaxation rate (associated with sigma^-).
        gamma_deph : float
            Dephasing rate (associated with sigma_z).

        Returns
        -------
        c_ops : list
            List of collapse operators for each site.
        """
        c_ops = []
        # Single-site lowering operator for spin-1/2: sigma^- = |0><1| (if |0> is up, |1> is down)
        # or equivalently (sx + i sy)/2, but we can use destroy(2) if qubit states are [|0>,|1>].
        for i in range(self.n_sites):
            if gamma_relax != 0.0:
                op_list = [qt.qeye(2)] * self.n_sites
                op_list[i] = qt.destroy(2)  # same as |1><0| for a 2-level system
                c_ops.append(np.sqrt(gamma_relax) * qt.tensor(op_list))

            if gamma_deph != 0.0:
                op_list = [qt.qeye(2)] * self.n_sites
                op_list[i] = qt.sigmaz()
                c_ops.append(np.sqrt(gamma_deph) * qt.tensor(op_list))

        return c_ops

    def evolve_system_lindblad(self, psi0, end_time, dt, c_ops=None, e_ops=None, progress_bar=False):
        """
        Evolve the system under the Lindblad master equation (mesolve).

        Parameters
        ----------
        psi0 : qutip.Qobj
            Initial state vector.
        end_time : float
            Final time up to which we evolve the system.
        dt : float
            Time step for recording.
        c_ops : list of qutip.Qobj, optional
            List of collapse operators (if not provided, no dissipation).
        e_ops : list of qutip.Qobj, optional
            List of operators for expectation values.
        progress_bar : bool, optional
            Show or hide the QuTiP progress bar.

        Returns
        -------
        result : qutip.mesolve.Result
            The result object from mesolve, containing times, states, and expectation values.
        """
        times = np.arange(0, end_time + dt, dt)

        if c_ops is None:
            c_ops = []

        if e_ops is None:
            # By default, record none or you can pick something like total magnetization, etc.
            e_ops = []

        result = qt.mesolve(self.H, psi0, times, c_ops, e_ops, progress_bar=progress_bar)
        return result

    def calculate_single_site_expectation(self, op_list, state):
        """
        Compute expectation values <op_i> for each site i, given a list of single-site ops.

        Parameters
        ----------
        op_list : list of Qobj
            Operators such as self.sx_ops, self.sz_ops, etc.
        state : Qobj
            A state vector or density matrix.

        Returns
        -------
        expect_values : list of float
            Expectation value for each site.
        """
        return [qt.expect(op_i, state) for op_i in op_list]

    def total_magnetization_z(self, state):
        """
        Compute total magnetization in the z direction: sum_i <sz_i>.

        Parameters
        ----------
        state : Qobj
            State vector or density matrix.

        Returns
        -------
        mz : float
            Expectation value of sum_i sz_i.
        """
        total_sz = sum(self.sz_ops)
        return qt.expect(total_sz, state)

    def total_magnetization_x(self, state):
        """
        Compute total magnetization in the x direction: sum_i <sx_i>.
        """
        total_sx = sum(self.sx_ops)
        return qt.expect(total_sx, state)

    # You can add more methods for correlation functions, etc.
