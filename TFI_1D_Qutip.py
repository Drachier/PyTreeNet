import qutip as qt
import numpy as np
import time
import gc


class TFI_1D_Qutip:
    """
    Transverse Field Ising model on a 1D chain with optional periodic boundary conditions.
    """

    def __init__(self, L, coupling, ext_magn, periodic=False, use_single_precision=False):
        """
        Initialize the 1D Transverse Field Ising model with memory optimizations.

        Parameters
        ----------
        L : int
            Length of the chain.
        coupling : float
            Ising coupling constant (sx_i sx_j).
        ext_magn : float
            Transverse field strength (sigma_z).
        periodic : bool, optional
            If True, applies periodic boundary conditions.
        use_single_precision : bool, optional
            If True, uses single precision (complex64) instead of double precision
            to reduce memory usage by approximately 50%.
        """
        self.L = L
        self.coupling = coupling
        self.ext_magn = ext_magn
        self.periodic = periodic
        self.n_sites = L
        
        # Optimize QuTiP settings for memory efficiency
        qt.settings.core["use_openmp"] = True
        qt.settings.core["auto_tidyup"] = True
        qt.settings.core["auto_tidyup_atol"] = 1e-10
        qt.settings.core["sparse_dense_matvec"] = True
        
        # Set precision if requested (use with caution as it affects accuracy)
        self.using_single_precision = use_single_precision
        if use_single_precision:
            print("Using single precision mode for reduced memory usage")
            qt.settings.core["precision"] = "single"
        else:
            qt.settings.core["precision"] = "double"


        self.hamiltonian, self.sx_ops, self.sy_ops, self.sz_ops = self.create_tfi_1d(
            L, coupling, ext_magn, periodic
        )

    def create_tfi_1d(self, L, coupling, ext_magn, periodic=False):
        """
        Create 1D Transverse-Field Ising Hamiltonian for spin-1/2 on a chain of length L.
        Uses optimized sparse matrix construction for larger systems.

        Hamiltonian: H = -coupling * sum_{<i,j>} sx_i sx_j  - ext_magn * sum_i sz_i

        Parameters
        ----------
        L : int
            Length of the 1D chain.
        coupling : float
            Ising coupling constant.
        ext_magn : float
            Transverse field strength (coefficient of sigma_z).
        periodic : bool, optional
            If True, enforces periodic boundary conditions.

        Returns
        -------
        hamiltonian : qutip.Qobj
            The Hamiltonian operator.
        sx_ops, sy_ops, sz_ops : list of qutip.Qobj
            Spin-1/2 operators for each site.
        """
        # Define spin operators for a single site
        si = qt.qeye(2)
        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()

        # Lists to store single-site operators
        sx_list, sy_list, sz_list = [], [], []
        
        # Construct single-site operators efficiently
        for i in range(L):
            op_list = [si for _ in range(L)]
            
            # Replace the i-th position with the Pauli operator
            op_list[i] = sx
            sx_list.append(qt.tensor(op_list))
            
            op_list[i] = sy
            sy_list.append(qt.tensor(op_list))
            
            op_list[i] = sz
            sz_list.append(qt.tensor(op_list))

        # Initialize Hamiltonian using qzero for better memory efficiency
        hamiltonian = qt.qzero([2] * L)
        
        # Add nearest-neighbor coupling terms (now using sx instead of sz)
        for i in range(L-1):
            hamiltonian -= coupling * sx_list[i] * sx_list[i+1]
        
        # Add periodic boundary term if requested (now using sx instead of sz)
        if periodic:
            hamiltonian -= coupling * sx_list[L-1] * sx_list[0]
        
        # Add transverse field terms (now using sz instead of sx)
        for i in range(L):
            hamiltonian -= ext_magn * sz_list[i]
        
        return hamiltonian, sx_list, sy_list, sz_list

    def _create_operators_optimized(self):
        """
        Create Hamiltonian and spin operators with enhanced memory efficiency.
        Uses direct tensor products and sparse matrices throughout.
        """
        # Define spin operators
        si = qt.qeye(2)
        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()
        
        # Lists to store operators
        sx_list = []
        sy_list = []
        sz_list = []
                
        # Create operators 
        for i in range(self.L):
            # Create operators using list comprehension for efficiency
            op_list_x = [si for _ in range(self.L)]
            op_list_y = [si for _ in range(self.L)]
            op_list_z = [si for _ in range(self.L)]
            
            # Set the operator at position i
            op_list_x[i] = sx
            op_list_y[i] = sy
            op_list_z[i] = sz
            
            # Create tensor products in one operation
            sx_list.append(qt.tensor(op_list_x))
            sy_list.append(qt.tensor(op_list_y))
            sz_list.append(qt.tensor(op_list_z))
            
            # Force cleanup to avoid memory buildup
            if (i+1) % 5 == 0:
                gc.collect()
        
        # Build Hamiltonian 
        hamiltonian = qt.qzero([2] * self.L)
        
        # Build coupling terms 
        coupling_terms = []
        for i in range(self.L-1):
            coupling_terms.append(sz_list[i] * sz_list[i+1])
            
        # Add all coupling terms at once
        if coupling_terms:
            hamiltonian -= self.coupling * sum(coupling_terms)
        
        # Add periodic boundary if requested
        if self.periodic:
            hamiltonian -= self.coupling * sz_list[self.L-1] * sz_list[0]
        
        # Add all field terms at once (more efficient than one by one)
        hamiltonian -= self.ext_magn * sum(sx_list)
        
        # Clean up small numerical noise
        hamiltonian.tidyup()
        
        return hamiltonian, sx_list, sy_list, sz_list

    def uniform_product_state(self, local_state):
        """
        Create a uniform product state across all lattice sites.

        Parameters
        ----------
        local_state : array_like
        Returns
        -------
        state : qutip.Qobj
            The tensor product state of all sites in the same local_state.
        """
        single_site = qt.Qobj(local_state, dims=[[2], [1]])
        # Create the product state
        return qt.tensor([single_site] * self.n_sites)

    def alternating_product_state(self, stateA, stateB):
        """
        Create an alternating product state (e.g., alternating up/down).

        Parameters
        ----------
        stateA, stateB : array_like
            State vectors for two different alternating sites.

        Returns
        -------
        product_state : qutip.Qobj
            The combined tensor product state.
        """
        # Convert states
        stateA_qobj = qt.Qobj(stateA, dims=[[2], [1]])
        stateB_qobj = qt.Qobj(stateB, dims=[[2], [1]])

        site_states = []
        for i in range(self.n_sites):
            if i % 2 == 0:
                site_states.append(stateA_qobj)
            else:
                site_states.append(stateB_qobj)
                
        product_state = qt.tensor(site_states)
        return product_state

    def build_jump_operators(self, relaxation_rate, dephasing_rate):
        """
        Build jump operators for Lindblad dynamics.
        Optimized to only add necessary operators when rates are non-zero.

        Parameters
        ----------
        relaxation_rate : float
            Relaxation rate (associated with sigma^-).
        dephasing_rate : float
            Dephasing rate (associated with sigma_z).

        Returns
        -------
        jump_operators : list
            List of jump operators for each site.
        """
        c_ops = []

        for i in range(self.n_sites):
            if relaxation_rate != 0.0:
                op_list = [qt.qeye(2)] * self.n_sites
                op_list[i] = qt.destroy(2)  
                c_ops.append(np.sqrt(relaxation_rate) * qt.tensor(op_list))

            if dephasing_rate != 0.0:
                op_list = [qt.qeye(2)] * self.n_sites
                op_list[i] = qt.sigmaz()
                c_ops.append(np.sqrt(dephasing_rate) * qt.tensor(op_list))

        return c_ops
    
    def evolve_system_lindblad(self, 
                               initial_state, 
                               final_time, 
                               time_step_size, 
                               jump_operators=None, 
                               e_ops=None,
                               options=None):
        """
        Evolve the system under the Lindblad master equation using MESolver.
        Memory-optimized with sparse matrix support and advanced solver options.

        Parameters
        ----------
        initial_state : qutip.Qobj
            Initial state vector or density matrix.
        final_time : float
            Final time up to which we evolve the system.
        time_step_size : float
            Time step for recording.
        jump_operators : list of qutip.Qobj, optional
            List of jump operators (if not provided, no dissipation).
        e_ops : list of qutip.Qobj, optional
            List of operators for expectation values.
        progress_bar : bool, optional
            Show or hide the QuTiP progress bar.

        Returns
        -------
        result, solve_time : tuple
            The result object from MESolver.run and the time taken to solve.
        """
        times = np.arange(0, final_time + time_step_size, time_step_size)

        if jump_operators is None:
            jump_operators = []

        if e_ops is None:
            e_ops = []
            
        # Set solver options for larger systems
        if options is None:
            options = {
                # ODE Integration Method
                'method': 'bdf',  # Options: 'adams', 'bdf', 'lsoda', 'dop853', 'vern9', etc. Default: 'adams'
                
            # Error Control
            'atol': 1e-8,     # Absolute tolerance. Default: 1e-8
            'rtol': 1e-6,     # Relative tolerance. Default: 1e-6
            'nsteps': 1000,  # Maximum number of internal steps allowed in one time step. Default: 1000
                        
            # Progress Reporting
            'progress_bar': 'tqdm',     # Options: 'text', 'enhanced', 'tqdm', '' (or False to disable). Default: None
            }
        
        # Create MESolver instance with Hamiltonian and collapse operators
        solver = qt.MESolver(self.hamiltonian, c_ops=jump_operators, options=options)

        # Time the solution
        start_time = time.time()
        
        # Run the solver
        result = solver.run(initial_state, times, e_ops=e_ops)
        
        # Calculate the time taken
        solve_time = time.time() - start_time
        print(f"Evolution completed in {solve_time:.2f} seconds")
                
        return result, solve_time
    
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

    def local_magnetisation_z(self, state):
        """
        Compute local magnetisation in the z direction: <sz_i> for each site.

        Parameters
        ----------
        state : Qobj
            State vector or density matrix.

        Returns
        -------
        mz : list of float
            Expectation values of sz_i for each site.
        """
        return self.calculate_single_site_expectation(self.sz_ops, state)