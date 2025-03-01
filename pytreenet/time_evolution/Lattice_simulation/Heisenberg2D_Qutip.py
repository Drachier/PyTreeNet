try:
    import qutip as qt
    import numpy as np
    import random
    from .util import get_neighbors_with_distance_HV, get_neighbors_with_distance_HDV

    # Assume get_neighbors_with_distance_HV and get_neighbors_with_distance_HDV are defined elsewhere

    def create_anisotropic_heisenberg_2d(Lx, Ly, spin, Jx, Jy, Jz, h, periodic=False):
        """
        Create 2D Anisotropic Heisenberg Hamiltonian with external magnetic field.

        Parameters:
        -----------
        Lx, Ly : int
            Lattice dimensions in x and y directions.
        spin : float
            Spin quantum number (e.g., 0.5 for spin-1/2).
        Jx, Jy, Jz : float
            Anisotropic coupling constants in x, y, z directions respectively.
        h : float
            Uniform external magnetic field in the z-direction.
        periodic : bool, optional
            If True, applies periodic boundary conditions. Default is False.

        Returns:
        --------
        H : qutip.Qobj
            The constructed Hamiltonian.
        Sx_ops, Sy_ops, Sz_ops : list of qutip.Qobj
            Lists of spin operators for each site.
        """

        n_sites = Lx * Ly
        dim = int(2 * spin + 1)  # Dimension of single spin

        # Generate spin operators for a single site
        Sx_single = qt.spin_Jx(spin)
        Sy_single = qt.spin_Jy(spin)
        Sz_single = qt.spin_Jz(spin)

        # Precompute identity operator
        identity = qt.qeye(dim)

        # Create spin operators for each site using tensor products
        Sx_ops = []
        Sy_ops = []
        Sz_ops = []

        for i in range(n_sites):
            op_list = [identity] * n_sites
            op_list[i] = Sx_single
            Sx_ops.append(qt.tensor(op_list))
            
            op_list = [identity] * n_sites
            op_list[i] = Sy_single
            Sy_ops.append(qt.tensor(op_list))
            
            op_list = [identity] * n_sites
            op_list[i] = Sz_single
            Sz_ops.append(qt.tensor(op_list))

        # Initialize Hamiltonian as a sparse Qobj
        H = qt.qzero([dim]*n_sites, [dim]*n_sites)

        # Helper function to get site index from 2D coordinates
        def get_index(x, y):
            return x * Ly + y

        # Define neighbor pairs based on boundary conditions
        neighbor_pairs = []
        for x in range(Lx):
            for y in range(Ly):
                current = get_index(x, y)
                # Right neighbor
                if x < Lx - 1:
                    neighbor = get_index(x + 1, y)
                    neighbor_pairs.append((current, neighbor))
                elif periodic:
                    neighbor = get_index(0, y)
                    neighbor_pairs.append((current, neighbor))
                # Up neighbor
                if y < Ly - 1:
                    neighbor = get_index(x, y + 1)
                    neighbor_pairs.append((current, neighbor))
                elif periodic:
                    neighbor = get_index(x, 0)
                    neighbor_pairs.append((current, neighbor))

        # Construct Heisenberg interaction terms
        for (i, j) in neighbor_pairs:
            H += Jx * Sx_ops[i] * Sx_ops[j]
            H += Jy * Sy_ops[i] * Sy_ops[j]
            H += Jz * Sz_ops[i] * Sz_ops[j]

        # External magnetic field term
        for i in range(n_sites):
            H += h * Sz_ops[i]

        return H, Sx_ops, Sy_ops, Sz_ops


    class AnisotropicHeisenberg2D_Qutip:
        def __init__(self, Lx, Ly, spin, Jx, Jy, Jz, h, periodic=False):
            """
            Initialize the 2D Anisotropic Heisenberg model.

            Parameters:
            -----------
            Lx, Ly : int
                Lattice dimensions in x and y directions.
            spin : float
                Spin quantum number (e.g., 0.5 for spin-1/2).
            Jx, Jy, Jz : float
                Anisotropic coupling constants in x, y, z directions respectively.
            h : float
                Uniform external magnetic field in the z-direction.
            periodic : bool, optional
                If True, applies periodic boundary conditions. Default is False.
            """
            self.Lx = Lx
            self.Ly = Ly
            self.spin = spin
            self.Jx = Jx
            self.Jy = Jy
            self.Jz = Jz
            self.h = h
            self.periodic = periodic

            # Create Hamiltonian and spin operators using sparse tensors
            self.H, self.Sx_ops, self.Sy_ops, self.Sz_ops = create_anisotropic_heisenberg_2d(
                Lx, Ly, spin, Jx, Jy, Jz, h, periodic
            )

        def uniform_product_state(self, local_state):
            """
            Create a uniform product state across all lattice sites.

            Parameters:
            -----------
            local_state : list or np.array
                The single-site state vector.

            Returns:
            --------
            state : qutip.Qobj
                The tensor product state.
            """
            # Create the single-site state in QuTiP format
            single_site = qt.Qobj(local_state)

            # Create the product state using list and tensor
            state = [single_site] * (self.Lx * self.Ly)
            product_state = qt.tensor(state)
            return product_state

        def get_checkerboard_pattern(self):
            """
            Generate a checkerboard pattern of sites.

            Returns:
            --------
            black_sites, white_sites : lists of tuples
                Lists containing the coordinates of black and white sites.
            """
            black_sites = []
            white_sites = []
            for x in range(self.Lx):
                for y in range(self.Ly):
                    if (x + y) % 2 == 0:
                        black_sites.append((x, y))  # Append as a tuple
                    else:
                        white_sites.append((x, y))  # Append as a tuple
            return black_sites, white_sites

        def get_random_half_sites(self):
            """
            Randomly split the lattice sites into two equal halves.

            Returns:
            --------
            black_sites, white_sites : lists of tuples
                Lists containing the coordinates of black and white sites.
            """
            total_sites = self.Lx * self.Ly

            # Generate a list of all sites as tuples
            all_sites = [(x, y) for x in range(self.Lx) for y in range(self.Ly)]

            # Shuffle and split into two halves
            random.shuffle(all_sites)
            half = total_sites // 2

            black_sites = all_sites[:half]
            white_sites = all_sites[half:]

            return black_sites, white_sites

        def alternating_product_state(self, black_state, white_state, pattern):
            """
            Create an alternating product state based on a specified pattern.

            Parameters:
            -----------
            black_state : list or np.array
                The state vector for black sites.
            white_state : list or np.array
                The state vector for white sites.
            pattern : str
                The pattern type: "checkerboard" or "half_random".

            Returns:
            --------
            product_state : qutip.Qobj
                The tensor product state.
            """
            # Create single-site states in QuTiP format
            black_site_qobj = qt.Qobj(black_state)
            white_site_qobj = qt.Qobj(white_state)
            
            # Choose the pattern and get black/white site lists
            if pattern == "checkerboard":
                black_sites, white_sites = self.get_checkerboard_pattern()
            elif pattern == "half_random":
                black_sites, white_sites = self.get_random_half_sites()
            else:
                raise ValueError("Invalid pattern. Use 'checkerboard' or 'half_random'.")

            # Initialize the state list with white states as placeholders
            product_state_components = [white_site_qobj] * (self.Lx * self.Ly)
            
            # Assign black states to the specific indices
            for site in black_sites:
                idx = site[0] * self.Ly + site[1]
                product_state_components[idx] = black_site_qobj

            # Tensor product to create the full product state
            product_state = qt.tensor(product_state_components)
            
            return product_state

        def evolve_system(self, psi0, end_time, dt=0.1):
            """
            Evolve the quantum state in real-time using the Schrödinger equation.

            Parameters:
            -----------
            psi0 : qutip.Qobj
                The initial state vector.
            end_time : float
                The final time up to which the system is evolved.
            dt : float, optional
                The time step for evaluation. Default is 0.1.

            Returns:
            --------
            result : qutip.sesolve.Result
                The result object containing the evolved states.
            """
            times = np.linspace(0, end_time, int(end_time/dt) + 1)
            self.result = qt.sesolve(self.H, psi0, times)
            return self.result

        def calculate_spatial_correlation_function(self, state, distance, mode="HV"):
            """
            Calculate the spatial correlation function (sum of x-x, y-y, z-z)
            without normalization.

            Parameters:
            -----------
            state : qutip.Qobj
                The quantum state.
            distance : int
                The distance for correlation.
            mode : str, optional
                The mode of neighbor selection: "HV" or "HDV". Default is "HV".

            Returns:
            --------
            result : float
                The summed correlation value for (SxSx + SySy + SzSz) across all
                site pairs at the specified distance.
            """
            if mode == "HV":
                current_sites, neighbor_sites = get_neighbors_with_distance_HV(self.Lx, self.Ly, distance)
            elif mode == "HDV":
                current_sites, neighbor_sites = get_neighbors_with_distance_HDV(self.Lx, self.Ly, distance)
            else:
                raise ValueError("Invalid mode. Use 'HV' or 'HDV'.")

            result = 0.0
            for site1, site2 in zip(current_sites, neighbor_sites):
                # Convert 2D coordinates to 1D index
                idx1 = site1[0] * self.Ly + site1[1]
                idx2 = site2[0] * self.Ly + site2[1]

                # Expectation values for each spin component
                corr_x = qt.expect(self.Sx_ops[idx1] * self.Sx_ops[idx2], state)
                corr_y = qt.expect(self.Sy_ops[idx1] * self.Sy_ops[idx2], state)
                corr_z = qt.expect(self.Sz_ops[idx1] * self.Sz_ops[idx2], state)

                # Sum up the correlations
                result += (corr_x + corr_y + corr_z)

            return result

        def calculate_specific_site_correlation(self, state, site1, site2):
            
            result = 0
            # Convert 2D coordinates to 1D indices
            idx1 = site1[0] * self.Ly + site1[1]
            idx2 = site2[0] * self.Ly + site2[1]

            # Expectation values for each spin component
            corr_x = qt.expect(self.Sx_ops[idx1] * self.Sx_ops[idx2], state)
            corr_y = qt.expect(self.Sy_ops[idx1] * self.Sy_ops[idx2], state)
            corr_z = qt.expect(self.Sz_ops[idx1] * self.Sz_ops[idx2], state)  

            # Sum up the correlations
            result += (corr_x + corr_y + corr_z)

            return result

        def calculate_spatial_correlation_function_sz(self, state, distance, mode="HV"):
            """
            Calculate the spatial correlation function without normalization.

            Parameters:
            -----------
            state : qutip.Qobj
                The quantum state.
            distance : int
                The distance for correlation.
            mode : str, optional
                The mode of neighbor selection: "HV" or "HDV". Default is "HV".

            Returns:
            --------
            result : complex
                The summed correlation value.
            """
            if mode == "HV":
                current_sites, neighbor_sites = get_neighbors_with_distance_HV(self.Lx, self.Ly, distance)
            elif mode == "HDV":
                current_sites, neighbor_sites = get_neighbors_with_distance_HDV(self.Lx, self.Ly, distance)   
            else:
                raise ValueError("Invalid mode. Use 'HV' or 'HDV'.")

            result = 0
            for site1, site2 in zip(current_sites, neighbor_sites):
                # Convert 2D coordinates to 1D index
                idx1 = site1[0] * self.Ly + site1[1]
                idx2 = site2[0] * self.Ly + site2[1]

                # Example: Using Sz operators for correlation
                # Modify as needed for different spin components
                corr = qt.expect(self.Sz_ops[idx1] * self.Sz_ops[idx2], state)
                result += corr  

            return result  

        def calculate_total_magnetization(self, state):
            """
            Calculate the total magnetization of the system.

            Parameters:
            -----------
            state : qutip.Qobj
                The quantum state.

            Returns:
            --------
            total_magnetization : float
                The expectation value of the total S_z magnetization.
            """
            # Sum all Sz operators
            total_Sz = sum(self.Sz_ops)
            
            # Calculate expectation value
            total_magnetization = qt.expect(total_Sz, state)
            
            return total_magnetization

        def total_magnetization_results(self, evaluation_time, results):
            """
            Calculate total magnetization at regular step intervals.

            Parameters:
            -----------
            evaluation_time : int
                The interval of steps at which to evaluate the magnetization.
            results : qutip.sesolve.Result
                The result object from the evolution.

            Returns:
            --------
            magnetizations : np.array
                Array of total magnetization values.
            eval_times : np.array
                Array of corresponding times.
            """
            times = np.array(results.times)
            states = results.states

            # Get indices at evaluation_time intervals
            evaluation_indices = np.arange(0, len(states), evaluation_time).astype(int)

            # Ensure we do not exceed the number of states
            evaluation_indices = evaluation_indices[evaluation_indices < len(states)]

            magnetizations = np.zeros(len(evaluation_indices), dtype=float)
            eval_times = times[evaluation_indices]

            for i, idx in enumerate(evaluation_indices):
                magnetizations[i] = self.calculate_total_magnetization(states[idx])
                    
            return magnetizations, eval_times

        def spatial_correlation_function_results(self, evaluation_time, results, 
                                                distance, mode="HV"):
            """
            Calculate spatial correlations at regular step intervals.

            Parameters:
            -----------
            evaluation_time : int
                The interval of steps at which to evaluate the correlation.
            results : qutip.sesolve.Result
                The result object from the evolution.
            distance : int
                The distance for correlation.
            mode : str, optional
                The mode of neighbor selection: "HV" or "HDV". Default is "HV".

            Returns:
            --------
            correlations : np.array
                Array of correlation values.
            eval_times : np.array
                Array of corresponding times.
            """
            times = np.array(results.times)
            states = results.states

            # Get indices at evaluation_time intervals
            evaluation_indices = np.arange(0, len(states), evaluation_time).astype(int)

            # Ensure we do not exceed the number of states
            evaluation_indices = evaluation_indices[evaluation_indices < len(states)]

            correlations = np.zeros(len(evaluation_indices), dtype=complex)
            eval_times = times[evaluation_indices]

            for i, idx in enumerate(evaluation_indices):
                correlations[i] = self.calculate_spatial_correlation_function(
                    states[idx], distance, mode
                )
                    
            return correlations, eval_times

        def specific_site_correlation_results(self, evaluation_time, results, 
                                                site1, site2):
            """
            Calculate spatial correlations at regular step intervals.

            Parameters:
            -----------
            evaluation_time : int
                The interval of steps at which to evaluate the correlation.
            results : qutip.sesolve.Result
                The result object from the evolution.
            distance : int
                The distance for correlation.
            mode : str, optional
                The mode of neighbor selection: "HV" or "HDV". Default is "HV".

            Returns:
            --------
            correlations : np.array
                Array of correlation values.
            eval_times : np.array
                Array of corresponding times.
            """
            times = np.array(results.times)
            states = results.states

            # Get indices at evaluation_time intervals
            evaluation_indices = np.arange(0, len(states), evaluation_time).astype(int)

            # Ensure we do not exceed the number of states
            evaluation_indices = evaluation_indices[evaluation_indices < len(states)]

            correlations = np.zeros(len(evaluation_indices), dtype=complex)
            eval_times = times[evaluation_indices]

            for i, idx in enumerate(evaluation_indices):
                correlations[i] = self.calculate_specific_site_correlation(
                    states[idx], site1, site2
                )
                    
            return correlations, eval_times

except ImportError:
    # mock or skip if qutip is not available
    class qt:
        pass



