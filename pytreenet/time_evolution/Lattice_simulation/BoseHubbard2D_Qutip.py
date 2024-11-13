import qutip as qt
import random
import numpy as np
from .util import get_neighbors_with_distance_HV, get_neighbors_with_distance_HDV


def create_bose_hubbard_2d(Lx, Ly, n_bosons, J, U, mu):
    """
    Create 2D Bose-Hubbard Hamiltonian with chemical potential
    
    Parameters:
    -----------
    Lx, Ly: int
        Lattice dimensions
    n_bosons: int
        Maximum number of bosons per site
    J: float
        Hopping strength
    U: float
        On-site interaction strength
    mu: float
        Chemical potential
    """
    n_sites = Lx * Ly
    
    # Create destruction operators for each site
    a_ops = []
    for i in range(n_sites):
        op_list = [qt.qeye(n_bosons) for j in range(n_sites)]
        op_list[i] = qt.destroy(n_bosons)
        a_ops.append(qt.tensor(op_list))
    
    # Create Hamiltonian
    H = 0
    
    # Helper function to get site index from 2D coordinates
    def get_index(x, y):
        return x * Ly + y
    
    # Hopping terms in x-direction
    for x in range(Lx):
        for y in range(Ly):
            # x-direction hopping
            if x < Lx - 1:
                i = get_index(x, y)
                j = get_index(x + 1, y)
                H += -J * (a_ops[i].dag() * a_ops[j] + a_ops[j].dag() * a_ops[i])
            
            # y-direction hopping
            if y < Ly - 1:
                i = get_index(x, y)
                j = get_index(x, y + 1)
                H += -J * (a_ops[i].dag() * a_ops[j] + a_ops[j].dag() * a_ops[i])
    
    # On-site interaction and chemical potential terms
    for i in range(n_sites):
        n_op = a_ops[i].dag() * a_ops[i]
        H += (U/2) * n_op * (n_op - 1)  # Interaction
        H += -mu * n_op                  # Chemical potential
    
    return H, a_ops

class BoseHubbard2D_Qutip:
    def __init__(self, Lx, Ly, n_bosons, J, U, mu):
        self.Lx = Lx
        self.Ly = Ly
        self.n_bosons = n_bosons + 1 
        self.J = J
        self.U = U
        self.mu = mu
        
        self.H, self.a_ops = create_bose_hubbard_2d(Lx, Ly, n_bosons, J, U, mu)
        self.n_ops = [a.dag() * a for a in self.a_ops]

    def uniform_product_state(self, local_state) :
        # Create the single-site state in QuTiP format
        single_site = qt.Qobj(local_state)

        # Create the product state
        state = single_site
        for _ in range(self.Lx * self.Ly - 1):
            state = qt.tensor(state, single_site)
        return state

    def get_checkerboard_pattern(self):
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
        total_sites = self.Lx * self.Ly

        # Generate a list of all sites as tuples instead of strings
        all_sites = [(x, y) for x in range(self.Lx) for y in range(self.Ly)]

        # Shuffle and split into two halves
        random.shuffle(all_sites)
        half = total_sites // 2

        black_sites = all_sites[:half]
        white_sites = all_sites[half:]

        return black_sites, white_sites

    def alternating_product_state(self, black_state, white_state, pattern):
        # Create single-site states in QuTiP format
        black_site_qobj = qt.Qobj(black_state)
        white_site_qobj = qt.Qobj(white_state)
        
        # Initialize an empty list for product state components
        product_state_components = []
        
        # Choose the pattern and get black/white site lists
        if pattern == "checkerboard":
            black_sites, white_sites = self.get_checkerboard_pattern()
        elif pattern == "half_random":
            black_sites, white_sites = self.get_random_half_sites()
        else:
            raise ValueError("Invalid pattern. Use 'checkerboard' or 'half_random'.")

        # Initialize the state list with white states as placeholders
        for x in range(self.Lx):
            for y in range(self.Ly):
                product_state_components.append(white_site_qobj)
        
        # Assign black and white states to the specific indices
        for site in black_sites:
            idx = site[0] * self.Ly + site[1]
            product_state_components[idx] = black_site_qobj

        # Tensor product to create the full product state
        product_state = product_state_components[0]
        for state in product_state_components[1:]:
            product_state = qt.tensor(product_state, state)
        
        return product_state

    def evolve_system(self, psi0, end_time, dt=0.1):
        """[Previous implementation remains the same]"""
        times = np.linspace(0, end_time, int(end_time/dt) + 1)
        self.result = qt.sesolve(self.H, psi0, times)
        return self.result

    def calculate_occupation(self, state):
        """
        Calculate occupation number for each site
        
        Parameters:
        -----------
        state: qutip.Qobj
            Quantum state at a specific time
            
        Returns:
        --------
        numpy.ndarray
            2D array of shape (Lx, Ly) containing occupation numbers
        """
        occupations = np.zeros((self.Lx, self.Ly))
        
        for i in range(self.Lx):
            for j in range(self.Ly):
                idx = i * self.Ly + j
                occupations[i, j] = qt.expect(self.n_ops[idx], state)

        return occupations

    def calculate_spatial_correlation_function(self, state, distance, mode = "HV" , Normalize = False):
        """
        Calculate first-order correlation function: g₁(r1,r2) = ⟨a†(r1)a(r2)⟩/√(⟨n(r1)⟩⟨n(r2)⟩)
        """
        if mode == "HV":
            current_sites, neighbor_sites = get_neighbors_with_distance_HV(self.Lx, self.Ly, distance)
        elif mode == "HDV":
            current_sites, neighbor_sites = get_neighbors_with_distance_HDV(self.Lx, self.Ly, distance)   

        result = 0
        if Normalize:
            # Scales with density (Larger values in high-density regions)
            # gives actual coherence magnitude
            # Studying transport properties
            for site1, site2 in zip(current_sites, neighbor_sites):
                # Convert 2D coordinates to 1D index
                idx1 = site1[0] * self.Ly + site1[1]
                idx2 = site2[0] * self.Ly + site2[1]
            
                n1 = qt.expect(self.n_ops[idx1], state)
                n2 = qt.expect(self.n_ops[idx2], state)
                corr = abs(qt.expect(self.a_ops[idx1].dag() * self.a_ops[idx2], state))
                if n1 > 0 and n2 > 0:
                   corr /= np.sqrt(n1 * n2)
                result += abs(corr)
        else:
            # Independent of density (Same value for same "degree of coherence")
            # Studying phase transitions / universal behavior
            for site1, site2 in zip(current_sites, neighbor_sites):
                # Convert 2D coordinates to 1D index
                idx1 = site1[0] * self.Ly + site1[1]
                idx2 = site2[0] * self.Ly + site2[1]

                corr = abs(qt.expect(self.a_ops[idx1].dag() * self.a_ops[idx2], state))
                result += abs(corr)       

        return result / len(current_sites)
    
    def calculate_density_density_correlation(self, state, distance, mode = "HV" , Normalize = False):
        """
        Calculate density-density correlation: g₂(r1,r2) = ⟨n(r1)n(r2)⟩/(⟨n(r1)⟩⟨n(r2)⟩)
        """

        if mode == "HV":
            current_sites, neighbor_sites = get_neighbors_with_distance_HV(self.Lx, self.Ly, distance)
        elif mode == "HDV":
            current_sites, neighbor_sites = get_neighbors_with_distance_HDV(self.Lx, self.Ly, distance)   

        result = 0
        # Mott insulator : both negative
        # Superfluid : both positive
        if Normalize:
            # Measures genuine correlations beyond mean-field
            # Goes to zero for uncorrelated states
            # Can be positive (attraction) or negative (repulsion)
            for site1, site2 in zip(current_sites, neighbor_sites):
                # Convert 2D coordinates to 1D index
                idx1 = site1[0] * self.Ly + site1[1]
                idx2 = site2[0] * self.Ly + site2[1]
            
                n1 = qt.expect(self.n_ops[idx1], state)
                n2 = qt.expect(self.n_ops[idx2], state)
                n1n2 = qt.expect(self.n_ops[idx1] * self.n_ops[idx2], state)
                if n1 * n2 == 0:
                    result += (n1n2 - (n1 * n2)) 
                else:
                    result += (n1n2 - (n1 * n2)) / (n1 * n2)
        else:
            # g^(2) = 0: uncorrelated (Poissonian)
            # g^(2) > 0: super-Poissonian (bunching)
            # g^(2) < 0: sub-Poissonian (antibunching)
            for site1, site2 in zip(current_sites, neighbor_sites):
                # Convert 2D coordinates to 1D index
                idx1 = site1[0] * self.Ly + site1[1]
                idx2 = site2[0] * self.Ly + site2[1]

                n1 = qt.expect(self.n_ops[idx1], state)
                n2 = qt.expect(self.n_ops[idx2], state)
                n1n2 = qt.expect(self.n_ops[idx1] * self.n_ops[idx2], state)
                result += n1n2 - (n1 * n2)       

        return result / len(current_sites)


    def calculate_specific_site_correlation(self, state, site1, site2, normalize=False):
        """
        Calculate the correlation between two specific sites.
        
        Parameters:
        -----------
        state : qutip.Qobj
            The quantum state to calculate the correlation on.
        site1 : tuple of int
            Coordinates (x1, y1) of the first site.
        site2 : tuple of int
            Coordinates (x2, y2) of the second site.
        normalize : bool
            If True, normalize the correlation by the occupations.

        Returns:
        --------
        float
            The correlation between the two sites.
        """
        # Convert 2D coordinates to 1D indices
        idx1 = site1[0] * self.Ly + site1[1]
        idx2 = site2[0] * self.Ly + site2[1]

        # Calculate correlation ⟨a†(site1) * a(site2)⟩
        correlation = qt.expect(self.a_ops[idx1].dag() * self.a_ops[idx2], state)
        
        # Normalize if requested
        if normalize:
            n1 = qt.expect(self.n_ops[idx1], state)
            n2 = qt.expect(self.n_ops[idx2], state)
            if n1 * n2 > 0:  # Avoid division by zero
                correlation /= np.sqrt(n1 * n2)

        return abs(correlation)
    
    def calculate_specific_density_density_correlation(self, state, site1, site2, normalize=False):
        """
        Calculate the density-density correlation between two specific sites.

        Parameters:
        -----------
        state : qutip.Qobj
            The quantum state to calculate the correlation on.
        site1 : tuple of int
            Coordinates (x1, y1) of the first site.
        site2 : tuple of int
            Coordinates (x2, y2) of the second site.
        normalize : bool
            If True, normalize the correlation by the occupations.

        Returns:
        --------
        float
            The density-density correlation between the two sites.
        """
        # Convert 2D coordinates to 1D indices
        idx1 = site1[0] * self.Ly + site1[1]
        idx2 = site2[0] * self.Ly + site2[1]

        # Calculate density-density correlation ⟨n(site1) * n(site2)⟩
        n1n2 = qt.expect(self.n_ops[idx1] * self.n_ops[idx2], state)
        
        # Normalize if requested
        if normalize:
            n1 = qt.expect(self.n_ops[idx1], state)
            n2 = qt.expect(self.n_ops[idx2], state)
            if n1 * n2 > 0:  # Avoid division by zero
                n1n2 /= np.sqrt(n1 * n2)

        return abs(n1n2)

    def specific_density_density_correlation_results(self, evaluation_time, results, site1, site2, normalize=False):
        """
        Calculate specific density-density correlation at regular time steps.

        Parameters:
        -----------
        evaluation_time : int
            Interval at which to calculate results (every nth time step).
        results : qutip.Result
            The result object from the time evolution.
        site1, site2 : tuple of int
            Coordinates of the two sites to calculate the density-density correlation between.
        normalize : bool
            Whether to normalize the correlation by site occupations.

        Returns:
        --------
        tuple : (numpy.ndarray, numpy.ndarray)
            correlations : array of density-density correlations over time.
            eval_times : array of times corresponding to each correlation.
        """
        times = np.array(results.times)
        states = results.states

        # Get indices at evaluation_time intervals
        evaluation_indices = np.arange(0, len(states), evaluation_time).astype(int)
        evaluation_indices = evaluation_indices[evaluation_indices < len(states)]  # Ensure we do not exceed bounds

        correlations = np.zeros(len(evaluation_indices), dtype=complex)
        eval_times = times[evaluation_indices]

        for i, idx in enumerate(evaluation_indices):
            correlations[i] = self.calculate_specific_density_density_correlation(
                state=states[idx], site1=site1, site2=site2, normalize=normalize
            )

        return correlations, eval_times

    def specific_site_correlation_results(self, evaluation_time, results, site1, site2, normalize=False):
        """
        Calculate specific site correlation at regular time steps.

        Parameters:
        -----------
        evaluation_time : int
            Interval at which to calculate results (every nth time step).
        results : qutip.Result
            The result object from the time evolution.
        site1, site2 : tuple of int
            Coordinates of the two sites to calculate the correlation between.
        normalize : bool
            Whether to normalize the correlation by site occupations.

        Returns:
        --------
        tuple : (numpy.ndarray, numpy.ndarray)
            correlations : array of correlations over time.
            eval_times : array of times corresponding to each correlation.
        """
        times = np.array(results.times)
        states = results.states

        # Get indices at evaluation_time intervals
        evaluation_indices = np.arange(0, len(states), evaluation_time).astype(int)
        evaluation_indices = evaluation_indices[evaluation_indices < len(states)]  # Ensure we do not exceed bounds

        correlations = np.zeros(len(evaluation_indices), dtype=complex)
        eval_times = times[evaluation_indices]

        for i, idx in enumerate(evaluation_indices):
            correlations[i] = self.calculate_specific_site_correlation(
                state=states[idx], site1=site1, site2=site2, normalize=normalize
            )

        return correlations, eval_times
    
    def occupation_results(self, evaluation_time, results):
        """
        Calculate occupation numbers at regular step intervals

        Parameters:
        -----------
        evaluation_time: int
            Calculate results every nth step
        results: qutip.Result
            Results from time evolution
            
        Returns:
        --------
        tuple: (numpy.ndarray, numpy.ndarray)
            - occupations: array of shape (n_evaluations, Lx, Ly)
            - eval_times: array of corresponding times
        """
        times = np.array(results.times)
        states = results.states

        # Get indices at evaluation_time intervals (0, evaluation_time, 2*evaluation_time, ...)
        evaluation_indices = np.arange(0, len(states), evaluation_time).astype(int)

        # Ensure we do not exceed the number of states
        evaluation_indices = evaluation_indices[evaluation_indices < len(states)]

        occupations = np.zeros((len(evaluation_indices), self.Lx, self.Ly))
        eval_times = times[evaluation_indices]

        for i, idx in enumerate(evaluation_indices):
            occupations[i] = self.calculate_occupation(states[idx])
                
        return occupations, eval_times

    def spatial_correlation_function_results(self, evaluation_time, results, 
                                            distance, mode="HV", Normalize=False):
        """
        Calculate spatial correlations at regular step intervals
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
                states[idx], distance, mode, Normalize)
                
        return correlations, eval_times

    def density_density_correlation_results(self, evaluation_time, results, 
                                        distance, mode="HV", Normalize=False):
        """
        Calculate density-density correlations at regular step intervals
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
            correlations[i] = self.calculate_density_density_correlation(
                states[idx], distance, mode, Normalize)
                
        return correlations, eval_times
    
