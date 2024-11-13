import numpy as np
import pytreenet as ptn
import copy
import tenpy
from tenpy.models.lattice import Square
from tenpy.networks.site import BosonSite
from tenpy.networks.mps import MPS
from tenpy.algorithms.tdvp import SingleSiteTDVPEngine, TwoSiteTDVPEngine
from tenpy.models.model import CouplingMPOModel
from tenpy.linalg.np_conserved import Array
from .util import get_neighbors_with_distance_HV, get_neighbors_with_distance_HDV


# Define the BoseHubbard2D_Tenpy class
class BoseHubbard2D_Tenpy(CouplingMPOModel):
    def __init__(self, Lx, Ly, n_bosons, t, U, mu):
        self.Lx = Lx
        self.Ly = Ly
        self.n_bosons = n_bosons + 1
        self.t = t
        self.U = U
        self.mu = mu
        self.bc_MPS = 'finite'  # Set the MPS boundary conditions

        # Define the lattice and sites
        site = BosonSite(Nmax=n_bosons - 1, conserve=None)
        lat = Square(Lx=Lx, Ly=Ly, site=site)
        self.lat = lat
        self.sites = lat.mps_sites()

        # Prepare model parameters without 'bc_MPS' and 'verbose'
        model_params = {
            'lattice': lat,
            't': t,
            'U': U,
            'mu': mu,
        }

        # Initialize the base class
        CouplingMPOModel.__init__(self, model_params)

    def pos_to_idx(self, site):
        """
        Convert 2D coordinates to 1D index.
        
        Parameters:
        ----------
        site : tuple
            A tuple (x, y) representing the coordinates of the site.
            
        Returns:
        -------
        int
            The 1D index corresponding to the (x, y) coordinates.
        """
        x, y = site
        return x * self.Lx + y

    def init_terms(self, model_params):
        """
        Initialize the terms of the Hamiltonian.
        """
        U = model_params.get('U', self.U)
        mu = model_params.get('mu', self.mu)
        t = model_params.get('t', self.t)

        # On-site interaction and chemical potential
        u = 0  # Only one site type
        self.add_onsite(0.5 * U, u, 'N N')
        self.add_onsite(-mu, u, 'N')

        # Hopping terms
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Bd', u2, 'B', dx, plus_hc=True)

    def uniform_product_state(self, local_state_label):
        """
        Create a uniform product state MPS.
        """
        psi = [local_state_label for _ in range(self.lat.N_sites)]
        return MPS.from_product_state(self.sites, psi, bc=self.bc_MPS)

    def alternating_product_state_custom(self, black_state, white_state, pattern="checkerboard"):
        """
        Create an alternating product state MPS with custom pattern and arbitrary state vectors.
        
        Parameters:
        ----------
        black_state : array-like
            The state vector for black sites. Must be of length `site.dim` and normalized.
        white_state : array-like
            The state vector for white sites. Must be of length `site.dim` and normalized.
        pattern : str
            The pattern to apply. Currently supports "checkerboard" or "half_random".
        
        Returns:
        -------
        MPS
            The initialized MPS with the specified alternating product state.
        """
        # Validate and normalize input state vectors
        black_state = np.asarray(black_state, dtype=np.complex128)
        white_state = np.asarray(white_state, dtype=np.complex128)
        
        if black_state.shape[0] != self.sites[0].dim or white_state.shape[0] != self.sites[0].dim:
            raise ValueError(f"State vectors must match site dimension {self.sites[0].dim}.")
        
        black_state /= np.linalg.norm(black_state)
        white_state /= np.linalg.norm(white_state)

        # Determine the sites assigned to black and white states based on the pattern
        if pattern == "checkerboard":
            black_sites, white_sites = self.get_checkerboard_pattern()
        elif pattern == "half_random":
            black_sites, white_sites = self.get_random_half_sites()


        # Create tensors for each site and a list of identity singular values
        psi = []
        for idx in range(self.lat.N_sites):
            x, y = idx % self.Lx, idx // self.Lx
            state_vector = black_state if (x, y) in black_sites else white_state
            psi.append(state_vector)

        # return the MPS with specified boundary conditions
        return MPS.from_product_state(self.sites, psi, bc=self.bc_MPS, dtype=np.complex128)

    def get_checkerboard_pattern(self):
        """
        Generate a checkerboard pattern of black and white sites.
        """
        black_sites = []
        white_sites = []
        for x in range(self.Lx):
            for y in range(self.Ly):
                if (x + y) % 2 == 0:
                    black_sites.append((x, y))
                else:
                    white_sites.append((x, y))
        return black_sites, white_sites

    def evolve_system_two_site(self, psi0, end_time, dt=0.1, chi_max=100, svd_min=1e-10, lanczos_params=None):
        """
        Evolve the system using Two-Site TDVP and store intermediate states.

        Parameters:
        ----------
        psi0 : MPS
            Initial MPS state.
        end_time : float
            Total time to evolve.
        dt : float
            Time step.
        chi_max : int
            Maximum bond dimension.
        svd_min : float
            Minimum singular value to keep.
        lanczos_params : dict
            Parameters for the Lanczos algorithm.

        Returns:
            - state_history: List of MPS states at each time step
        """
        N_steps = int(end_time / dt)
        tdvp_params = {
            'dt': dt,
            'start_time': 0,
            'N_steps': 1,  # We'll do steps manually
            'trunc_params': {'chi_max': chi_max, 'svd_min': svd_min},
            'lanczos_params': lanczos_params
        }

        # Initialize lists to store states and times
        state_history = [psi0.copy()]  # Store initial state
        times = [0.0]
        
        # Create initial engine
        current_psi = psi0.copy()
        
        # Evolve step by step
        for step in range(N_steps):
            # Create new engine for each step
            eng = TwoSiteTDVPEngine(current_psi, self, tdvp_params)
            eng.run()
            
            # Store the state after this step
            current_psi = eng.psi.copy()
            state_history.append(current_psi)
            times.append((step + 1) * dt)

        return state_history


    def calculate_occupation(self, psi):
        """
        Calculate occupation number for each site.
        """
        occupations = np.zeros((self.Lx, self.Ly))
        for idx in range(self.lat.N_sites):
            x = idx % self.Lx
            y = idx // self.Lx
            n_op = self.sites[idx].N
            occupations[x, y] = psi.expectation_value(n_op, [idx])[0]
        return occupations

    def calculate_spatial_correlation_function(self, psi, distance, mode="HV", Normalize=False):
        """
        Calculate first-order correlation function.
        """
        # Obtain pairs of sites at the specified distance
        if mode == "HV":
            current_sites, neighbor_sites = get_neighbors_with_distance_HV(self.Lx, self.Ly, distance)
        elif mode == "HDV":
            current_sites, neighbor_sites = get_neighbors_with_distance_HDV(self.Lx, self.Ly, distance)
        else:
            raise ValueError("Invalid mode. Use 'HV' or 'HDV'.")

        # Initialize variables for summing correlation values
        result = 0
        count = 0

        for site1, site2 in zip(current_sites, neighbor_sites):
            idx1 = self.pos_to_idx(site1)
            idx2 = self.pos_to_idx(site2)

            # Calculate correlation using operator names as strings
            corr = psi.expectation_value_term([("Bd", idx1), ("B", idx2)])

            if Normalize:
                n1 = psi.expectation_value(self.sites[idx1].N, [idx1])[0]
                n2 = psi.expectation_value(self.sites[idx2].N, [idx2])[0]
                if n1 > 0 and n2 > 0:
                    corr /= np.sqrt(n1 * n2)

            result += abs(corr)
            count += 1

        return result / count if count > 0 else 0

    def calculate_specific_site_correlation(self, psi, site1, site2, Normalize=False):
        idx1 = self.pos_to_idx(site1)
        idx2 = self.pos_to_idx(site2)
        # Calculate correlation using operator names as strings
        corr = psi.expectation_value_term([("Bd", idx1), ("B", idx2)])

        if Normalize:
            n1 = psi.expectation_value(self.sites[idx1].N, [idx1])[0]
            n2 = psi.expectation_value(self.sites[idx2].N, [idx2])[0]
            if n1 > 0 and n2 > 0:
                corr /= np.sqrt(n1 * n2)

        return abs(corr)

    def calculate_density_density_correlation(self, psi, distance, mode="HV", Normalize=False):
        """
        Calculate density-density correlation function.
        """
        if mode == "HV":
            current_sites, neighbor_sites = get_neighbors_with_distance_HV(self.Lx, self.Ly, distance)
        elif mode == "HDV":
            current_sites, neighbor_sites = get_neighbors_with_distance_HDV(self.Lx, self.Ly, distance)
        else:
            raise ValueError("Invalid mode. Use 'HV' or 'HDV'.")

        result = 0
        count = 0
        for site1, site2 in zip(current_sites, neighbor_sites):
            idx1 = self.pos_to_idx(site1)  # Corrected
            idx2 = self.pos_to_idx(site2)  # Corrected
            n1n2 = psi.expectation_value_term([("N", idx1), ("N", idx2)])
            n1 = psi.expectation_value(self.sites[idx1].N, [idx1])[0]
            n2 = psi.expectation_value(self.sites[idx1].N, [idx2])[0]
            if Normalize:
                if n1 * n2 > 0:
                    n1n2 -= n1 * n2
                    n1n2 /= np.sqrt(n1 * n2)
            else:
                n1n2 -= n1 * n2
            result += n1n2
            count += 1
        return result / count if count > 0 else 0

    def specific_site_correlation_results(self, psi_list, evaluation_time, site1, site2, Normalize, dt):
        """
        Calculate spatial correlations at regular intervals for each state in psi_list.
        """
        correlations = []
        times = []
        for step, psi in enumerate(psi_list):
            if step % evaluation_time == 0:
                correlations.append(self.calculate_specific_site_correlation(psi, site1, site2, Normalize))
                times.append(step * dt)
        return np.array(correlations), np.array(times)

    def occupation_results(self, psi_list, evaluation_time, dt):
        """
        Calculate occupation numbers at regular intervals for each state in psi_list.
        """
        occupations = []
        times = []
        for step, psi in enumerate(psi_list):
            if step % evaluation_time == 0:
                occupations.append(self.calculate_occupation(psi))
                times.append(step * dt)
        return np.array(occupations), np.array(times)

    def spatial_correlation_function_results(self, psi_list, evaluation_time, distance, mode, Normalize, dt):
        """
        Calculate spatial correlations at regular intervals for each state in psi_list.
        """
        correlations = []
        times = []
        for step, psi in enumerate(psi_list):
            if step % evaluation_time == 0:
                correlations.append(self.calculate_spatial_correlation_function(psi, distance, mode, Normalize))
                times.append(step * dt)
        return np.array(correlations), np.array(times)

    def density_density_correlation_results(self, psi_list, evaluation_time, distance, mode, Normalize, dt):
        """
        Calculate density-density correlations at regular intervals for each state in psi_list.
        """
        correlations = []
        times = []
        for step, psi in enumerate(psi_list):
            if step % evaluation_time == 0:
                correlations.append(self.calculate_density_density_correlation(psi, distance, mode, Normalize))
                times.append(step * dt)
        return np.array(correlations), np.array(times)
