import numpy as np
from tqdm import tqdm

from ..ttn import TreeTensorNetwork
from ..util import fast_exp_action

from copy import deepcopy
"""
Implements a wrapper for different time evolution algorithms.
"""
class TimeEvolutionAlgorithm:
    def __init__(self, state: TreeTensorNetwork, operators, time_step_size, final_time) -> None:
        """
        Parameters
        ----------
        state : TreeTensorNetwork
            A pytreenet/ttn.py TreeTensorNetwork object.
        operators: list of dict
            A list containing dictionaries that contain node identifiers as keys and single-site
            operators as values. Each represents an operator that will be
            evaluated after every time step.
        time_step_size : float
            Size of each time-step.
        final_time : float
            Total time for which TDVP should be run.
        """
        self.state = state

        self._time_step_size = time_step_size
        self.final_time = final_time
        self.num_time_steps = int(np.ceil(final_time / time_step_size))

        if type(operators) == dict:
            # In this case a single operator has been provided
            self.operators = [operators]
        else:
            self.operators = operators

        # Place to hold the results obtained during computation
        # Each row contains the data obtained during the run and the last row
        # contains the time_steps
        if operators != None:
            self._results = np.zeros((len(self.operators) + 1,
                                     self.num_time_steps + 1), dtype=complex)
        else:
            self._results = np.asarray([], dtype=complex)
    
    @property
    def time_step_size(self):
        return self._time_step_size

    @property
    def results(self):
        return self._results
    
    @staticmethod
    def _permutation_svdresult_u_to_fit_node(node):
        """
        After the SVD the two tensors associated to each site have an incorrect
        leg ordering. This function finds the permutation of legs to correct
        this. In more detail:
        The nodes currently have shape
        (virtual-legs, open-legs, contracted leg)
        and the tensor u from the svd has the leg order
        (open-legs, virtual-legs, contracted leg)

        Returns
        -------
        permutation: list of int
            Permutation to find correct legs. Compatible with numpy transpose,
            i.e. the entries are the old leg index and the position in the
            permutation is the new leg index.
        """
        num_open_legs = node.nopen_legs()
        permutation = list(range(num_open_legs, num_open_legs + node.nvirt_legs() -1))
        permutation.extend(range(0,num_open_legs))
        permutation.append(node.nlegs() -1)

        return permutation

    @staticmethod
    def _permutation_svdresult_v_to_fit_node(node):
        """
        After the SVD the two tensors associated to each site have an incorrect
        leg ordering. This function finds the permutation of legs to correct
        this. In more detail:
        The nodes currently have shape
        (virtual-legs, open-legs, contracted leg)
        and the tensor v from the svd has the leg order
        (contracted leg, open-legs, virtual-legs)

        Returns
        -------
        permutation: list of int
            Permutation to find correct legs. Compatible with numpy transpose,
            i.e. the entries are the old leg index and the position in the
            permutation is the new leg index.
        """
        num_open_legs = node.nopen_legs()
        permutation = list(range(num_open_legs + 1, num_open_legs + 1 + node.nvirt_legs() -1))
        permutation.extend(range(1, num_open_legs + 1))
        permutation.append(0)

        return permutation
    
    def evaluate_operators(self):
        """
        Evaluates the expectation value for all operators given in
        self.operators for the current TNS.

        Returns
        -------
        current_results : list
            The expectation values with indeces corresponding to those in
            operators.

        """
        if self.operators != None:
            current_results = np.zeros(len(self.operators), dtype=complex)

            for i, operator_dict in enumerate(self.operators):
                state = deepcopy(self.state)
                exp_val = state.operator_expectation_value(operator_dict)

                current_results[i] = exp_val

            return current_results
        else:
            return []

    def save_results(self, filepath):
        """
        Saves the data in `self.results` into a .npz file.
        
        Parameters
        ----------
        filepath : str
            If results are to be saved in an external file a path can be given
            here.     
        """
        if filepath == None:
            print("No filepath given. Data wasn't saved.")
            return
        
        # We have to lable our data
        kwarg_dict = {}
        for i, operator in enumerate(self.operators):
            kwarg_dict["operator" + str(i)] = operator
            
            kwarg_dict["operator" + str(i) + "results"] = self.results[i]
            
        kwarg_dict["time"] = self.results[-1]
        
        np.savez(filepath, **kwarg_dict)

    def run(self, filepath=None, pgbar=True):
        """
        Runs the time evolution algorithm for the given parameters and saves the computed
        expectation values in `self.results`.
        
        Parameters
        ----------
        filepath : str
            If results are to be saved in an external file a path can be given
            here. Default is None.
        pgbar: bool
            Toggles the progress bar on (True) or off (False). Default is True.
        
        """

        for i in tqdm(range(self.num_time_steps + 1), disable=(not pgbar)):
            if i != 0: # We also measure the initial expectation_values
                self.run_one_time_step()

            if len(self.results) > 0:
                current_results = self.evaluate_operators()
    
                self.results[0:-1,i] = current_results
                # Save current time
                self.results[-1,i] = i*self.time_step_size
                
        if filepath != None:
            self.save_results(filepath)

    def run_one_time_step(self):
        """
        This method needs to be implemented for each individual time evolution
        algorithm.
        """
        raise(NotImplementedError)
    

def time_evolve(psi, H, dt, forward=True):
    """
    psi(t +/- dt) = e^( -/+ i*H*dt ) * psi(t)
    -iHdt: forward = True
    +iHdt: forward = False

    Parameters
    ----------

    psi : ndarray with shape (n,)
        State to be evolved.
    H : ndarray with shape (n, n)
        Hamiltonian sácting on the state.
    dt : float
        Timestep size.
    forward : bool
        Determines time direction of the time evolution.

    Returns
    -------
    psi : ndarray (n,)
        Time-evolved state.
    """
    sign = -2 * forward + 1  # forward=True -> -1; forward=False -> +1
    exponent = sign * 1.0j * H * dt
    return np.reshape(fast_exp_action(exponent, psi.flatten(), mode="fastest"), newshape=psi.shape)