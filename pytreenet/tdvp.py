"""
Implements the time-dependent variational principle TDVP for tree tensor
networks.

Reference:
    D. Bauernfeind, M. Aichhorn; "Time Dependent Variational Principle for Tree
    Tensor Networks", DOI: 10.21468/SciPostPhys.8.2.024
"""

def tdvp(state, hamiltonian, time_step_size, final_time, mode="1site"):
    """

    Parameters
    ----------
    state : TreeTensorState
        The TTN representing the intial state on which the TDVP is to be 
        performed
    hamiltonian: TreeTensorOperator
        A TTN representing the model Hamiltonian. Each tensor in
        tree_tensor_network should be associated to one node in hamiltonian
    time_step_size : float
        Size of each time-step in the trotterisation.
    final_time : float
        Tital time for which TDVP should be run.
    mode : str, optional
        Decides which version of the TDVP is run. The options are 1site and
        2site. The default is "1site".

    Returns
    -------
    None.

    """
    assert len(state.nodes) == len(hamiltonian.nodes)
    raise NotImplementedError
    # TODO: Implement