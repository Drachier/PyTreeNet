import numpy as np

from pytreenet.ttn import TreeTensorNetwork

from special_operators import create_bosonic_operators

class ttno_HOPS(TreeTensorNetwork):
    """
    Creates the TTNO that represents the effective Hamiltonian for the HOPS in
    TTN-form.
    """

    def __init__(self):
        """
        Initialises a new TreeTensorNetwork
        """
        TreeTensorNetwork.__init__(self)
        
    def set_up_ttno(self, physical_MPO, system_operators, stochastic_processes,
                 coupling_strength, oscillator_frequency, num_aux_sites,
                 aux_dimension=3):
        """
        Builds a TreeTensorNetwork in HOPS-form. 
        
        Parameters
        ----------
        physical_MPO: list of ndarray
            Contains the physical Hamiltonian in MPO-form. Every entry is a
            degree 4 tensor with the convention:

                   |2
                 __|__
             3  |     |  4
             ---|     |---
                |_____|
                   |
                   |1
            
            The system size is inferred from the length of this list.
                   

        system_operators: list of ndarray
            Contains the system operators, that couple to the external harmonic
            oscillators.
       
        stochastic_processes: list of float
            Contains the stochastic process for every site for a given time t.
        
        coupling_strength: list of float
            The coupling strength of the auxilliary oscillator to the system. 
            #Todo: Make special cases for something other than different for each oscillator (2D list).

        oscillator_frequency: list of float
            The chracteristic frequency of each auxilliary oscillator. 
            #Todo: Make special cases for something other than different for each oscillator (2D list).
        
        num_aux_sites: int
            The number of auxilliary sites/oscillators per physical sites
        
        aux_dimensions: int
            Max. dimension of each in principle infintely dimensional bosonic
            auxilliary site/oscillator
        
        """
        assert len(system_operators) == len(stochastic_processes)
        
        #Adding the local term to MPO
        modified_MPO = []
        for mpo_tensor, j in enumerate(physical_MPO):
            modified_system_oprator = stochastic_processes[j] * system_operators[j]
            padded_operator = self._operator_in_lower_left_corner(modified_system_oprator, mpo_tensor)
            new_mpo_tensor = mpo_tensor + padded_operator
            modified_MPO.append(new_mpo_tensor)
            
        #Adding the additional leg to the physical MPO and putting the required system_ops into it
        identity = np.eye(system_operators[j].shape[0])
        physical_part_of_TTNO = []
        for mpo_tensor, j in enumerate(modified_MPO):
            tensor_with_one_more_leg = [mpo_tensor]
            tensor_with_one_more_leg.appen(self._operator_in_lower_left_corner(system_operators[j], mpo_tensor))
            tensor_with_one_more_leg.appen(self._operator_in_lower_left_corner(np.conj(system_operators[j].T), mpo_tensor))
            tensor_with_one_more_leg.appen(self._operator_in_lower_left_corner(identity, mpo_tensor))
            tensor_with_one_more_leg = np.asarray(tensor_with_one_more_leg)
            
            # Currently the leg pointing towards the auxiliary sites is the first one, we want it to be the last
            tensor_with_one_more_leg.transpose(1,2,3,4,0)
            physical_part_of_TTNO.append(tensor_with_one_more_leg)
        
        #Building the auxilliary part of the MPO, which is always the same up to scalars
        auxillliary_TTNO = self._build_aux_TTNO_part(coupling_strength, oscillator_frequency, num_aux_sites, aux_dimension)
        
    def _operator_in_lower_left_corner(self, operator, reference_mpo):
        '''
        Creates a TTNO-tensor that has the operator in the lower left corner
        and the total shape like reference_mpo, where all but the lower left
        corner is zero.
        Parameters
        ----------
        operator : ndarray
        reference_array : ndarray

        Returns
        -------
        padded_operator : ndarray
        
        '''
        padded_operator = np.zeros_like(reference_mpo)
        padded_operator[:,:,-1,0] = operator
        return padded_operator
    
    def _build_aux_TTNO_part(coupling_strength, oscillator_frequency, num_aux_sites, aux_dimension):
        
        creation_op, annihilation_op, number_op = create_bosonic_operators(aux_dimension)        
        number_times_creation_op = number_op @ creation_op #creation_op only needed here
        bosonic_identity = np.eye(aux_dimension)
        
        #Same for every auxilliary site
        constant_part = np.zeros((aux_dimension, aux_dimension, 4, 4))
        for i in range(0,3):
            constant_part[:,:,i,i] = bosonic_identity
        constant_part[:,:,2,0]
        
        ##CONTINUE BUILDING THE AUX_TTNO HERE!!1
        
        