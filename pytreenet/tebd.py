import numpy as np

from scipy.linalg import expm

from node_contraction import contract_tensors_of_nodes

class Hamiltonian:
    """
    Represents the Hamiltonian on a TTN.
    The entries should represent the terms of the Hamiltonian and be a
    dictionary. The key is an identifier which could be matched to a TensorNode
    in the TTN and the value is an operator that is to be applied to that
    node/site.
    """
    
    def __init__(self, terms=None):
        if terms == None:
            self.terms = []
        else:
            self.terms = terms
        
    def add_term(self, term):
        if not (type(term) == dict):
            term = dict(term)
        
        self.terms.append(term)
        
    def add_multiple_terms(self, terms):
        if type(terms) == list:
            for term in terms:
                self.add_term(term)
        else:
            raise TypeError("'terms' has to be a list of dictionaries")
        
    def __add__(self, other_hamiltonian):
        
        total_terms = []
        total_terms.extend(self.terms)
        total_terms.extend(other_hamiltonian.term)
        
        new_hamiltonian = Hamiltonian(terms=total_terms)
        return new_hamiltonian
    
    def __repr__(self):
        return str(self.terms)

class TEBD:
    """
    Runs the TEBD algorithm on a TTN
    """
    
    def __init__(self, state, hamiltonian, time_step_size, final_time,
                 custom_splitting=None):
        """
        The state is a TreeTensorNetwork representing an initial state which is
        to be time-evolved under the Hamiltonian hamiltonian until final_time,
        where each time step has size time_step_size.
        
        If a custom_splitting is provided it should be a lists of integers.
        The integers should give the order in which the Hamiltonian terms are
        to be applied, i.e. the order of the Trotter-Suzuki splitting.
        """
        
        self.state = state
        self._hamiltonian = hamiltonian
        self._time_step_size = time_step_size
        self.final_time = final_time
        
        if custom_splitting==None:
            raise NotImplementedError()
        else:
            self.splitting = custom_splitting
        
        self.exponents = self.exponentiate_terms(time_step_size=self.time_step_size)
        
    def exponentiate_terms(self, time_step_size=None):
        """
        If time_step_size or hamiltonian is to be changed, this function has
        to be rerun.
        """
        
        if not (time_step_size == None):
            self.time_step_size = time_step_size
            
        for interaction_operator in self.hamiltonian:
            total_operator = 1
            site_ids = []
            
            for site in interaction_operator:
                total_operator = np.tensordot(total_operator,
                                              interaction_operator[site],
                                              axes=0)

                site_ids.append(site)
            
            exponentiated_operator = expm((-1j*self.time_step_size) * total_operator)
                    
            self.exponents.append({"operator":exponentiated_operator,
                                   "site_ids": site_ids})
            
    @property
    def time_step_size(self):
        return self._time_step_size
    
    @time_step_size.setter
    def time_step_size(self, new_time_step_size):
        self._time_step_size = new_time_step_size
        self.exponents = self.exponentiate_terms()
    
    @property
    def hamiltonian(self):
        return self._hamiltonian

    @hamiltonian.setter
    def time_step_size(self, new_hamiltonian):
        self._hamiltonian = new_hamiltonian
        self.exponents = self.exponentiate_terms()
    
    
    
    def run_one_time_step(self):
        """
        Running one time_step on the TNS according to the exponentials and the
        splitting provided in the object.
        """
        
        for index in self.splitting:
            two_site_exponent = self.exponents[index]
            
            operator = two_site_exponent["operator"]
            
            identifiers = two_site_exponent["site_ids"]
            
            # TODO: Generalise to more than two sites
            node1 = self.state.nodes[identifiers[0]]
            node2 = self.state.nodes[identifiers[1]]
            
            two_site_tensor = contract_tensors_of_nodes(node1, node2)
            
            
            
            
    
    
    def run(self):
        pass