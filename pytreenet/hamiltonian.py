from numpy.random import default_rng

class Hamiltonian(object):
    """
    Represents the Hamiltonian on a TTN.
    The entries of the main list should represent the terms of the Hamiltonian 
    and be a dictionary. The key is an identifier which could be matched to a
    TensorNode in the TTN and the value is an operator that is to be applied to
    that node/site.
    """
    
    def __init__(self, terms=None):
        """
        Parameters
        ----------
        terms : list of dictionaries, optional
            A list of dictionaries containing the terms of the Hamiltonian
            The default is None.

        """
        if terms == None:
            self.terms = []
        else:
            self.terms = terms
        
    def add_term(self, term):
        if not (type(term) == dict):
            try:
                term = dict(term)
            except TypeError:
                raise TypeError(f"{term} cannot be converted to a dictionary.")
                
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
        total_terms.extend(other_hamiltonian.terms)
        
        new_hamiltonian = Hamiltonian(terms=total_terms)
        return new_hamiltonian
    
    def __repr__(self):
        return str(self.terms)
    
    def __eq__(self, other_hamiltonian):
        """
        Two Hamiltonians are equal, if all of their terms are equal.
        """
        return self.terms == other_hamiltonian.terms
    
def random_terms(num_of_terms, possible_operators, sites, min_strength = -1, max_strength = 1,
                 min_num_sites=2,  max_num_sites=2):
    """
    Creates random interaction terms.

    Parameters
    ----------
    num_of_terms : int
        The number of random terms to be generated.
    possible_operators : list of arrays
        A list of all possible single site operators. We assume all sites have
        the same physical dimension.
    sites : list of str
        A list containing the possible identifiers of site nodes.
    min_strength : float, optional
        Minimum strength an interaction term can have. The strength is
        multiplied to the first operator of the term. The default is -1.
    max_strength : float, optional
        Minimum strength an interaction term can have. The strength is
        multiplied to the first operator of the term. The default is 1.
    min_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction 
        term. The default is 2.
    max_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction 
        term. The default is 2.

    Returns
    -------
    rterms : list of dictionaries
        A list containing all the random terms.
    """
    
    rterms= []
    
    rng = default_rng()
    number_of_sites = rng.integers(low=min_num_sites, high=max_num_sites + 1,
                                   size=num_of_terms)
    strength = rng.uniform(low=min_strength, high=max_strength,
                           size=num_of_terms)
    
    for index, nsites in enumerate(number_of_sites):
        term = {}
        operator_indices = rng.integers(len(possible_operators), size=nsites)
        sites_list = []
        first = True
        
        for operator_index in operator_indices:
            
            operator = possible_operators[operator_index]
            
            if first == True:
                # The first operator has the interaction strength
                operator = strength[index] * operator
                first = False
                
            site = sites[rng.integers(len(sites))]
            # Every site should appear maximally once (Good luck)
            while site in sites_list:
                site = sites[rng.integers(len(sites))]
            
            term[site] = operator
        
        rterms.append(term)
    
    return rterms