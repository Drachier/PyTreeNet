"""
This module returns an abstract Hamiltonian for the molecule cavity system.
"""
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct

def generate_hamiltonian(num_mol: int = 2,
                         num_mol_bath: int = 4,
                         num_cav_bath: int = 4,
                         homogenous_mc: bool = False,
                         homogenous_mm: bool = False
                         ) -> Hamiltonian:
    """
    Generate the Hamiltonian for the molecule cavity system.

    The convention is that the cavity site is named "cavity" and the molecule
    sites are named "mol0", "mol1", etc. The bath sites are named "cav_bath0",
    "cav_bath1", etc. for the cavity bath and "mol_bath0_0", "mol_bath0_1",
    etc., where the first integer is the molecule index and the second integer
    is the bath index. The twin sites of the cavity and the molecules are named
    "cavity_twin" and "mol0_twin", "mol1_twin", etc., respectively.

    Args:
        num_mol (int, optional): Number of molecules. Defaults to 2.
        num_mol_bath (int, optional): Number of molecular bath modes. Defaults
            to 4.
        num_cav_bath (int, optional): Number of cavity bath modes. Defaults
            to 4.
        homogenous_mc (bool, optional): Whether the cavity-molecule coupling
            is the same for all molecules. Defaults to False.
        homogenous_mm (bool, optional): Whether the molecule-molecule coupling
            is the same for all pairs of molecules. Defaults to False.
    
    Returns:
        Hamiltonian: The Hamiltonian for the molecule cavity system.
    """
    ham = Hamiltonian()
    # Cavity part
    term = TensorProduct({"cavity": "Hc"})
    term_twin = TensorProduct({"cavity_twin": "Hc"})
    ham.add_multiple_terms([(1,"1",term), (-1,"1",term_twin)])
    # Cavity-Molecule interaction
    for i in range(num_mol):
        term = TensorProduct({"cavity": "Cc", "mol"+str(i): "Cm"})
        term_twin = TensorProduct({"cavity_twin": "Cc",
                                   "mol"+str(i)+"_twin": "Cm"})
        # facmc is the coupling factor sqrt(2w_c^3)*eta
        if homogenous_mc:
            ham.add_multiple_terms([(1,"facmc",term) ,
                                    (-1, "facmc",term_twin)])
        else:
            ham.add_multiple_terms([(1, f"facmc{i}", term),
                                    (-1, f"facmc{i}", term_twin)])
    # Molecule part
    for i in range(num_mol):
        term = TensorProduct({"mol"+str(i): "Hm"})
        term_twin = TensorProduct({"mol"+str(i)+"_twin": "-Hm"})
        ham.add_multiple_terms([(1,"1",term), (-1,"1",term_twin)])
    # Molecule-Molecule interaction
    for i in range(num_mol):
        for j in range(i+1, num_mol):
            if i < j:
                term = TensorProduct({"mol"+str(i): "Cm", "mol"+str(j): "Cm"})
                term_twin = TensorProduct({"mol"+str(i)+"_twin": "Cm",
                                        "mol"+str(j)+"_twin": "Cm"})
                # Delta is the dipole-dipole strength between i and j
                if homogenous_mm:
                    ham.add_multiple_terms([(1, "Delta", term),
                                            (-1, "Delta", term_twin)])
                else:
                    ham.add_multiple_terms([(1, f"Delta{i}{j}", term),
                                            (-1, f"Delta{i}{j}", term_twin)])
    # Cavity-Bath part
    for i in range(num_cav_bath):
        term = TensorProduct({"cav_bath"+str(i): "N"})
        ham.add_term((-1,"im*gammac",term))
    # Molecule-Bath part
    for i in range(num_mol):
        for j in range(num_mol_bath):
            bath_id = "mol_bath"+str(i)+"_"+str(j)
            term = TensorProduct({bath_id: "N"})
            if homogenous_mm:
                # All molecules have the same interaction with their baths
                ham.add_term((-1,f"i*gamma{j}",term))
            else:
                ham.add_term((-1,f"i*gamma{i}_{j}",term))
    # Cavity Bath Interaction
    for i in range(num_cav_bath):
        bath_id = "cav_bath"+str(i)
        term = TensorProduct({"cavity": "Cc",
                              bath_id: "b"})
        term_twin = TensorProduct({"cavity_twin": "Cc",
                                   bath_id: "b"})
        ham.add_multiple_terms([(1,"sqlambdac",term),
                                (-1,"sqlambdac",term_twin)])
        term = TensorProduct({"cavity": "Cc",
                              bath_id: f"etac{i}*bdag"})
        term_twin = TensorProduct({"cavity_twin": "Cc",
                                   bath_id: f"-etac{i}dag*bdag"})
        ham.add_multiple_terms([(1,f"eta{i}",term),
                                (-1,f"eta{i}",term_twin)])
    # Molecule Bath Interaction
    for i in range(num_mol):
        for j in range(num_mol_bath):
            bath_id = "mol_bath"+str(i)+"_"+str(j)
            mol_id = "mol"+str(i)
            term = TensorProduct({mol_id: "Cm",
                                  bath_id: "b"})
            term_twin = TensorProduct({mol_id+"_twin": "Cm",
                                      bath_id: "b"})
            if homogenous_mm:
                ham.add_multiple_terms([(1,"sqlambdam",term),
                                        (-1,"sqlambdam_dag",term_twin)])
            else:
                ham.add_multiple_terms([(1,f"sqlambdam{i}",term),
                                        (-1,f"sqlambdam{i}_dag",term_twin)])
            term = TensorProduct({mol_id: "Cm",
                                  bath_id: "bdag"})
            term_twin = TensorProduct({mol_id+"_twin": "Cm",
                                      bath_id: "bdag"})
            if homogenous_mm:
                ham.add_multiple_terms([(1,f"etam_bath{j}",term),
                                        (-1,f"etam_bath{j}_dag",term_twin)])
            else:
                ham.add_multiple_terms([(1,f"etam{i}_{j}",term),
                                        (-1,f"etam{i}_{j}_dag",term_twin)])
    return ham

if __name__ == "__main__":
    ham = generate_hamiltonian()
    print(ham)
