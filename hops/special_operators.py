import numpy as np

def create_bosonic_operators(dimension=2):
    
    creation_op = np.eye(dimension, k=-1)
    annihilation_op = np.conj(creation_op.T)
    
    number_vector = np.asarray(range(0,dimension))
    number_op = np.diag(number_vector)
    
    return creation_op, annihilation_op, number_op
