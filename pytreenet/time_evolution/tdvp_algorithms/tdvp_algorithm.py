"""
Implements the time-dependent variational principle (TDVP) for tree tensor
networks.

Reference:
    [1] D. Bauernfeind, M. Aichhorn; "Time Dependent Variational Principle for Tree
        Tensor Networks", DOI: 10.21468/SciPostPhys.8.2.024
"""
from __future__ import annotations
from typing import Union, List
from dataclasses import dataclass

from ..time_evolution import TimeEvoMode
from ..ttn_time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from ...util.tensor_splitting import SplitMode
from ...ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TTNO
from ...operators.tensorproduct import TensorProduct
from ...contractions.sandwich_caching import SandwichCache
from ..time_evo_util.effective_time_evolution import single_site_time_evolution
from ..time_evo_util.update_path import TDVPUpdatePathFinder

@dataclass
class TDVPConfig(TTNTimeEvolutionConfig):
    """
    Configuration class for TDVP algorithms.

    Attributes:
        update_path (List[str]): The order in which the nodes are updated.
        orthogonalisation_path (List[List[str]]): The path along which the
            TTNS has to be orthogonalised between each node update.
    """
    time_evo_mode: TimeEvoMode = TimeEvoMode.FASTEST

class TDVPAlgorithm(TTNTimeEvolution):
    """
    The general abstract class of a TDVP algorithm.
    
    Subclasses the general time evolution for tree tensor networks.

    Attributes:
        initial_state (TreeTensorNetworkState): The initial state of the system.
        hamiltonian (TTNO): The Hamiltonian under which to time-evolve the system.
        time_step_size (float): The size of one time-step.
        final_time (float): The final time until which to run the evolution.
        operators (Union[TensorProduct, List[TensorProduct]]): Operators to be
            measured during the time-evolution.
        update_path (List[str]): The order in which the nodes are updated.
        orthogonalisation_path (List[List[str]]): The path along which the
            TTNS has to be orthogonalised between each node update.
        partial_tree_cache (PartialTreeCacheDict): A dictionary to hold
            already contracted subtrees of the TTNS.
    """
    config_class = TDVPConfig

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],
                 config: Union[TDVPConfig,None] = None) -> None:
        """
        Initilises an instance of a TDVP algorithm.

        Args:
            intial_state (TreeTensorNetworkState): The initial state of the
                system.
            hamiltonian (TTNO): The Hamiltonian in TTNO form under which to
                time-evolve the system.
            time_step_size (float): The size of one time-step.
            final_time (float): The final time until which to run the
                evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): Operators
                to be measured during the time-evolution.
        """
        assert len(initial_state.nodes) == len(hamiltonian.nodes)
        self.hamiltonian = hamiltonian
        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators,
                         config)
        self.config: TDVPConfig
        self.update_path = self._finds_update_path()
        self.orthogonalization_path = self._find_tdvp_orthogonalization_path(self.update_path)
        self._orthogonalize_init()

        # Caching for speed up
        self.partial_tree_cache = self._init_partial_tree_cache()

    def _orthogonalize_init(self, force_new: bool=False):
        """
        Orthogonalises the state to the start of the TDVP update path.
        
        If the state is already orthogonalised, the orthogonalisation center
        is moved to the start of the update path.

        Args:
            force_new (bool, optional): If True a complete orthogonalisation
                is always enforced, instead of moving the orthogonality center.
                Defaults to False.
        """
        if self.state.orthogonality_center_id is None or force_new:
            self.state.canonical_form(self.update_path[0],
                                      mode=SplitMode.KEEP)
        else:
            self.state.move_orthogonalization_center(self.update_path[0],
                                                     mode=SplitMode.KEEP)

    def _find_tdvp_orthogonalization_path(self,
                                          update_path: List[str]) -> List[List[str]]:
        """
        Find the TDVP orthogonalisation path.

        Args:
            update_path (List[str]): The path along which tdvp updates sites.

        Returns:
            List[List[str]]: a list of paths, along which the TTNS should be
            orthogonalised between every node update.
        """
        orthogonalization_path = []
        for i in range(len(update_path)-1):
            sub_path = self.state.path_from_to(update_path[i], update_path[i+1])
            orthogonalization_path.append(sub_path[1::])
        return orthogonalization_path

    def _finds_update_path(self) -> List[str]:
        """
        Finds the update path for this TDVP Algorithm.

        Overwrite to create custom update paths for specific tree topologies.

        Returns:
            List[str]: The order in which the nodes in the TTN should be time
                evolved.
        """
        return TDVPUpdatePathFinder(self.initial_state).find_path()

    def _init_partial_tree_cache(self) -> SandwichCache:
        """
        Initialises the partial tree cache such that for all sites the bra,
        ket, and Hamiltonian tensor are contracted and saved in the cache,
        except for the first site to be updated.
        """
        return SandwichCache.init_cache_but_one(self.state,
                                                self.hamiltonian,
                                                self.update_path[0])

    def update_tree_cache(self, node_id: str, next_node_id: str):
        """
        Updates a tree tensor for given node identifiers.
        
        Updates the tree cache tensor that ends in the node with identifier
        `node_id` and has open legs pointing towards the neighbour node with
        identifier `next_node_id`.

        Remains for now to keep compatability. Will be removed in the future.

        Args:
            node_id (str): The identifier of the node to which this cache
                corresponds.
            next_node_id (str): The identifier of the node to which the open
                legs of the tensor point.
        """
        self.partial_tree_cache.update_tree_cache(node_id, next_node_id)

    def _update_site(self, node_id: str,
                     time_step_factor: float = 1):
        """
        Updates a single site using the effective Hamiltonian for that site.

        Args:
            node_id (str): The identifier of the site to update.
            time_step_factor (float, optional): A factor that should be
                multiplied with the internal time step size. Defaults to 1.
        """
        updated_tensor = single_site_time_evolution(node_id,
                                                    self.state,
                                                    self.hamiltonian,
                                                    self.time_step_size * time_step_factor,
                                                    self.partial_tree_cache,
                                                    forward=True,
                                                    mode=self.config.time_evo_mode)
        self.state.tensors[node_id] = updated_tensor

    def _move_orth_and_update_cache_for_path(self, path: List[str]):
        """
        Moves the orthogonalisation center and updates all required caches
        along a given path.

        Args:
            path (List[str]): The path to move from. Should start with the
                orth center and end at with the final node. If the path is empty
                or only the orth center nothing happens.
        """
        if len(path) == 0:
            return
        assert self.state.orthogonality_center_id == path[0]
        for i, node_id in enumerate(path[1:]):
            self.state.move_orthogonalization_center(node_id,
                                                     mode=SplitMode.KEEP)
            previous_node_id = path[i] # +0, because path starts at 1.
            self.partial_tree_cache.update_tree_cache(previous_node_id, node_id)
