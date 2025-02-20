
from pytreenet.ttno.state_diagram import TTNOFinder

from script_mpo import main as main_mpo
from script_tree import main as main_tree
from plot import main as main_plot
from plot_mode_comparison import main as main_comp_plot

METADATA = {
    "min_mol": 2,
    "max_mol": 15,
    "num_baths_min": 1,
    "num_baths_max": 15}

HOOMS = [True, False]
MODES = [TTNOFinder.BIPARTITE, TTNOFinder.SGE]
DTYPE = ["avg_bond_dims", "max_bond_dims"]

def main_simulation(metadata: dict[str,int],
                    construction: bool = False,
                    plot: bool = False,
                    save: bool = True):
    """
    Run the simulations.

    Args:
        metadata (dict[str,int]): Gives the minimum and maximum numbers of
            molecules and baths for which to run the constructions.
        construction (bool): Whether to run the construction of the TTNO.
            The data will be saved. Defaults to False.
        plot (bool): Whether to plot the results. Defaults to False.
        save (bool): Whether to save the plots. Defaults to True.

    """
    if construction:
        print("Construction TTNOs!")
        main_mpo(metadata)
        main_tree(metadata)

    if plot:
        print("Plotting Results")
        for hom in HOOMS:
            for mode in MODES:
                main_plot(homogenous=hom,
                          mode=mode,
                          save=save)
        for hom in HOOMS:
            for dtype in DTYPE:
                main_comp_plot(homogenous=hom,
                               datatype=dtype,
                               save=save)

    if not (construction or plot):
        print("Doin Nothing!")

if __name__ == "__main__":
    main_simulation(METADATA,
                    construction=True,
                    plot=True,
                    save=True)
