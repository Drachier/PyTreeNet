from typing import Union

from .time_evolution import TimeEvolution

def run_time_dependt(time_evolution: TimeEvolution,
                     evaluation_time: Union[int,"inf"] = 1,
                     pgbar: bool = True):
    """
    Runs a time evolution that has a time dependent Hamiltonian.

    Args:
        time_evolution: The time evolution object. It must contain a
            hamiltonian attribute that has an update method.
        evaluation_time: The time at which the time evolution is evaluated.
        pgbar: If True, a progress bar is shown.

    """
    time_evolution.init_results(evaluation_time)
    for i in time_evolution.create_run_tqdm(pgbar):
        if i != 0:  # We also measure the initial expectation_values
            time_evolution.run_one_time_step()
            time_evolution.hamiltonian.update(time_evolution.time_step_size)
        time_evolution.evaluate_and_save_results(evaluation_time, i)
