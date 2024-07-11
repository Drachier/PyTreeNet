# Simulations in the User Guide
This document explains how to rerun the simulations discussed in PyTreeNets user guide.

In general any script can be run on its own using a python environment in which `pytreenet` is installed. They always require at least the filepath under which to store the data to be passed as the first inline argument. Additional arguments required can be found by looking at the `input_handling` function of the respective script.

It is also possible to run all scripts in one go using the bash file `user_guide_auto_run.sh`. To run all the simulations with the inputs as in the user guide go into the folder with the user guide scripts and run
```bash
bash user_guide_auto_run.sh <path_to_python_with_pytreenet> <path_in_which_all_scipts_are_stored> <path_under_which_to_save_the_simulation_data>
```

In a similar manner all plots can be generated at the same time using the `user_guide_auto_plot.sh` script. In the folder with the script run
```bash
bash user_guide_auto_plot.sh <path_to_python_with_pytreenet> <path_in_which_all_scipts_are_stored> <save_path_of_the_simulation_data> <path_under_which_to_save_the_plots>
```
The plots will be saved as .pdf files in the specified folder.