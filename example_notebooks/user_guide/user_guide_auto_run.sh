#!/bin/bash

# Assign command line arguments to variables
python_path="$1"
script_path="$2"
data_path="$3"

# Function to check if a directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        echo "Directory $1 does not exist."
        exit 1
    fi
}

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "File $1 does not exist."
        exit 1
    fi
}

# Function to check the last directory and create it if it does not exist
check_and_create_directory() {
    if [ ! -d "$1" ]; then
        echo "Save directory $1 does not exist, creating it now."
        mkdir -p "$1"
    fi
}

# Check if the first three arguments are valid directory paths
check_file "$python_path"
check_directory "$script_path"
check_and_create_directory "$data_path"

# Example usage of the arguments
echo "Running Python from: $python_path"
echo "Loading scripts from: $script_path"
echo "Saving data to: $data_path"

# Run the simulation of the trotter steps
run_trotter() {
    local trotter_script="$script_path/trotterisation_scaling.py"
    local save_file="$data_path/trotter_user_guide.hdf5"

    # Run the trotterisation scaling script
    "$python_path" "$trotter_script" "$save_file" > /dev/null
    
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "Trotter simulation completed successfully."
    else
        echo "An error occurred during simulation of the Totterisation."
    fi
}

# Run the plotting of the TTNO construction
run_ttno_construction() {
    local ttno_script="$script_path/random_hamiltonians_to_TTNO_utils.py"
    
    local save_file="$data_path/ttno_constr_user_guide/option_1_standard_mode"
    "$python_path" "$ttno_script" "$save_file" 1 > /dev/null
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TTNO Construction completed successfully."
    else
        echo "An error occurred during TTNO Construction."
    fi

    save_file="$data_path/ttno_constr_user_guide/option_2_standard_mode"
    "$python_path" "$ttno_script" "$save_file" 2 > /dev/null
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TTNO Construction completed successfully."
    else
        echo "An error occurred during TTNO Construction."
    fi

    save_file="$data_path/ttno_constr_user_guide/option_1_perfect_mode"
    "$python_path" "$ttno_script" "$save_file" 1 --mode "CM" > /dev/null
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TTNO Construction completed successfully."
    else
        echo "An error occurred during TTNO Construction."
    fi


    save_file="$data_path/ttno_constr_user_guide/option_2_perfect_mode"
    "$python_path" "$ttno_script" "$save_file" 2 --mode "CM" > /dev/null
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TTNO Construction completed successfully."
    else
        echo "An error occurred during TTNO Construction."
    fi

}

# Run the TEBD simulations
run_tebd_simulation() {
    local tebd_script="$script_path/user_guide_tebb_simple_magn.py"
    local save_file="$data_path/tebd/simple_magn"
    "$python_path" "$tebd_script" "$save_file" > /dev/null
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TEBD simulation completed successfully."
    else
        echo "An error occurred during TEBD simulation."
    fi

    tebd_script="$script_path/user_guide_tebb_dep_on_length.py"
    save_file="$data_path/tebd/dep_on_length"
    "$python_path" "$tebd_script" "$save_file" 0 500 > /dev/null
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TEBD simulation completed successfully."
    else
        echo "An error occurred during TEBD simulation."
    fi
}

# Run the TDVP simulations
run_tdvp_simulation() {
    local tdvp_script="$script_path/user_guide_tdvp_magn.py"
    local save_file="$data_path/simple_magn"
    "$python_path" "$tdvp_script" "$save_file" > /dev/null
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TDVP simulation completed successfully."
    else
        echo "An error occurred during TDVP simulation."
    fi

    # Run the initial length dependent simulation
    tdvp_script="$script_path/user_guide_tdvp_dep_length.py"
    save_file="$data_path/dep_length"
    for i in 0 1 2; do
        "$python_path" "$tdvp_script" "$save_file" 0 50 $i > /dev/null
        # Check if the Python script ran successfully
        if [ $? -eq 0 ]; then
            echo "TDVP simulation for i=$i completed successfully."
        else
            echo "An error occurred during TDVP simulation for i=$i."
        fi
    done

    # Run the 1TDVP simulatons with adapted bond dimensions
    tdvp_script="$script_path/user_guide_tdvp_dep_length_combined.py"
    "$python_path" "$tdvp_script" "$save_file" 0 50 > /dev/null
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TDVP simulation for adapted bond dimensions completed successfully."
    else
        echo "An error occurred during TDVP simulation for adapted bond dimensions."
    fi

}

run_trotter
run_ttno_construction
run_tebd_simulation
run_tdvp_simulation

