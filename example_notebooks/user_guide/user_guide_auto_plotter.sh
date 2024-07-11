#!/bin/bash

# Assign command line arguments to variables
python_path="$1"
script_path="$2"
data_path="$3"
save_path="$4"

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
check_directory "$data_path"

# Check the fourth directory and create it if it does not exist
check_and_create_directory "$save_path"

# Example usage of the arguments
echo "Running Python from: $python_path"
echo "Loading scripts from: $script_path"
echo "Loading data from: $data_path"
echo "Saving plots to: $save_path"

# Function to move PDF files from a source directory to a save directory
move_pdfs_to_save_dir() {
    local source_directory="$1"
    local save_directory="$save_path"

    # Ensure the save directory exists
    check_and_create_directory "$save_directory"

    # Find all PDF files in the source directory and move them to the save directory
    find "$source_directory" -type f -name '*.pdf' -exec mv {} "$save_directory" \;
}

# Run the plotting of the trotter steps
plot_trotter() {
    local trotter_data="$data_path/trotter_user_guide"
    local trotter_script="$script_path/trotterisation_plots.py"
    
    # Call the Python script with the Python executable
    "$python_path" "$trotter_script" "$trotter_data" > /dev/null
    
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "Trotter plotting completed successfully."
    else
        echo "An error occurred during plotting the Totter errors."
    fi
}

# Run the plotting of the TTNO construction
plot_ttno_construction() {
    local ttno_script="$script_path/plotting_bd.py"
    local ttno_data_directory="$data_path/ttno_constr_user_guide"
    local ttno_data_file_names=("option_1_standard_mode" "option_2_standard_mode" "option_1_perfect_mode" "option_2_perfect_mode")
    for file_name in "${ttno_data_file_names[@]}"; do
        local ttno_data="$ttno_data_directory/$file_name"
        # Call the Python script with the Python executable
        "$python_path" "$ttno_script" "$ttno_data" > /dev/null

        # Check if the Python script ran successfully
        if [ $? -eq 0 ]; then
            echo "TTNO construction plotting completed successfully."
        else
            echo "An error occurred during plotting the TTNO construction."
        fi
    done
}

# Run the plotting of the TEBD results
plot_tebd() {
    local tebd_data_folder="$data_path/tebd"
    local tebd_simple_script="$script_path/user_guide_tebd_simple_magn_plot.py"
    local tebd_simple_data="$tebd_data_folder/simple_magn"
    # Call the Python script with the Python executable
    "$python_path" "$tebd_simple_script" "$tebd_simple_data" > /dev/null

    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TEBD plotting completed successfully."
    else
        echo "An error occurred during plotting the TEBD results."
    fi

    # Call the Python script with the Python executable
    local tebd_complex_script="$script_path/user_guide_tebd_dep_on_length_plots.py"
    local tebd_complex_data="$tebd_data_folder/dep_on_length"
    "$python_path" "$tebd_complex_script" "$tebd_complex_data" > /dev/null

    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TEBD plotting completed successfully."
    else
        echo "An error occurred during plotting the TEBD results."
    fi
}

# Run the plotting of the TDVP results
plot_tdvp() {
    local tdvp_data_folder="$data_path/tdvp"
    local tdvp_simple_script="$script_path/user_guide_tdvp_magn_plot.py"
    local tdvp_simple_data="$tdvp_data_folder/simple_magn"

    # Call the Python script with the Python executable
    "$python_path" "$tdvp_simple_script" "$tdvp_simple_data" > /dev/null

    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TDVP plotting completed successfully."
    else
        echo "An error occurred during plotting the TDVP results."
    fi

    # Call the Python script with the Python executable
    local tdvp_complex_script="$script_path/user_guide_tdvp_dep_length_plot.py"
    local tdvp_complex_data="$tdvp_data_folder/dep_length"
    "$python_path" "$tdvp_complex_script" "$tdvp_complex_data" 2 100 > /dev/null

    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "TDVP length dependend plotting completed successfully."
    else
        echo "An error occurred during plotting the length dependend TDVP results."
    fi
}

# You can add your script logic here, using the directory paths as needed
plot_trotter
plot_ttno_construction
plot_tebd
plot_tdvp
move_pdfs_to_save_dir "$data_path"