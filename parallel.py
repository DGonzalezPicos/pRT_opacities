import subprocess
import multiprocessing
import os

import argparse

argparser = argparse.ArgumentParser()
# species prefix -s, default Fe
argparser.add_argument('--species', '-s', type=str, default='Fe', help='Species to run')
argparser.add_argument('--ncpus', '-n', type=int, default=1, help='Number of CPUs to use')

args = argparser.parse_args()

# Define the path to the PTpaths.ls file
PT_file = "input_data/PT_grids/PTpaths_high_dario.ls"
species = args.species
# script = "input/Kurucz_Ni.py"
script = f'input/example_kurucz.py'
ncpus = args.ncpus

# Print statement to check the path, script, and ncpus
print(f"** Running {script} for {species} with {ncpus} CPUs on {PT_file} **")

# Function to run the Python script with P and T as arguments
def run_script(pt_values):
    P, T = pt_values
    print(f"Running for P = {P} bar, T = {T} K")

    # Run the Python script
    try:
        result = subprocess.run(
            ["python", "main_kurucz.py", script, "-cs", "--P", str(P), "--T", str(T), "--species", species],
            # exampl: python main_kurucz.py input/example_kurucz.py -cs --P 0.1 --T 1000 --species Mn
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)

        # Check if the run was successful
        print(f"Successfully ran for P = {P} bar, T = {T} K")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred for P = {P} bar, T = {T} K")
        print(e.stderr)

# Read the PT_file and create a list of (P, T) tuples
def read_pt_file(file_path):
    pt_list = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                P, T = parts
                pt_list.append((P, T))
    return pt_list

# Run the jobs in parallel using multiprocessing
def run_parallel(pt_list, ncpus):
    with multiprocessing.Pool(ncpus) as pool:
        pool.map(run_script, pt_list)

if __name__ == "__main__":
    # Read the (P, T) values from the PT_file
    pt_values = read_pt_file(PT_file)

    # run for one value to download required files before running in parallel
    run_script(pt_values[0])
    # Run in parallel using ncpus
    run_parallel(pt_values, ncpus)

    print("All runs completed")

    # Combine grid
    subprocess.run(["python", "main_kurucz.py", script, "--combine_grid", "--species", species], check=True)

    # Convert to pRT2 format
    subprocess.run(
        ["python", "main_kurucz.py", script, "--convert_to_pRT2", "--ncpus", str(ncpus), "--species", species],
        check=True
    )
    print(f' Finished running {script} for {species} with {ncpus} CPUs on {PT_file} ')
# run with nohup as: nohup python parallel.py >& parallel.log &