import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def parse_arguments() -> dict:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="1D Curve Experiment")
    parser.add_argument(
        "--vectorD",
        type=int,
        default=256,
        required=False,
        help="Dimensionality of the vectors."
    )
    parser.add_argument(
        "--scratch_dir",
        type=str,
        default="scratch",
        help="Directory to store experiment results."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="1d_curve_experiment",
        help="Name of the experiment."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the experiment on."
    )
    args = parser.parse_args()
    return vars(args)

def main():
    """
    Main entry point for the script.
    """
    args: dict = parse_arguments()

    print(f"Running 1D curve experiment.\n")
    print(f"Parameters:")
    print(json.dumps(args, indent=4))
    print("\n")

    # -------------------------------
    # Set up experiment directory
    # -------------------------------
    experiment_dir: str = os.path.join(args["scratch_dir"], args["experiment_name"])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        print(f"Created experiment directory at {experiment_dir}.")
    else:
        raise FileExistsError("Experiment directory already exists. Please choose a different name or delete the existing directory.")

if __name__ == "__main__":
    main()