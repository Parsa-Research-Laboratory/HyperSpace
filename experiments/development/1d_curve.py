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
        "--num_points",
        type=int,
        default=25,
        help="Number of points to sample on the curve."
    )
    parser.add_argument(
        "--scratch_dir",
        type=str,
        default="scratch",
        help="Directory to store experiment results."
    )
    parser.add_argument(
        "--overwrite_scratch",
        action="store_true",
        help="Overwrite the scratch directory if it exists."
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
        if args["overwrite_scratch"]:
            print(f"Overwriting existing experiment directory at {experiment_dir}.")
        else:
            raise FileExistsError("Experiment directory already exists. Please choose a different name or delete the existing directory.")

    # -------------------------------
    # Save parameters to a JSON file
    # -------------------------------
    params_path: str = os.path.join(experiment_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(args, f, indent=4)
    print(f"Saved experiment parameters to {params_path}.")

    # -------------------------------
    # Generate base curve data
    # -------------------------------
    x_values: np.ndarray = np.linspace(-5, 5, 100)
    y_values: np.ndarray = np.sin(x_values)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="sin(x)", color="blue")
    plt.title("1D Curve: sin(x)")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    curve_path: str = os.path.join(experiment_dir, "curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved curve plot to {curve_path}.")

    # -------------------------------
    # Sample points on the curve
    # -------------------------------
    sampled_x: np.ndarray = np.linspace(-5, 5, args["num_points"])
    sampled_y: np.ndarray = np.sin(sampled_x)
    samples_path: str = os.path.join(experiment_dir, "sampled_points.npz")
    np.savez(samples_path, x=sampled_x, y=sampled_y)
    print(f"Saved sampled points to {samples_path}.")

    # -------------------------------
    # Plot sampled points
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="sin(x)", color="blue")
    plt.scatter(sampled_x, sampled_y, color="red", label="Sampled Points")
    plt.title("Sampled Points on 1D Curve")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    sampled_curve_path: str = os.path.join(experiment_dir, "sampled_curve.png")
    plt.savefig(sampled_curve_path)
    plt.close()
    print(f"Saved sampled curve plot to {sampled_curve_path}.")



if __name__ == "__main__":
    main()