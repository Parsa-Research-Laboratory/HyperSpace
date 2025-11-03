import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import Tensor

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
    x_values: np.ndarray = np.linspace(0, 2 * np.pi, 100)
    y_values: np.ndarray = np.sin(x_values)   
    x_values /= (2 * np.pi)  # Normalize x to [0, 1]
    y_values += 1
    y_values /= 2  # Normalize y to [0, 1] 
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="sin(x)", color="blue")
    plt.title("1D Curve: sin(x)")
    plt.xlabel("x (rad normalized to [0, 1])")
    plt.ylabel("sin(x) (normalized to [0, 1])")
    plt.legend()
    curve_path: str = os.path.join(experiment_dir, "curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved curve plot to {curve_path}.")

    # -------------------------------
    # Sample points on the curve
    # -------------------------------
    sampled_x: np.ndarray = np.linspace(0, 2 * np.pi, args["num_points"])
    sampled_y: np.ndarray = np.sin(sampled_x)
    sampled_x /= (2 * np.pi)  # Normalize x to [0, 1]
    sampled_y += 1
    sampled_y /= 2  # Normalize y to [0, 1]
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
    plt.xlabel("x (rad normalized to [0, 1])")
    plt.ylabel("sin(x) (normalized to [0, 1])")
    plt.legend()
    sampled_curve_path: str = os.path.join(experiment_dir, "sampled_curve.png")
    plt.savefig(sampled_curve_path)
    plt.close()
    print(f"Saved sampled curve plot to {sampled_curve_path}.")

    # -----------------------------------
    # Extract the dataset for HyperSpace
    # -----------------------------------
    X_true: Tensor = torch.from_numpy(sampled_x).float().unsqueeze(1)
    Y_true: Tensor = torch.from_numpy(sampled_y).float().unsqueeze(1)

    # --------------------------------------
    # Run HyperSpace
    # --------------------------------------
    X_hat, y_hat, run_info = run_hyperspace_experiment(X_true, Y_true, args)

    # ---------------------------------------
    # Save predictions
    # ---------------------------------------
    predictions_path: str = os.path.join(experiment_dir, "predictions.npz")
    np.savez(predictions_path, X_hat=X_hat.numpy(), y_hat=y_hat.numpy())
    print(f"Saved predictions to {predictions_path}.")

    # Plot predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="sin(x)", color="blue")
    plt.scatter(sampled_x, sampled_y, color="red", label="Sampled Points", alpha=0.5)
    plt.scatter(X_hat.numpy(), y_hat.numpy(), color="green", label="Predictions", alpha=0.5)
    plt.title("HyperSpace Predictions on 1D Curve")
    plt.xlabel("x (rad normalized to [0, 1])")
    plt.ylabel("sin(x) (normalized to [0, 1])")
    plt.legend()
    predictions_curve_path: str = os.path.join(experiment_dir, "predictions_curve.png")
    plt.savefig(predictions_curve_path)
    plt.close()
    print(f"Saved predictions curve plot to {predictions_curve_path}.")

    # ----------------------------------
    # Calculate and print error metrics
    # ----------------------------------
    mse: float = torch.mean((y_hat - Y_true) ** 2).item()
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    # ----------------------------------
    # print run info
    # ----------------------------------
    print("Run Info:")
    print(json.dumps(run_info, indent=4))

    # ----------------------------------
    # Finish
    # ----------------------------------
    print("1D curve experiment completed successfully.")

def run_hyperspace_experiment(X_true: Tensor, Y_true: Tensor, args: dict) -> tuple[Tensor, Tensor]:
    """
    Run the HyperSpace experiment using the provided dataset and parameters.
    """

    from hyperspace.core import (
        CleanupModule,
        MemoryStorageModule,
        PositionalEncoderModule,
        PositionalInversionModule,
        RegressionModule,
        ValueEncoderModule
    )

    # Placeholder for HyperSpace experiment logic
    print("Running HyperSpace experiment... (this is a placeholder)")
    # Here you would initialize your HyperSpace model, train it, and get predictions
    X_hat = X_true  # Placeholder: replace with actual predictions
    y_hat = Y_true.clone()  # Placeholder: replace with actual predictions
    y_hat += torch.randn_like(y_hat) * 0.05  # Add slight noise for demonstration
    return X_hat, y_hat, {}


if __name__ == "__main__":
    main()