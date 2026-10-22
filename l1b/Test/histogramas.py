from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

# Folder with your L1B NetCDF files
L1B_DIR = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-E2E\myoutput_l1b"
)

# Folder where the histograms will be saved
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# VNIR bands
BANDS = [
    "VNIR-0",
    "VNIR-1",
    "VNIR-2",
    "VNIR-3"
]


# ============================================================
# READ TOA
# ============================================================

def read_toa(filepath):
    """
    Read TOA variable from a L1B NetCDF file.
    """

    print()
    print("Reading:")
    print(filepath)

    with Dataset(filepath, "r") as nc:

        if "toa" not in nc.variables:
            raise KeyError(
                f"Variable 'toa' not found in file:\n{filepath}"
            )

        toa = np.array(
            nc.variables["toa"][:],
            dtype=float
        )

    return toa


# ============================================================
# HISTOGRAM
# ============================================================

def plot_histogram(toa, band):
    """
    Plot histogram using all pixels of the L1B TOA image.
    """

    # Convert 2D TOA image into a 1D vector
    values = toa.flatten()

    # Remove NaN and infinite values
    values = values[
        np.isfinite(values)
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_value = np.mean(values)
    median_value = np.median(values)
    min_value = np.min(values)
    max_value = np.max(values)
    std_value = np.std(values)

    print()
    print("------------------------------------------")
    print(f"BAND: {band}")
    print("------------------------------------------")
    print(f"TOA shape        = {toa.shape}")
    print(f"Number of pixels = {len(values)}")
    print(f"Mean TOA         = {mean_value:.6f}")
    print(f"Median TOA       = {median_value:.6f}")
    print(f"Min TOA          = {min_value:.6f}")
    print(f"Max TOA          = {max_value:.6f}")
    print(f"STD              = {std_value:.6f}")

    # --------------------------------------------------------
    # Create histogram
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        values,
        bins=80,
        edgecolor="black"
    )

    # --------------------------------------------------------
    # Format
    # --------------------------------------------------------

    plt.title(
        f"Histogram of L1B TOA - {band}"
    )

    plt.xlabel(
        "TOA"
    )

    plt.ylabel(
        "Number of pixels"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    figure_path = (
        FIGURES_DIR
        / f"histogram_{band}.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    print()
    print("Histogram saved:")
    print(figure_path)

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("L1B HISTOGRAMS")
    print("==========================================")

    print()
    print("L1B input folder:")
    print(L1B_DIR)

    print()
    print("Figures output folder:")
    print(FIGURES_DIR)

    # --------------------------------------------------------
    # Process every VNIR band
    # --------------------------------------------------------

    for band in BANDS:

        print()
        print("==========================================")
        print(f"PROCESSING {band}")
        print("==========================================")

        filepath = (
            L1B_DIR
            / f"l1b_toa_{band}.nc"
        )

        # Check file existence
        if not filepath.exists():

            print()
            print("ERROR - FILE NOT FOUND:")
            print(filepath)

            continue

        # Read TOA
        toa = read_toa(
            filepath
        )

        # Plot histogram
        plot_histogram(
            toa,
            band
        )

    print()
    print("==========================================")
    print("HISTOGRAMS FINISHED")
    print("==========================================")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()