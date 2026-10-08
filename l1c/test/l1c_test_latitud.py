from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.io.readGeodetic import readGeodetic


# ============================================================
# CONFIGURATION
# ============================================================

GEO_DIR = (
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021"
    r"\EODP-TS-L1C\input\gm_alt100_act_150"
)

GEO_FILE = "geolocation.nc"

FIGURE_FILE = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021"
    r"\EODP-TS-L1C\myoutput\spatial_sampling_vs_latitude.png"
)


# ============================================================
# HAVERSINE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Haversine distance between two geographical points.

    Input coordinates: degrees
    Output distance: metres
    """

    R = 6371000.0  # Earth radius [m]

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1.0 - a)
    )

    return R * c


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("SPATIAL SAMPLING DISTANCE VS LATITUDE")
    print("==========================================")

    # --------------------------------------------------------
    # Read geolocation
    # --------------------------------------------------------

    lat, lon = readGeodetic(
        GEO_DIR,
        GEO_FILE
    )

    print()
    print("Latitude shape:", lat.shape)
    print("Longitude shape:", lon.shape)

    # --------------------------------------------------------
    # Distances between consecutive ACT pixels
    # for every ALT line
    # --------------------------------------------------------

    lat1 = lat[:, :-1]
    lon1 = lon[:, :-1]

    lat2 = lat[:, 1:]
    lon2 = lon[:, 1:]

    distances = haversine(
        lat1,
        lon1,
        lat2,
        lon2
    )

    # Representative latitude between both pixels
    latitude = (
        lat1 + lat2
    ) / 2.0

    # --------------------------------------------------------
    # Flatten for plotting
    # --------------------------------------------------------

    distances_flat = distances.flatten()
    latitude_flat = latitude.flatten()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("Spatial sampling statistics:")
    print(
        f"Mean = {np.mean(distances_flat):.3f} m"
    )
    print(
        f"Min  = {np.min(distances_flat):.3f} m"
    )
    print(
        f"Max  = {np.max(distances_flat):.3f} m"
    )

    print()
    print("Latitude range:")
    print(
        f"Min = {np.min(latitude_flat):.6f} deg"
    )
    print(
        f"Max = {np.max(latitude_flat):.6f} deg"
    )

    # --------------------------------------------------------
    # Sort by latitude
    # --------------------------------------------------------

    order = np.argsort(latitude_flat)

    latitude_sorted = latitude_flat[order]
    distances_sorted = distances_flat[order]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        latitude_sorted,
        distances_sorted,
        s=4
    )

    plt.title(
        "Spatial sampling distance vs Latitude"
    )

    plt.xlabel(
        "Latitude [deg]"
    )

    plt.ylabel(
        "Spatial sampling distance [m]"
    )

    plt.grid(True)

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        FIGURE_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    print()
    print("Figure saved:")
    print(FIGURE_FILE)

    plt.show()


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()