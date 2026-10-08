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
    r"\EODP-TS-L1C\myoutput\spatial_sampling_vs_ACT.png"
)

ALT_LINE = 50


# ============================================================
# HAVERSINE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):

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
    print("SPATIAL SAMPLING DISTANCE VS ACT")
    print("==========================================")

    # Read geolocation using the same reader as l1c.py
    lat, lon = readGeodetic(
        GEO_DIR,
        GEO_FILE
    )

    print()
    print("Latitude shape:", lat.shape)
    print("Longitude shape:", lon.shape)

    # --------------------------------------------------------
    # Central ALT line
    # --------------------------------------------------------

    lat_line = lat[ALT_LINE, :]
    lon_line = lon[ALT_LINE, :]

    # --------------------------------------------------------
    # Distance between consecutive ACT pixels
    # --------------------------------------------------------

    distances = haversine(
        lat_line[:-1],
        lon_line[:-1],
        lat_line[1:],
        lon_line[1:]
    )

    act_pixels = np.arange(
        len(distances)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("Spatial sampling statistics:")
    print(f"Mean = {np.mean(distances):.3f} m")
    print(f"Min  = {np.min(distances):.3f} m")
    print(f"Max  = {np.max(distances):.3f} m")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        act_pixels,
        distances
    )

    plt.title(
        f"Spatial sampling distance - ALT {ALT_LINE}"
    )

    plt.xlabel(
        "ACT pixel [-]"
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


if __name__ == "__main__":
    main()