from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import cKDTree


# ============================================================
# CONFIGURATION
# ============================================================

BAND = "VNIR-0"

L1C_DIR = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-L1C\myoutput"
)

# Cambia este nombre si tu archivo tiene otro nombre
L1C_FILE = L1C_DIR / f"l1c_toa_{BAND}.nc"

FIGURE_FILE = (
    L1C_DIR
    / f"spatial_sampling_distance_{BAND}.png"
)


# ============================================================
# HAVERSINE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Haversine distance between geographical coordinates.

    Parameters are in degrees.
    Output distance is in metres.
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
# READ LATITUDE / LONGITUDE
# ============================================================

def read_l1c_coordinates(filepath):

    print("Reading:")
    print(filepath)

    with Dataset(filepath, "r") as nc:

        print()
        print("Variables:")
        print(list(nc.variables.keys()))

        # Try common latitude names
        if "lat" in nc.variables:
            lat = np.array(nc.variables["lat"][:])

        elif "latitude" in nc.variables:
            lat = np.array(nc.variables["latitude"][:])

        else:
            raise KeyError(
                "Latitude variable not found."
            )

        # Try common longitude names
        if "lon" in nc.variables:
            lon = np.array(nc.variables["lon"][:])

        elif "longitude" in nc.variables:
            lon = np.array(nc.variables["longitude"][:])

        else:
            raise KeyError(
                "Longitude variable not found."
            )

    lat = lat.flatten()
    lon = lon.flatten()

    return lat, lon


# ============================================================
# NEAREST NEIGHBOUR
# ============================================================

def nearest_neighbour_distance(lat, lon):
    """
    Find the closest geographical neighbour of every L1C point.

    cKDTree is used to identify the closest neighbour.
    Haversine is then used for the actual distance.
    """

    # --------------------------------------------------------
    # Convert coordinates to 3D unit-sphere coordinates
    # --------------------------------------------------------

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    xyz = np.column_stack(
        (x, y, z)
    )

    # --------------------------------------------------------
    # Find the two closest points
    #
    # First one = the point itself
    # Second one = actual nearest neighbour
    # --------------------------------------------------------

    tree = cKDTree(xyz)

    _, indexes = tree.query(
        xyz,
        k=2
    )

    nearest_index = indexes[:, 1]

    # --------------------------------------------------------
    # Exact Haversine distance
    # --------------------------------------------------------

    distances = haversine(
        lat,
        lon,
        lat[nearest_index],
        lon[nearest_index]
    )

    return distances


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("L1C SPATIAL SAMPLING DISTANCE")
    print("==========================================")
    print()

    # --------------------------------------------------------
    # Read L1C
    # --------------------------------------------------------

    lat, lon = read_l1c_coordinates(
        L1C_FILE
    )

    print()
    print("Number of L1C points:")
    print(len(lat))

    # --------------------------------------------------------
    # Calculate nearest-neighbour sampling distance
    # --------------------------------------------------------

    distance = nearest_neighbour_distance(
        lat,
        lon
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("Spatial sampling statistics:")
    print(
        f"Mean   = {np.mean(distance):.3f} m"
    )
    print(
        f"Median = {np.median(distance):.3f} m"
    )
    print(
        f"Min    = {np.min(distance):.3f} m"
    )
    print(
        f"Max    = {np.max(distance):.3f} m"
    )
    print(
        f"STD    = {np.std(distance):.3f} m"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        np.arange(len(distance)),
        distance,
        ".",
        markersize=3,
        label="L1C spatial sampling"
    )

    # Expected MGRS grid size
    plt.axhline(
        y=100.0,
        linestyle="--",
        label="Expected MGRS sampling = 100 m"
    )

    plt.title(
        f"Spatial sampling distance - {BAND}"
    )

    plt.xlabel(
        "L1C point [-]"
    )

    plt.ylabel(
        "Haversine distance [m]"
    )

    plt.grid(True)

    plt.legend()

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