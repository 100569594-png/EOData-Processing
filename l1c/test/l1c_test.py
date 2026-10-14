
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

# ============================================================
# DIRECTORIES
# ============================================================

TEST_ROOT = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-L1C"
)

REF_DIR = TEST_ROOT / "output"
MY_DIR = TEST_ROOT / "myoutput"

REPORT_FILE = (
    Path(__file__).resolve().parent
    / "cross_validation_L1C.txt"
)

LIMIT = 0.001  # Maximum relative error [%]
BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]


# ============================================================
# READ NETCDF
# ============================================================

def read_l1c(filename):

    with Dataset(str(filename), "r") as nc:
        lat = np.asarray(nc.variables["lat"][:], dtype=float)
        lon = np.asarray(nc.variables["lon"][:], dtype=float)
        toa = np.asarray(nc.variables["toa"][:], dtype=float)

    return lat.ravel(), lon.ravel(), toa.ravel()


# ============================================================
# CROSS VALIDATION
# ============================================================

def main():

    lines = []
    passed_tests = 0

    header = (
        f"{'FILE':30s} "
        f"{'POINTS':>8s} "
        f"{'MAX ABS':>14s} "
        f"{'MAX REL %':>14s} "
        f"{'RESULT':>8s}"
    )

    lines.append(header)
    lines.append("-" * 85)

    for band in BANDS:

        filename = f"l1c_toa_{band}.nc"

        ref_file = REF_DIR / filename
        my_file = MY_DIR / filename

        try:
            ref_lat, ref_lon, ref_toa = read_l1c(ref_file)
            my_lat, my_lon, my_toa = read_l1c(my_file)

            if len(ref_toa) != len(my_toa):
                raise ValueError("Different number of points")

            # Sort both datasets by geographic coordinates.
            # This avoids comparing points by their original index.

            ref_order = np.lexsort((ref_lon, ref_lat))
            my_order = np.lexsort((my_lon, my_lat))

            ref_lat = ref_lat[ref_order]
            ref_lon = ref_lon[ref_order]
            ref_toa = ref_toa[ref_order]

            my_lat = my_lat[my_order]
            my_lon = my_lon[my_order]
            my_toa = my_toa[my_order]

            if not (
                np.all(np.isfinite(ref_lat))
                and np.all(np.isfinite(ref_lon))
                and np.all(np.isfinite(ref_toa))
                and np.all(np.isfinite(my_lat))
                and np.all(np.isfinite(my_lon))
                and np.all(np.isfinite(my_toa))
            ):
                raise ValueError("Non-finite values found")

            # Check that geographic points coincide exactly.
            if not (
                np.array_equal(ref_lat, my_lat)
                and np.array_equal(ref_lon, my_lon)
            ):
                raise ValueError("Geographic coordinates differ")

            # Compare radiance values.
            diff = np.abs(my_toa - ref_toa)

            max_abs = float(np.max(diff))

            valid = np.abs(ref_toa) > 1e-12

            if np.any(valid):
                max_rel = float(np.max(
                    diff[valid] / np.abs(ref_toa[valid]) * 100
                ))
            else:
                max_rel = 0.0

            # Also check points with zero reference radiance.
            zero_error = (
                float(np.max(diff[~valid]))
                if np.any(~valid) else 0.0
            )

            passed = max_rel < LIMIT and zero_error == 0
            status = "PASS" if passed else "FAIL"

            if passed:
                passed_tests += 1

            line = (
                f"{filename:30s} "
                f"{len(ref_toa):8d} "
                f"{max_abs:14.6e} "
                f"{max_rel:14.6e} "
                f"{status:>8s}"
            )

        except Exception as exc:
            line = f"{filename}: ERROR - {exc}"

        lines.append(line)

    lines.append("-" * 85)
    lines.append(
        f"OVERALL: {'PASS' if passed_tests == len(BANDS) else 'FAIL'}"
    )
    lines.append(
        f"Passed: {passed_tests}/{len(BANDS)}"
    )
    lines.append(f"Required limit: {LIMIT} %")

    report = "\n".join(lines)

    print(report)

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print("\nReport saved in:", REPORT_FILE)


if __name__ == "__main__":
    main()
