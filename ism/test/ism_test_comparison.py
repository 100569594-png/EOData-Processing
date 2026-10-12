
from pathlib import Path
import csv
import logging

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# ==========================================================
# 1. DIRECTORIOS
# ==========================================================

BASE = Path(
    r"C:\Users\anaes\OneDrive\Escritorio"
    r"\EODP_TER_2021\EODP-TS-ISM"
)

REF_DIR = BASE / "output"
MY_DIR = BASE / "myoutput"

TEST_DIR = BASE / "ISM_validation"
FIG_DIR = TEST_DIR / "figures"

TEST_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = TEST_DIR / "ism_validation_summary.csv"
LOG_FILE = TEST_DIR / "ism_validation.log"

TOLERANCE_PERCENT = 0.001
ZERO_THRESHOLD = 1e-12

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

# ==========================================================
# 2. LECTURA DE NETCDF
# ==========================================================

def read_netcdf(path):

    with Dataset(str(path), "r") as nc:

        # Detectar el nombre a partir del archivo:
        # Hdiff_VNIR-0.nc -> Hdiff
        # Hsys_VNIR-0.nc -> Hsys

        prefix = path.stem.split("_VNIR-")[0]

        variable_names = [
            prefix,
            "toa",
            "mat"
        ]

        # Añadir otras posibles variables MTF
        variable_names += [
            "Hdiff",
            "Hdefoc",
            "Hwfe",
            "Hdet",
            "Hsmear",
            "Hmotion",
            "Hsys"
        ]

        for name in variable_names:

            if name in nc.variables:

                variable = nc.variables[name]

                if variable.ndim != 2:
                    continue

                data = np.ma.filled(
                    variable[:],
                    np.nan
                )

                return np.asarray(data, dtype=float)

        raise KeyError(
            f"No se encontró una matriz 2D en {path.name}. "
            f"Variables: {list(nc.variables.keys())}"
        )


# ==========================================================
# 3. MÉTRICAS
# ==========================================================

def compare_arrays(reference, candidate):

    if reference.shape != candidate.shape:
        raise ValueError(
            f"Dimensiones diferentes: "
            f"{reference.shape} vs {candidate.shape}"
        )

    if not (
        np.all(np.isfinite(reference))
        and np.all(np.isfinite(candidate))
    ):
        raise ValueError(
            "Se encontraron valores NaN o infinitos."
        )

    difference = candidate - reference
    absolute = np.abs(difference)

    rmse = np.sqrt(np.mean(difference ** 2))
    mae = np.mean(absolute)
    max_abs = np.max(absolute)

    # Comparación relativa solo para referencias
    # suficientemente alejadas de cero.
    valid = np.abs(reference) > ZERO_THRESHOLD

    if np.any(valid):
        relative = (
            absolute[valid] / np.abs(reference[valid])
        ) * 100

        max_rel = float(np.max(relative))
    else:
        max_rel = np.nan

    if np.any(~valid):
        max_zero_error = float(
            np.max(absolute[~valid])
        )
    else:
        max_zero_error = 0.0

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Max_absolute_error": float(max_abs),
        "Max_relative_error_percent": max_rel,
        "Max_error_near_zero": max_zero_error
    }


# ==========================================================
# 4. FIGURAS COMPARATIVAS
# ==========================================================

def plot_comparison(reference, candidate, filename):

    alt = reference.shape[0] // 2
    x = np.arange(reference.shape[1])

    is_mtf = filename.startswith("H")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        reference[alt, :],
        label="Reference",
        linewidth=2
    )

    ax.plot(
        x,
        candidate[alt, :],
        label="My output",
        linestyle="--"
    )

    ax.set_xlabel("ACT pixels")

    if is_mtf:
        ax.set_ylabel("MTF [-]")
    elif filename.startswith("ism_toa_") and (
        "_optical_" in filename
        or "_isrf_" in filename
    ):
        ax.set_ylabel("TOA (see file units)")
    else:
        ax.set_ylabel("TOA (see processing stage)")

    ax.set_title(f"{filename} | ALT = {alt}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    destination = (
        FIG_DIR / f"{Path(filename).stem}_comparison.png"
    )

    try:
        fig.savefig(
            str(destination),
            dpi=180
        )
        return "SAVED"
    finally:
        plt.close(fig)


# ==========================================================
# 5. VALIDACIÓN
# ==========================================================

def main():

    results = []

    if not REF_DIR.is_dir():
        raise FileNotFoundError(REF_DIR)

    if not MY_DIR.is_dir():
        raise FileNotFoundError(MY_DIR)

    reference_files = sorted(REF_DIR.glob("*.nc"))

    print("=" * 65)
    print("ISM CROSS-VALIDATION")
    print("=" * 65)

    for reference_file in reference_files:

        filename = reference_file.name
        candidate_file = MY_DIR / filename

        if not candidate_file.exists():
            print(f"MISSING: {filename}")
            logging.warning("MISSING: %s", filename)

            results.append({
                "File": filename,
                "RMSE": "",
                "MAE": "",
                "Max_absolute_error": "",
                "Max_relative_error_percent": "",
                "Max_error_near_zero": "",
                "Status": "MISSING",
                "Figure": "NOT GENERATED"
            })
            continue

        try:
            reference = read_netcdf(reference_file)
            candidate = read_netcdf(candidate_file)

            metrics = compare_arrays(
                reference,
                candidate
            )

            max_rel = metrics[
                "Max_relative_error_percent"
            ]

            zero_error = metrics[
                "Max_error_near_zero"
            ]

            # Criterio relativo y comprobación de
            # diferencias absolutas cerca del cero.
            if np.isnan(max_rel):
                if zero_error == 0:
                    status = "PASS"
                else:
                    status = "REVIEW"

            elif (
                max_rel < TOLERANCE_PERCENT
                and zero_error == 0
            ):
                status = "PASS"

            else:
                status = "FAIL / REVIEW"

            # La figura es evidencia auxiliar:
            # si falla, no se pierde el resultado numérico.
            try:
                figure_status = plot_comparison(
                    reference,
                    candidate,
                    filename
                )
            except Exception as plot_error:
                figure_status = "FIGURE ERROR"
                logging.exception(
                    "Error guardando figura %s: %s",
                    filename,
                    plot_error
                )

            results.append({
                "File": filename,
                **metrics,
                "Status": status,
                "Figure": figure_status
            })

            logging.info(
                "%s | %s | %s",
                filename,
                status,
                metrics
            )

            print(
                f"{filename}: {status} | "
                f"Max rel: {max_rel:.8g}% | "
                f"Figure: {figure_status}"
            )

        except Exception as error:

            logging.exception(
                "Error comparando %s",
                filename
            )

            results.append({
                "File": filename,
                "RMSE": "",
                "MAE": "",
                "Max_absolute_error": "",
                "Max_relative_error_percent": "",
                "Max_error_near_zero": "",
                "Status": "ERROR",
                "Figure": "NOT GENERATED"
            })

            print(f"ERROR: {filename}: {error}")

    # ======================================================
    # 6. GUARDAR CSV
    # ======================================================

    fieldnames = [
        "File",
        "RMSE",
        "MAE",
        "Max_absolute_error",
        "Max_relative_error_percent",
        "Max_error_near_zero",
        "Status",
        "Figure"
    ]

    with CSV_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # ======================================================
    # 7. RESUMEN
    # ======================================================

    passed = sum(
        row["Status"] == "PASS"
        for row in results
    )

    failed = sum(
        row["Status"] == "FAIL / REVIEW"
        for row in results
    )

    errors = sum(
        row["Status"] == "ERROR"
        for row in results
    )

    missing = sum(
        row["Status"] == "MISSING"
        for row in results
    )

    print()
    print("=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Total reference files : {len(results)}")
    print(f"PASS                  : {passed}")
    print(f"FAIL / REVIEW         : {failed}")
    print(f"ERROR                 : {errors}")
    print(f"MISSING               : {missing}")

    print()
    print("CSV:", CSV_FILE)
    print("LOG:", LOG_FILE)
    print("FIGURES:", FIG_DIR)


if __name__ == "__main__":
    main()
