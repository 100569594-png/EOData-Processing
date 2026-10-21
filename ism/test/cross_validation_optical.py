
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


BASE = Path(r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-ISM")
REF_DIR = BASE / "output"
MY_DIR = BASE / "myoutput"
SAVE_DIR = BASE / "ISM_validation" / "optical_phase"

BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
MAX_REL_PERCENT = 1e-3
ZERO_THRESHOLD = 1e-12
ABS_TOL_ZERO = 1e-8


def read_toa(path):
    if not path.is_file():
        raise FileNotFoundError(f"No existe: {path}")
    with Dataset(str(path), "r") as nc:
        if "toa" not in nc.variables:
            raise KeyError(f"No existe la variable 'toa' en {path.name}. "
                           f"Variables: {list(nc.variables.keys())}")
        values = np.ma.asarray(nc.variables["toa"][:])
        data = np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Hay valores no finitos o enmascarados en {path.name}")
    return data


def validate(ref, own):
    if ref.shape != own.shape:
        raise ValueError(f"Dimensiones diferentes: {ref.shape} y {own.shape}")
    if ref.size == 0:
        raise ValueError("El archivo no contiene datos")

    abs_error = np.abs(own - ref)
    max_abs = float(np.max(abs_error))
    normal = np.abs(ref) > ZERO_THRESHOLD

    max_rel = (float(np.max(100.0 * abs_error[normal] / np.abs(ref[normal])))
               if np.any(normal) else 0.0)
    max_zero = float(np.max(abs_error[~normal])) if np.any(~normal) else 0.0
    passed = max_rel < MAX_REL_PERCENT and max_zero <= ABS_TOL_ZERO
    return max_abs, max_rel, "PASS" if passed else "FAIL"


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for band in BANDS:
        name = f"ism_toa_optical_{band}.nc"
        try:
            ref = read_toa(REF_DIR / name)
            own = read_toa(MY_DIR / name)
            max_abs, max_rel, status = validate(ref, own)
            rows.append([name, f"{max_abs:.7e}", f"{max_rel:.7e}", status])
        except Exception as exc:
            rows.append([name, "--", "--", "ERROR"])
            print(f"ERROR al validar {name}: {exc}")

    headers = ["File", "Max abs difference", "Max rel difference (%)", "PASS/FAIL"]

    # Resultados en consola
    print("\nOPTICAL PHASE CROSS-VALIDATION")
    print("-" * 100)
    print(f"{headers[0]:32s} {headers[1]:>20s} {headers[2]:>25s} {headers[3]:>10s}")
    for filename, abs_err, rel_err, result in rows:
        print(f"{filename:32s} {abs_err:>20s} {rel_err:>25s} {result:>10s}")
    print("-" * 100)
    passed_count = sum(row[3] == "PASS" for row in rows)
    print(f"RESULT: {passed_count}/{len(BANDS)} PASS")

    # CSV (para Test Harness)
    csv_path = SAVE_DIR / "optical_phase_cross_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    # TXT (para Test Harness)
    txt_path = SAVE_DIR / "optical_phase_cross_validation.txt"
    with txt_path.open("w", encoding="utf-8") as file:
        file.write("OPTICAL PHASE CROSS-VALIDATION\n")
        file.write("-" * 100 + "\n")
        file.write(f"{headers[0]:32s} {headers[1]:>20s} {headers[2]:>25s} {headers[3]:>10s}\n")
        for filename, abs_err, rel_err, result in rows:
            file.write(f"{filename:32s} {abs_err:>20s} {rel_err:>25s} {result:>10s}\n")
        file.write(f"\nRESULT: {passed_count}/{len(BANDS)} PASS\n")
        file.write(f"Relative error acceptance threshold: {MAX_REL_PERCENT}%\n")
        file.write(f"Absolute tolerance for near-zero reference values: {ABS_TOL_ZERO}\n")

    # Imagen de tabla (para insertar en Word)
    fig, ax = plt.subplots(figsize=(12.2, 3.0))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.22, 0.27, 0.17],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row_i, col_i), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#BBBBBB")
        if row_i == 0:
            cell.set_facecolor("#EEEEEE")
            cell.get_text().set_fontweight("bold")
    fig.text(0.5, 0.04,
             "Optical Phase cross-validation results for the four VNIR bands",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.99))
    image_path = SAVE_DIR / "optical_phase_cross_validation.png"
    fig.savefig(image_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print("CSV:", csv_path)
    print("TXT:", txt_path)
    print("PNG:", image_path)


if __name__ == "__main__":
    main()
