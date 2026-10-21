# ISM: tabla MTF cerca de Nyquist (consola + CSV + PNG).
# Ejecutar desde PyCharm en ism/test/.
# Lee solo los NetCDF de EODP-TS-ISM/myoutput.
# No modifica ningun resultado de ISM.

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# 1. CONFIGURACION DE CARPETAS
MY_DIR = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-ISM\myoutput"
)

# Los productos para el informe se guardan junto a este script.
SAVE_DIR = Path(__file__).resolve().parent
CSV_FILE = SAVE_DIR / "mtf_nyquist_summary.csv"
PNG_FILE = SAVE_DIR / "mtf_nyquist_table.png"

BANDS = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")
COMPONENTS = ("Hdiff", "Hdefoc", "Hwfe", "Hdet", "Hsmear", "Hmotion")
NAMES = {
    "Hdiff": "Diffraction",
    "Hdefoc": "Defocus",
    "Hwfe": "WFE aberrations",
    "Hdet": "Detector",
    "Hsmear": "Smearing",
    "Hmotion": "Motion",
}


def read_matrix(component, band):
    """Lee una matriz 2D de su archivo NetCDF."""
    path = MY_DIR / f"{component}_{band}.nc"
    if not path.is_file():
        raise FileNotFoundError(f"Falta el archivo: {path}")

    with Dataset(str(path), "r") as nc:
        if component not in nc.variables:
            raise KeyError(
                f"Variable {component!r} no encontrada en {path.name}. "
                f"Disponibles: {list(nc.variables)}"
            )
        data = np.asarray(
            np.ma.filled(nc.variables[component][:], np.nan), dtype=float
        )

    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError(f"Datos invalidos en {path.name}: {data.shape}")
    return data


def sample_at_nyquist(matrix, direction):
    """Muestra el ultimo punto positivo de ACT o ALT, en su corte central.

    Para 100 x 150: ACT=0.493333 y ALT=0.490000, inmediatamente
    por debajo de la frecuencia normalizada de Nyquist (0.5).
    """
    n_alt, n_act = matrix.shape
    if direction == "ACT":
        return float(matrix[n_alt // 2, -1]), 0.5 - 1 / n_act
    if direction == "ALT":
        return float(matrix[-1, n_act // 2]), 0.5 - 1 / n_alt
    raise ValueError(f"Direccion incorrecta: {direction}")


def calculate_rows():
    """Obtiene una fila por banda y direccion."""
    rows = []
    for band in BANDS:
        matrices = {
            name: read_matrix(name, band)
            for name in ("Hsys", *COMPONENTS)
        }

        shapes = {matrix.shape for matrix in matrices.values()}
        if len(shapes) != 1:
            raise ValueError(f"Dimensiones diferentes para {band}: {shapes}")

        for direction in ("ACT", "ALT"):
            hsys_value, freq = sample_at_nyquist(matrices["Hsys"], direction)
            contributions = {
                name: sample_at_nyquist(matrices[name], direction)[0]
                for name in COMPONENTS
            }
            most_attenuating = min(contributions, key=contributions.get)

            rows.append({
                "Band": band,
                "Direction": direction,
                "Normalized spatial frequency": f"{freq:.6f}",
                "System MTF": f"{hsys_value:.6f}",
                "Most attenuating MTF": NAMES[most_attenuating],
                "MTF value": f"{contributions[most_attenuating]:.6f}",
            })
    return rows


def save_csv(rows):
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_png(rows):
    """Tabla de 6 columnas legible para insertar directamente en Word."""
    headers = [
        "Band", "Direction", "Normalized\nspatial frequency",
        "System MTF", "Most attenuating\nMTF", "MTF value",
    ]
    cells = [[str(value) for value in row.values()] for row in rows]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    ax.set_title(
        "System MTF and largest individual attenuation near Nyquist",
        fontsize=14, fontweight="bold", pad=20,
    )
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.105, 0.11, 0.205, 0.14, 0.27, 0.17],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)

    # Presentacion neutra y legible para un informe academico.
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f8f8f8")

    fig.savefig(PNG_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if not MY_DIR.is_dir():
        raise FileNotFoundError(f"No existe la carpeta myoutput: {MY_DIR}")

    rows = calculate_rows()
    save_csv(rows)
    save_png(rows)

    print("\nSYSTEM MTF NEAR NYQUIST - OWN RESULTS\n")
    for row in rows:
        print(" | ".join(row.values()))
    print(f"\nCSV guardado en: {CSV_FILE}")
    print(f"Imagen guardada en: {PNG_FILE}")
    print("Total: 8 filas (4 bandas x 2 direcciones).")


if __name__ == "__main__":
    main()
