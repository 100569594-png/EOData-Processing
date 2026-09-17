from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# ============================================================
# PROJECT ROOT
# ============================================================

# l1b_test.py está en:
#
# C:\Users\anaes\OneDrive\Documentos\GitHub\
# EOData-Processing\l1b\Test\l1b_test.py
#
# parents[0] -> Test
# parents[1] -> l1b
# parents[2] -> EOData-Processing

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Añadimos la raíz del proyecto al path
# para poder importar config y common
sys.path.insert(0, str(PROJECT_ROOT))


from config.l1bConfig import l1bConfig
from common.src.auxFunc import getIndexBand


# ============================================================
# DIRECTORIES
# ============================================================

# Carpeta de datos de prueba:
#
# C:\Users\anaes\OneDrive\Escritorio\
# EODP_TER_2021\EODP-TS-L1B

TEST_ROOT = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-L1B"
)


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

INPUT_DIR = TEST_ROOT / "input"


# ------------------------------------------------------------
# OUTPUT DE REFERENCIA DEL PROFESOR
# ------------------------------------------------------------

REF_OUTPUT_DIR = TEST_ROOT / "output"


# ------------------------------------------------------------
# MI OUTPUT
# ------------------------------------------------------------

MY_OUTPUT_DIR = TEST_ROOT / "myoutput"


# ------------------------------------------------------------
# FIGURAS
# ------------------------------------------------------------

# Las figuras se guardarán en:
#
# EOData-Processing\l1b\Test\figures

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# INFORME DE CROSS VALIDATION
# ------------------------------------------------------------

VALIDATION_REPORT = (
    Path(__file__).resolve().parent
    / "cross_validation_L1B.txt"
)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

def check_directories():

    print()
    print("================================")
    print("DIRECTORY CHECK")
    print("================================")

    print()
    print("PROJECT_ROOT:")
    print(PROJECT_ROOT)

    print()
    print("TEST_ROOT:")
    print(TEST_ROOT)
    print("Exists:", TEST_ROOT.exists())

    print()
    print("INPUT_DIR:")
    print(INPUT_DIR)
    print("Exists:", INPUT_DIR.exists())

    print()
    print("REF_OUTPUT_DIR:")
    print(REF_OUTPUT_DIR)
    print("Exists:", REF_OUTPUT_DIR.exists())

    print()
    print("MY_OUTPUT_DIR:")
    print(MY_OUTPUT_DIR)
    print("Exists:", MY_OUTPUT_DIR.exists())

    print()
    print("FIGURES_DIR:")
    print(FIGURES_DIR)
    print("Exists:", FIGURES_DIR.exists())

    print()

    # Si falta alguna carpeta importante,
    # paramos el programa.

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"INPUT_DIR does not exist:\n{INPUT_DIR}"
        )

    if not REF_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"REF_OUTPUT_DIR does not exist:\n"
            f"{REF_OUTPUT_DIR}"
        )

    if not MY_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"MY_OUTPUT_DIR does not exist:\n"
            f"{MY_OUTPUT_DIR}"
        )


# ============================================================
# UTILITIES
# ============================================================

def read_toa(filepath):
    """
    Read the 'toa' variable from a NetCDF file.
    """

    filepath = Path(filepath)

    if not filepath.exists():

        raise FileNotFoundError(
            f"NetCDF file not found:\n{filepath}"
        )

    with Dataset(filepath, "r") as dataset:

        if "toa" not in dataset.variables:

            raise KeyError(
                f"Variable 'toa' not found in:\n"
                f"{filepath}"
            )

        toa = np.array(
            dataset.variables["toa"][:],
            dtype=float
        )

    return toa


# ============================================================
# PLOT: EFFECT OF EQUALIZATION
# ============================================================

def plot_equalization():

    # --------------------------------------------------------
    # BAND
    # --------------------------------------------------------

    band = "VNIR-0"

    # Línea ALT central
    alt_line = 50


    print()
    print("================================")
    print(f"EQUALIZATION PLOT - {band}")
    print("================================")


    # ========================================================
    # FILES
    # ========================================================

    # --------------------------------------------------------
    # Radiancia después del ISRF
    # --------------------------------------------------------

    isrf_file = (
        INPUT_DIR
        / f"ism_toa_isrf_{band}.nc"
    )


    # --------------------------------------------------------
    # Input L1B en DN
    # --------------------------------------------------------

    dn_file = (
        INPUT_DIR
        / f"ism_toa_{band}.nc"
    )


    # --------------------------------------------------------
    # Resultado final de MI implementación
    #
    # Equalization + restoration
    # --------------------------------------------------------

    my_l1b_file = (
        MY_OUTPUT_DIR
        / f"l1b_toa_{band}.nc"
    )


    print()
    print("Files used:")

    print()
    print("ISRF:")
    print(isrf_file)

    print()
    print("Input DN:")
    print(dn_file)

    print()
    print("My L1B:")
    print(my_l1b_file)


    # ========================================================
    # READ DATA
    # ========================================================

    toa_isrf = read_toa(
        isrf_file
    )

    toa_dn = read_toa(
        dn_file
    )

    toa_l1b_with_eq = read_toa(
        my_l1b_file
    )


    # ========================================================
    # DIMENSIONS
    # ========================================================

    print()
    print("Array dimensions:")

    print(
        "TOA ISRF:",
        toa_isrf.shape
    )

    print(
        "TOA DN:",
        toa_dn.shape
    )

    print(
        "TOA L1B with EQ:",
        toa_l1b_with_eq.shape
    )


    # Comprobar que ALT=50 existe

    if alt_line >= toa_l1b_with_eq.shape[0]:

        raise IndexError(
            f"ALT line {alt_line} does not exist. "
            f"Number of ALT lines: "
            f"{toa_l1b_with_eq.shape[0]}"
        )


    # ========================================================
    # L1B WITHOUT EQUALIZATION
    # ========================================================

    config = l1bConfig()

    band_index = getIndexBand(
        band
    )

    gain = config.gain[
        band_index
    ]


    print()
    print(
        f"Band index: {band_index}"
    )

    print(
        f"Radiometric gain: {gain}"
    )


    # --------------------------------------------------------
    # SOLO RESTORATION
    #
    # TOA_L1B = TOA_DN * Gain
    # --------------------------------------------------------

    toa_l1b_no_eq = (
        toa_dn
        * gain
    )


    # ========================================================
    # ACT AXIS
    # ========================================================

    act = np.arange(
        toa_l1b_with_eq.shape[1]
    )


    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )


    # --------------------------------------------------------
    # NEGRO:
    # L1B con ecualización
    # --------------------------------------------------------

    plt.plot(
        act,
        toa_l1b_with_eq[
            alt_line,
            :
        ],
        color="black",
        linewidth=1.2,
        label="TOA L1B with eq"
    )


    # --------------------------------------------------------
    # ROJO:
    # L1B SIN ecualización
    # --------------------------------------------------------

    plt.plot(
        act,
        toa_l1b_no_eq[
            alt_line,
            :
        ],
        color="red",
        linewidth=1.2,
        label="TOA L1B no eq"
    )


    # --------------------------------------------------------
    # AZUL:
    # Radiancia después del ISRF
    # --------------------------------------------------------

    plt.plot(
        act,
        toa_isrf[
            alt_line,
            :
        ],
        color="blue",
        linewidth=1.2,
        label="TOA after the ISRF"
    )


    # ========================================================
    # FORMAT
    # ========================================================

    plt.title(
        f"Effect of the Equalization for {band}"
    )

    plt.xlabel(
        "ACT pixel [-]"
    )

    plt.ylabel(
        "TOA [mW/m2/sr]"
    )

    plt.grid(
        True,
        alpha=0.4
    )

    plt.legend(
        loc="upper left"
    )

    plt.tight_layout()


    # ========================================================
    # SAVE FIGURE
    # ========================================================

    figure_path = (
        FIGURES_DIR
        / f"equalization_{band}.png"
    )


    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )


    print()
    print(
        f"Figure saved:"
    )

    print(
        figure_path
    )


    # Mostrar la gráfica

    plt.show()

    plt.close()


# ============================================================
# CROSS-VALIDATION
# ============================================================

def cross_validation():

    # --------------------------------------------------------
    # Máxima diferencia relativa permitida [%]
    # --------------------------------------------------------

    limit = 1e-3


    lines = []


    header = (
        f"{'FILE':35s} "
        f"{'MAX ABS':>14s} "
        f"{'MAX REL %':>14s} "
        f"{'RESULT':>8s}"
    )


    separator = "-" * 76


    lines.append(
        header
    )

    lines.append(
        separator
    )


    overall = True


    # ========================================================
    # ARCHIVOS MYOUTPUT
    # ========================================================

    my_files = sorted(
        MY_OUTPUT_DIR.glob(
            "*.nc"
        )
    )


    print()
    print(
        f"Number of myoutput NetCDF files: "
        f"{len(my_files)}"
    )


    if len(my_files) == 0:

        raise FileNotFoundError(
            f"No NetCDF files found in:\n"
            f"{MY_OUTPUT_DIR}"
        )


    # ========================================================
    # COMPARISON
    # ========================================================

    for my_file in my_files:


        # Archivo equivalente de referencia

        ref_file = (
            REF_OUTPUT_DIR
            / my_file.name
        )


        # ----------------------------------------------------
        # NO REFERENCE
        # ----------------------------------------------------

        if not ref_file.exists():

            lines.append(

                f"{my_file.name:35s} "
                f"{'-':>14s} "
                f"{'-':>14s} "
                f"{'NO REF':>8s}"

            )

            overall = False

            continue


        # ----------------------------------------------------
        # READ ARRAYS
        # ----------------------------------------------------

        ref_toa = read_toa(
            ref_file
        )

        my_toa = read_toa(
            my_file
        )


        # ----------------------------------------------------
        # CHECK SHAPE
        # ----------------------------------------------------

        if ref_toa.shape != my_toa.shape:

            lines.append(

                f"{my_file.name:35s} "
                f"{'-':>14s} "
                f"{'-':>14s} "
                f"{'SHAPE':>8s}"

            )

            overall = False

            continue


        # ----------------------------------------------------
        # ABSOLUTE DIFFERENCE
        # ----------------------------------------------------

        diff = np.abs(
            my_toa
            - ref_toa
        )


        max_abs = np.max(
            diff
        )


        # ----------------------------------------------------
        # RELATIVE DIFFERENCE
        # ----------------------------------------------------

        # Solo donde la referencia no es cero

        mask = (
            np.abs(ref_toa)
            > 1e-15
        )


        if np.any(mask):

            relative_difference = (

                diff[mask]
                / np.abs(
                    ref_toa[mask]
                )
                * 100.0

            )


            max_rel = np.max(
                relative_difference
            )


        else:

            max_rel = 0.0


        # ----------------------------------------------------
        # PASS / FAIL
        # ----------------------------------------------------

        passed = (
            max_rel
            < limit
        )


        overall &= passed


        # ----------------------------------------------------
        # REPORT LINE
        # ----------------------------------------------------

        lines.append(

            f"{my_file.name:35s} "
            f"{max_abs:14.6e} "
            f"{max_rel:14.6e} "
            f"{'PASS' if passed else 'FAIL':>8s}"

        )


    # ========================================================
    # REPORT END
    # ========================================================

    lines.append(
        separator
    )


    lines.append(

        f"OVERALL: "
        f"{'PASS' if overall else 'FAIL'}"

    )


    lines.append(

        f"Required limit: "
        f"{limit} %"

    )


    # ========================================================
    # PRINT REPORT
    # ========================================================

    report = "\n".join(
        lines
    )


    print()
    print("==============================")
    print("L1B CROSS-VALIDATION")
    print("==============================")
    print()

    print(
        report
    )


    # ========================================================
    # SAVE REPORT
    # ========================================================

    VALIDATION_REPORT.write_text(
        report,
        encoding="utf-8"
    )


    print()
    print(
        "Validation report saved:"
    )

    print(
        VALIDATION_REPORT
    )


    return overall


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("================================")
    print("EODP L1B TEST")
    print("================================")
    print()


    # --------------------------------------------------------
    # Comprobar rutas
    # --------------------------------------------------------

    check_directories()


    # --------------------------------------------------------
    # Generar gráfica de equalization
    # --------------------------------------------------------

    plot_equalization()


    # --------------------------------------------------------
    # Cross-validation contra output
    # --------------------------------------------------------

    result = cross_validation()


    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    if result:

        print()
        print(
            "L1B TEST RESULT: PASS"
        )

    else:

        print()
        print(
            "L1B TEST RESULT: FAIL"
        )