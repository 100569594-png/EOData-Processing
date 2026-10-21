from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from common.io.writeToa import readToa
from common.src.auxFunc import getIndexBand
from config.l1bConfig import l1bConfig

# ---------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------
L1B_INPUT_DIR = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-L1B\input"
)

L1B_OUTPUT_DIR = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-L1B\myoutput"
)

ISM_OUTPUT_DIR = Path(
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-ISM\myoutput"
)

SAVE_DIR = Path(__file__).resolve().parent / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------
bands = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
cfg = l1bConfig()

# ---------------------------------------------------------------------
# GENERACIÓN DE FIGURAS
# ---------------------------------------------------------------------
for band in bands:
    # 1) L1B with equalization (negro)
    # salida final L1B, ya en radiancias
    toa_l1b_eq = readToa(str(L1B_OUTPUT_DIR), f"l1b_toa_{band}.nc")

    # 2) L1B without equalization (rojo)
    # entrada a L1B (salida ISM final en DN), convertida a radiancias solo con gain
    toa_dn_no_eq = readToa(str(L1B_INPUT_DIR), f"ism_toa_{band}.nc")
    gain = cfg.gain[getIndexBand(band)]
    toa_l1b_no_eq = toa_dn_no_eq * gain

    # 3) TOA after ISRF (azul)
    toa_after_isrf = readToa(str(ISM_OUTPUT_DIR), f"ism_toa_isrf_{band}.nc")

    # Slice central en ALT
    alt = toa_l1b_eq.shape[0] // 2
    x = np.arange(toa_l1b_eq.shape[1])

    plt.figure(figsize=(10, 6))

    plt.plot(
        x, toa_l1b_eq[alt, :],
        color="black", linewidth=2,
        label="L1B with equalization"
    )

    plt.plot(
        x, toa_l1b_no_eq[alt, :],
        color="red", linewidth=1.5,
        label="L1B without equalization"
    )

    plt.plot(
        x, toa_after_isrf[alt, :],
        color="blue", linewidth=1.5,
        label="TOA after ISRF"
    )

    plt.title(f"Effect of equalization — {band} (ALT pixel {alt})")
    plt.xlabel("ACT pixel [-]")
    plt.ylabel("TOA [mW/m²/sr]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(SAVE_DIR / f"equalization_{band}.png", dpi=200)
    plt.close()

print("Figuras guardadas en:", SAVE_DIR)