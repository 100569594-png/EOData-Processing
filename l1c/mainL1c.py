# MAIN FUNCTION TO CALL THE L1C MODULE

from l1c.src.l1c import l1c


auxdir = r"C:\Users\anaes\OneDrive\Documentos\GitHub\EOData-Processing\auxiliary"


# GM output + MY L1B output
indir = (
    r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-E2E\gm_out"
    r",C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-E2E\myoutput_l1b"
)


# MY L1C output
outdir = r"C:\Users\anaes\OneDrive\Escritorio\EODP_TER_2021\EODP-TS-E2E\myoutput_l1c"


myL1c = l1c(
    auxdir,
    indir,
    outdir
)

myL1c.processModule()