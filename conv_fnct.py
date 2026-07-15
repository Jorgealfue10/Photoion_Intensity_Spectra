import numpy as np 
import pandas as pd
from pathlib import Path

base_dir = Path("/home/jorgebdelafuente/Doctorado/Photoion/DUO/PHPHM")

transition_cols = [
    "v_i","J_i","Omega_i","Sigma_i","Lambda_i","parity_i","index_i",
    "v_f","J_f","Omega_f","Sigma_f","Lambda_f","parity_f","index_f",
    "Ei_eV","Ef_ev","Delta_eV",
    "matrix_coh_real","matrix_coh_imag","matrix_coh_sum_abs","matrix_coh_sum_abs",
    "Delta_v","Delta_J","Delta_Omega"
]

transition_files = {
    "PHGS -> PHMGS": base_dir / "test.out"
}

def read_transition_output_df(filepath,skiprows=1):

    arr = np.loadtxt(filepath,skiprows=skiprows)

    df = pd.DataFrame(arr,columns=transition_cols)

    int_cols = [
        "v_i","index_i","v_f","index_f","Delta_v"
    ]

    for col in int_cols:
        df[col] = np.rint(df[col]).astype(int)

    real_cols = [
        "J_i","Omega_i","Sigma_i","Lambda_i","parity_i",
        "J_f","Omega_f","Sigma_f","Lambda_f","parity_f",
        "Ei_eV","Ef_ev","Delta_eV",
        "matrix_coh_real","matrix_coh_imag","matrix_coh_sum_abs","matrix_coh_sum_abs",
        "Delta_v","Delta_J","Delta_Omega"
    ]

    for col in real_cols:
        df[col] = np.float(df[col])
    
    df.attrs["filepath"] = str(filepath)
    df.attrs["filename"] = Path(filepath).name

    return df

def gaussian(E,E0,sigma):
    return np.exp(-(E-E0)**2/(2.0*sigma**2))

def convolve_lines(E_grid,E_lines,I_lines,sigma=0.025):
    spec = np.zeros_like(E_grid,dtype=float)

    for E0,I0 in zip(E_lines,I_lines):
        spec += I0 * gaussian(E_grid,E0,sigma)

    return spec

transitions_df = {
    label: read_transition_output_df(path,skiprows=1)
    for label,path in transition_files.items()
}

