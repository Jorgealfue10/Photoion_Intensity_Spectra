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
        df[col] = df[col].astype(float)
    
    df.attrs["filepath"] = str(filepath)
    df.attrs["filename"] = Path(filepath).name

    return df

def vibpart(vib,Erv,Tvib):
    kb = 3.166811563e-6 #Eh/K
    Eh_to_eV = 27.2114 

    vi = vib[:].astype(int)
    Ei = Erv[:]

    v_un = np.unique(vi)

    Ev = {}
    for v in v_un:
        mask_v = vi == v
        Ev[v] = np.min(Ei[mask_v])
    
    Pvib = {} ; Qvib = None

    E0 = min(Ev.values())
    qpv = {
        v: np.exp(-((Ev[v]-E0)/Eh_to_eV)/(kb*Tvib))
        for v in v_un
    }

    Qvib = sum(qpv.values())
    Pvib = {
        v: qpv[v]/Qvib
        for v in v_un
    }

    return Pvib,Qvib,E0,Ev

def rotpart(v,rot,Om,Erv,Trot):
    kb = 3.166811563e-6 #Eh/K
    Eh_to_eV = 27.2114 

    vi = v[:].astype(int) ; Ji = rot[:] ; Omi = Om[:]
    Ei = Erv[:]
    v_un = np.unique(vi)

    Qrot = {} ; Prot = {}

    Ev = {}
    for v in v_un: 
        mask_v = vi == v
        Ev[v] = np.min(Ei[mask_v])

    for v in v_un:
        mask_v = vi == v
        J_v = Ji[mask_v] ; Om_v = Omi[mask_v] ; E_v = Ei[mask_v]

        levels = {}

        for J,Om,E in zip(J_v,Om_v,E_v):
            key = (J,Om)
            if key not in levels:
                levels[key] = E
            else:
                levels[key] = min(levels[key],E)

        Ev_min = E[v]
        Q = 0.0 ; tmp = {}
        for (J,Om), E in levels.items():
            Erot = (E - Ev_min) / Eh_to_eV
            w = (2*J+1)*np.exp(-Erot/(kb*Trot))
            tmp[(J,Om)] = w 
            Q += w
        
        Qrot[v] = Q
        Prot[v] = {
            key: val/Q
            for key,val in tmp.items()
        }

    return Prot,Qrot

def gaussian(E,E0,sigma):
    return np.exp(-(E-E0)**2/(2.0*sigma**2))

def convolve_lines(E_grid,E_lines,I_lines,sigma=0.025):
    spec = np.zeros_like(E_grid,dtype=float)

    for E0,I0 in zip(E_lines,I_lines):
        spec += I0 * gaussian(E_grid,E0,sigma)

    return spec

def build_conv_df(df, E_grid, energy_col="Delta_eV",intensity_col="matrix_coh_sum_abs",sigma=0.025,filters=None,normalize=False):
    data = df.copy()

    if filters is not None: 
        mask = np.ones(len(data),dtype=bool)

        for col, value in filters.items():
            if callable(value):
                mask &= value(data[col]).to_numpy()
            elif isinstance(value,(list,tuple,set,np.ndarray)):
                mask &= data[col].isin(value).to_numpy()
            else:
                mask &= (data[col] == value).to_numpy()
            
        data = data.loc[mask].copy()

    E_lines = data[energy_col].to_numpy(dtype=float)
    I_lines = data[intensity_col].to_numpy(dtype=float)

    keep = (
        np.isfinite(E_lines) & np.isfinite(I_lines)
        & (E_lines >= E_grid.min()) & (E_lines <= E_grid.max())
    )

    E_lines = E_lines[keep] ; I_lines = I_lines[keep]

    sticks = data.iloc[np.where(keep)[0]].copy()

    spec = convolve_lines(E_grid=E_grid,E_lines=E_lines,I_lines=I_lines,sigma=sigma)

    return spec,sticks

transitions_df = {
    label: read_transition_output_df(path,skiprows=1)
    for label,path in transition_files.items()
}

# print(transitions_df)