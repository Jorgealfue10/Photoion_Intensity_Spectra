import numpy as np 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
        "matrix_coh_real","matrix_coh_imag","matrix_coh_abs","matrix_coh_sum_abs",
        "Delta_v","Delta_J","Delta_Omega"
    ]

    for col in real_cols:
        df[col] = df[col].astype(float)
    
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

def build_conv_df(df,E_grid,energy_col="Delta_eV",intensity_col="matrix_coh_sum_abs",
    vi_col="v_i",sigma=0.025,filters=None,normalize=False):
    
    data = df.copy()

    # ========================================================
    # Apply filters
    # ========================================================

    if filters is not None:
        mask = np.ones(len(data), dtype=bool)

        for col, value in filters.items():

            if callable(value):
                mask &= value(data[col]).to_numpy()

            elif isinstance(value, (list, tuple, set, np.ndarray)):
                mask &= data[col].isin(value).to_numpy()

            else:
                mask &= (data[col] == value).to_numpy()

        data = data.loc[mask].copy()

    # ========================================================
    # Keep finite lines inside E_grid
    # ========================================================

    E_lines_all = data[energy_col].to_numpy(dtype=float)
    I_lines_all = data[intensity_col].to_numpy(dtype=float)

    keep = (
        np.isfinite(E_lines_all)
        & np.isfinite(I_lines_all)
        & (E_lines_all >= E_grid.min())
        & (E_lines_all <= E_grid.max())
    )

    data = data.loc[keep].copy()

    E_lines_all = data[energy_col].to_numpy(dtype=float)
    I_lines_all = data[intensity_col].to_numpy(dtype=float)

    # ========================================================
    # Normalize input intensities if desired
    # ========================================================

    if normalize:
        max_I = np.max(np.abs(I_lines_all)) if len(I_lines_all) > 0 else 0.0

        if max_I > 0:
            I_lines_all = I_lines_all / max_I
            data[intensity_col + "_normalized"] = I_lines_all

    data["_I_conv"] = I_lines_all

    # ========================================================
    # Total convolution
    # ========================================================

    spec = convolve_lines(
        E_grid=E_grid,
        E_lines=E_lines_all,
        I_lines=I_lines_all,
        sigma=sigma,
    )

    spec_max = np.max(spec) if len(spec) > 0 else 0.0
    spec_area = np.trapz(spec, E_grid) if len(spec) > 0 else 0.0

    # ========================================================
    # Convolution by v_i
    # ========================================================

    vi_values = sorted(data[vi_col].unique())

    by_vi = {}
    sticks_by_vi = {}
    max_by_vi = {}
    area_by_vi = {}

    for vi in vi_values:

        sub = data[data[vi_col] == vi].copy()

        E_lines_vi = sub[energy_col].to_numpy(dtype=float)
        I_lines_vi = sub["_I_conv"].to_numpy(dtype=float)

        spec_vi = convolve_lines(
            E_grid=E_grid,
            E_lines=E_lines_vi,
            I_lines=I_lines_vi,
            sigma=sigma,
        )

        by_vi[vi] = spec_vi
        sticks_by_vi[vi] = sub

        max_by_vi[vi] = np.max(spec_vi) if len(spec_vi) > 0 else 0.0
        area_by_vi[vi] = np.trapz(spec_vi, E_grid) if len(spec_vi) > 0 else 0.0

    # ========================================================
    # Return
    # ========================================================

    return {
        "spec": spec,"spec_max": spec_max,"spec_area": spec_area,
        "by_vi": by_vi,"vi_values": vi_values,"sticks_by_vi": sticks_by_vi,
        "max_by_vi": max_by_vi,"area_by_vi": area_by_vi,
        "sticks": data,"n_lines": len(data),
        "energy_col": energy_col,"intensity_col": intensity_col,
        "vi_col": vi_col,"sigma": sigma,"filters": filters,"normalize": normalize,
    }

def get_vib_energies_df(df,vi_col="v_i",Ei_col="Ei_eV"):

    Evib = (df.groupby(vi_col)[Ei_col].min().to_dict())

    E0 = min(Evib.values())

    return Evib, E0

def get_rot_energies_df(df,vi_col="v_i",Ji_col="J_i",Omi_col="Omega_i",Ei_col="Ei_eV"):

    Evib,E0 = get_vib_energies_df(df,vi_col,Ei_col)

    Erot = {}

    grouped = df.groupby([vi_col,Ji_col,Omi_col])[Ei_col].min()

    for (v,J,Om), E in grouped.items():
        v = int(v)
        key = (J,Om)

        if v not in Erot:
            Erot[v] = {}

        Erot[v][key] = E
    
    return Erot,Evib,E0

def thermal_populations_df(df,Tvib,Trot,vi_col="v_i",Ji_col="J_i",Omi_col="Omega_i",Ei_col="Ei_eV"):
    kB_eV = 8.61733326145e-5 #eV/K

    Erot,Evib,E0 = get_rot_energies_df(df,vi_col,Ji_col,Omi_col,Ei_col)

    qvib = {}
    for v, E_v in Evib.items():
        qvib[v] = np.exp(-(E_v-E0)/(kB_eV*Tvib))
    
    Qvib = sum(qvib.values())
    
    Pvib = {
        v: qvib[v] / Qvib
        for v in qvib
    }

    qrot = {} ; Qrot = {} ; Prot = {}

    for v in Erot: 
        qrot[v] = {}
        E_v_min = Evib[v]
        for (J,Om), E in Erot[v].items():
            E_rot = E-E_v_min

            weight = (2.0*J+1.0)*np.exp(-E_rot/(kB_eV*Trot))

            qrot[v][(J,Om)] = weight

        Qrot[v] = sum(qrot[v].values())
        Prot[v] = {
            key:val/Qrot[v]
            for key,val in qrot[v].items()
        }
    
    pop_data = {
        "Tvib": Tvib, "Trot": Trot, 
        "Pvib": Pvib, "Prot": Prot,
        "Qvib": Qvib, "Qrot": Qrot,
        "Evib": Evib, "Erot": Erot,
        "E0": E0
    }

    return Pvib, Prot, pop_data

def sticks_by_temp_df(df,Tvib_values,Trot_values,Ei_col="Ei_eV",
            vi_col="v_i",Ji_col="J_i",Omi_col="Omega_i",intensity_cols=("matrix_coh_sum_abs",)):

    sticks_temp = {} ; pop_temp = {}

    for Tvib in Tvib_values: 
        for Trot in Trot_values:

            Pvib,Prot,pop_data = thermal_populations_df(df,Tvib,Trot,vi_col,Ji_col,Omi_col,Ei_col)

            data = df.copy()
            pop_factor = []

            for v,J,Om in zip(data[vi_col],data[Ji_col],data[Omi_col]):
                v = int(v)
                key = (J,Om)

                p = Pvib[v] * Prot[v][key]
                pop_factor.append(p)
            
            pop_factor = np.asarray(pop_factor,dtype=float)
            data["population_factor"] = pop_factor

            for col in intensity_cols: 
                data[col+"_T"] = data[col].to_numpy(dtype=float)*pop_factor
            
            key = (Tvib,Trot)

            sticks_temp[key] = data ; pop_temp[key] = pop_data
    
    return sticks_temp, pop_temp

def safe_filename_label(label):
    return (
        label
        .replace(" -> ", "_to_")
        .replace(" ", "")
        .replace("+", "p")
        .replace("/", "")
    )


def write_conv_file_one_transition_temperature(output_file,conv,E_grid,
                                        transition_label=None,Tvib=None,Trot=None):
    with open(output_file, "w") as f:

        if transition_label is not None:
            f.write(f"# Transition: {transition_label}\n")

        if Tvib is not None and Trot is not None:
            f.write(f"# Tvib = {Tvib} K ; Trot = {Trot} K\n")

        vi_values = conv["vi_values"]

        header = ["E_grid", "Conv_total"]

        for vi in vi_values:
            header.append(f"Conv_v{vi}")

        f.write("# " + " ".join(header) + "\n")

        cols = [E_grid, conv["spec"]]

        for vi in vi_values:
            cols.append(conv["by_vi"][vi])

        arr = np.column_stack(cols)

        np.savetxt(f,arr,fmt="%.10e")

base_dir = Path("/home/jorgebdelafuente/Doctorado/Photoion/DUO/PHPHM")

transition_cols = [
    "v_i","J_i","Omega_i","Sigma_i","Lambda_i","parity_i","index_i",
    "v_f","J_f","Omega_f","Sigma_f","Lambda_f","parity_f","index_f",
    "Ei_eV","Ef_ev","Delta_eV",
    "matrix_coh_real","matrix_coh_imag","matrix_coh_abs","matrix_coh_sum_abs",
    "Delta_v","Delta_J","Delta_Omega"
]

transition_files = {
    "PHGS -> PHMGS": base_dir / "test.out"
}

transitions_df = {
    label: read_transition_output_df(path,skiprows=1)
    for label,path in transition_files.items()
}

df = transitions_df["PHGS -> PHMGS"]
E_min = df["Delta_eV"].min()
E_max = df["Delta_eV"].max()

E_grid = np.linspace(E_min, E_max, 6000)

Tvib_values = [5000]
Trot_values = [100]

transition_label = "PHGS -> PHMGS"
df = transitions_df[transition_label]

sticks_temp, pop_temp = sticks_by_temp_df(df,Tvib_values=Tvib_values,Trot_values=Trot_values,
                                        Ei_col="Ei_eV",vi_col="v_i",Ji_col="J_i",Omi_col="Omega_i",
                                        intensity_cols=("matrix_coh_sum_abs",))

conv_temp = {}

for (Tvib, Trot), df_T in sticks_temp.items():

    conv_T = build_conv_df(df_T,E_grid,energy_col="Delta_eV",
        intensity_col="matrix_coh_sum_abs_T",vi_col="v_i",sigma=0.015,
        filters={
            "Delta_J": lambda x: np.abs(x) <= 3.5
        },normalize=False)

    conv_temp[(Tvib, Trot)] = conv_T

    safe_label = safe_filename_label(transition_label)

    output_file = f"{safe_label}_Tvib{Tvib}_Trot{Trot}.dat"

    write_conv_file_one_transition_temperature(output_file=output_file,conv=conv_T,E_grid=E_grid,
        transition_label=transition_label,Tvib=Tvib,Trot=Trot)