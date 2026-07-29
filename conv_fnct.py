import numpy as np 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

#---------------------------------------------------------------------------------------
# Reading photoion sticks spectra
#---------------------------------------------------------------------------------------
def read_transition_output_df(filepath,skiprows=1,transition_cols=None):
    """
    Params:
    filepath: str or Path - Output file containing state-to-strate intensities
    skiprows: int - Skiprows if there is any header in filepath
    transition_cols: list - Column names and attr

    Returns:
    df: pandas df - Contains all state-to-state transitions' info
    """

    if transition_cols is None:
        transition_cols = [
            "v_i", "J_i", "Omega_i", "Sigma_i", "Lambda_i", "parity_i", "index_i",
            "v_f", "J_f", "Omega_f", "Sigma_f", "Lambda_f", "parity_f", "index_f",
            "Ei_eV", "Ef_ev", "Delta_eV",
            "matrix_coh_real", "matrix_coh_imag", "matrix_coh_abs", "matrix_coh_sum_abs",
            "Delta_v", "Delta_J", "Delta_Omega",
        ]

    arr = np.loadtxt(filepath, skiprows=skiprows)

    if arr.ndim == 1:
        arr = arr[None, :]

    df = pd.DataFrame(arr, columns=transition_cols)

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

#---------------------------------------------------------------------------------------
# Convolution functions
#---------------------------------------------------------------------------------------

# Stick-centered gaussian function
def gaussian(E,E0,sigma):
    return np.exp(-(E-E0)**2/(2.0*sigma**2))

# Convolve lines in a fixed grid
def convolve_lines(E_grid,E_lines,I_lines,sigma=0.025):
    """
    Params: 
    E_grid: list or np.ndarray - Constant energy grid for every transition
    E_lines: df, list or np.ndarray - State-to-state transitions energies
    I_lines: df, list or np.ndarray - State-to-state transitions intensities
    sigma: float - gaussian half-width

    Returns:
    spec: np.ndarray - Convolve transition in E_grid energies
    """
    spec = np.zeros_like(E_grid,dtype=float)

    for E0,I0 in zip(E_lines,I_lines):
        spec += I0 * gaussian(E_grid,E0,sigma)

    return spec

# Convolution of all line transitions and sum over the whole E_grid
def build_conv_df(df,E_grid,energy_col="Delta_eV",intensity_col="matrix_coh_sum_abs",
    vi_col="v_i",sigma=0.025,filters=None,normalize=False):
    """
    Params: 
    df: pandas df - transitions dataframe
    E_grid: list or np.ndarray - constant energy grid for every transition
    energy_col: str - energy column in df
    intensity_col: str - intensity column in df
    vi_col: str - initial vibrational level column in df 
    sigma: float - convolution gaussian half-width 
    filters: mask - considering  quantum number parameters
    normalize: boolean - normalization of spectrum

    Returns:
    dict - contains total convoluted spectra and vi-level convolution
    """

    data = df.copy()

    # Apply filters
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

    # Keep finite lines inside E_grid
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

    # Normalize input intensities if desired
    if normalize:
        max_I = np.max(np.abs(I_lines_all)) if len(I_lines_all) > 0 else 0.0

        if max_I > 0:
            I_lines_all = I_lines_all / max_I
            data[intensity_col + "_normalized"] = I_lines_all

    data["_I_conv"] = I_lines_all

    # Total convolution
    spec = convolve_lines(E_grid=E_grid,E_lines=E_lines_all,
        I_lines=I_lines_all,sigma=sigma)

    spec_max = np.max(spec) if len(spec) > 0 else 0.0
    spec_area = np.trapz(spec, E_grid) if len(spec) > 0 else 0.0

    # Convolution by v_i
    vi_values = sorted(data[vi_col].unique())

    by_vi = {}
    sticks_by_vi = {}
    max_by_vi = {}
    area_by_vi = {}

    for vi in vi_values:
        sub = data[data[vi_col] == vi].copy()

        E_lines_vi = sub[energy_col].to_numpy(dtype=float)
        I_lines_vi = sub["_I_conv"].to_numpy(dtype=float)

        spec_vi = convolve_lines(E_grid=E_grid,E_lines=E_lines_vi,
            I_lines=I_lines_vi,sigma=sigma)

        by_vi[vi] = spec_vi
        sticks_by_vi[vi] = sub

        max_by_vi[vi] = np.max(spec_vi) if len(spec_vi) > 0 else 0.0
        area_by_vi[vi] = np.trapz(spec_vi, E_grid) if len(spec_vi) > 0 else 0.0

    # Return
    return {
        "spec": spec,"spec_max": spec_max,"spec_area": spec_area,
        "by_vi": by_vi,"vi_values": vi_values,"sticks_by_vi": sticks_by_vi,
        "max_by_vi": max_by_vi,"area_by_vi": area_by_vi,
        "sticks": data,"n_lines": len(data),
        "energy_col": energy_col,"intensity_col": intensity_col,
        "vi_col": vi_col,"sigma": sigma,"filters": filters,"normalize": normalize,
    }

#-------------------------------------------------------------------------------------------
# Boltzmann distribution ; Temp effect
#-------------------------------------------------------------------------------------------

# Vibrational energy values @ J = 0
def get_vib_energies_df(df,vi_col="v_i",Ei_col="Ei_eV"):
    """
    Params: 
    df: pandas df - transitions df
    vi_col: str - initial vib level column in df
    Ei_col: str - initial rovib lvl E column in df

    Returns:
    Evib: dict - E by vib level
    E0: list - min E in vibrational levels
    """
    Evib = (df.groupby(vi_col)[Ei_col].min().to_dict())
    E0 = min(Evib.values())
    return Evib, E0

# Rovibrational every values
def get_rot_energies_df(df,vi_col="v_i",Ji_col="J_i",Omi_col="Omega_i",Ei_col="Ei_eV"):
    """
    Params: 
    df: pandas df - transitions df
    vi_col: str - init vib level column in df
    Ji_col: str - init rotvib level column in df
    Omi_col: str - init omega level column in df
    Ei_col: str - init rovib level E column in df

    Returns:
    Erot: dict - rovibrational levels energies by v,J,Om
    Evib: dict - vibrational levels energies by v
    E0: list - min energies for vib levels
    """
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

# Computing transition intensities Tvib and Trot dependent
def thermal_populations_df(df,Tvib,Trot,vi_col="v_i",Ji_col="J_i",Omi_col="Omega_i",Ei_col="Ei_eV"):
    """
    Params: 
    df: pandas df - transitions df
    Tvib: float - vibrational Temp
    Trot: float - rotational Temp
    vi_col: str - init vib level column
    Ji_col: str - init rot level column
    Omi_col: str - init Omega level column
    Ei_col: str - init rovib level E value column

    Returns:
    Pvib: dict - Boltzmann pop by vib level (v)
    Prot: dict - Boltzmann pop by rovib level (J,Om)
    pop_dat: dict - Info about thermal contribution
    """
    
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

# Applying thermal pop to transition intensities
def sticks_by_temp_df(df,Tvib_values,Trot_values,Ei_col="Ei_eV",
            vi_col="v_i",Ji_col="J_i",Omi_col="Omega_i",intensity_cols=("matrix_coh_sum_abs",)):
    """
    Params: 
    df: pandas df - transition df
    Tvib_values: list - vibrational temp values
    Trot_values: list - rotrational temp values
    vi_col: str - init vib level column
    Ji_col: str - init rot level column
    Omi_col: str - init Omega level column
    Ei_col: str - init rovib level E value column
    intensity_cols: str - transition intensity column 

    Returns: 
    sticks_temp: dict - thermal distributed sticks transitions
    pop_temp: dict - thermal information
    """

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

#-----------------------------------------------------------------------------------
# File management
#-----------------------------------------------------------------------------------
def safe_filename_label(label):
    return (label.replace(" -> ", "_to_").replace(" ", "").replace("+", "p").replace("/", ""))

# Writing output - convoluted spectra
def write_conv_file_one_transition_temperature(output_file,conv,E_grid,
                                        transition_label=None,Tvib=None,Trot=None):
    
    """
    Params:
    output_file: str or path
    conv: dict - convoluted spectra
    E_grid: np.ndarray - 
    transition_label: str - electronic state transition labels
    Tvib: float - vibrational temp
    Trot: float - rotational temp
    """

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

def get_initial_state_from_label(label):
    if "->" not in label:
        raise ValueError(f"Transition label does not contain '->': {label}")
    return label.split("->")[0].strip()

def get_exp_max_in_range(exp_data, Emin, Emax, energy_col=0, intensity_col=3):
    E = exp_data[:, energy_col]
    I = exp_data[:, intensity_col]
    mask = (E >= Emin) & (E <= Emax)
    if not np.any(mask):
        raise ValueError(f"No experimental points in range {Emin} - {Emax} eV")

    I_range = I[mask] ; E_range = E[mask]
    imax = np.argmax(I_range)
    Imax = I_range[imax] ; E_at_max = E_range[imax]

    return Imax, E_at_max

def get_exp_max_by_state(exp_data, exp_ranges, energy_col=0, intensity_col=3):
    exp_max = {}

    for state_label, (Emin, Emax) in exp_ranges.items():
        Imax, E_at_max = get_exp_max_in_range(exp_data=exp_data,Emin=Emin,Emax=Emax,
            energy_col=energy_col,intensity_col=intensity_col)

        exp_max[state_label] = {"range": (Emin, Emax),
            "Imax": Imax,"E_at_max": E_at_max}

    return exp_max

def get_theory_total_by_state(conv_by_transition):
    theory_total_by_state = {}

    for transition_label, conv in conv_by_transition.items():
        initial_state = get_initial_state_from_label(transition_label)
        if initial_state not in theory_total_by_state:
            theory_total_by_state[initial_state] = np.zeros_like(conv["spec"], dtype=float)
        theory_total_by_state[initial_state] += conv["spec"]

    return theory_total_by_state

def get_theory_max_by_state(conv_by_transition):
    theory_total_by_state = get_theory_total_by_state(conv_by_transition)

    theory_max_by_state = {}
    for initial_state, spec in theory_total_by_state.items():
        theory_max_by_state[initial_state] = np.max(spec) if len(spec) > 0 else 0.0
    return theory_max_by_state, theory_total_by_state

def get_scale_by_state(conv_by_transition, exp_max):
    theory_max_by_state, theory_total_by_state = get_theory_max_by_state(conv_by_transition)

    scale_by_state = {}
    for initial_state, theory_max in theory_max_by_state.items():
        if initial_state not in exp_max:
            raise KeyError(f"No experimental maximum found for initial state: {initial_state}")

        exp_Imax = exp_max[initial_state]["Imax"]
        if theory_max > 0.0:
            scale_by_state[initial_state] = exp_Imax / theory_max
        else:
            scale_by_state[initial_state] = 0.0

    return scale_by_state, theory_max_by_state, theory_total_by_state

def add_state_normalization_to_convs(conv_by_transition, exp_max):
    scale_by_state, theory_max_by_state, theory_total_by_state = get_scale_by_state(
        conv_by_transition=conv_by_transition,exp_max=exp_max)

    for transition_label, conv in conv_by_transition.items():
        initial_state = get_initial_state_from_label(transition_label)
        scale = scale_by_state[initial_state]

        conv["initial_state"] = initial_state
        conv["scale_state"] = scale
        conv["spec_state_norm"] = conv["spec"] * scale
        conv["by_vi_state_norm"] = {}

        for vi, spec_vi in conv["by_vi"].items():
            conv["by_vi_state_norm"][vi] = spec_vi * scale

    return conv_by_transition, scale_by_state, theory_max_by_state, theory_total_by_state

#============================================================================
# Input file parser
#============================================================================
# Comments
def strip_inline_comment(line):
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("#") or stripped.startswith("!"):
        return ""
    clean = line.split("#", 1)[0]
    clean = clean.split("!", 1)[0]
    return clean.strip()

# Boolean parser
def parse_bool(value):
    value = str(value).strip().lower()
    if value in ["true", "t", "yes", "y", "1", ".true."]:
        return True
    if value in ["false", "f", "no", "n", "0", ".false."]:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")

# Generic parsers
def parse_float_list(values):
    return [float(v) for v in values]
def parse_int_list(values):
    return [int(v) for v in values]

# Read input file
def read_input_file(filename):
    filename = Path(filename)
    data = {}

    with open(filename, "r") as f:
        for iline, line in enumerate(f, start=1):
            clean = strip_inline_comment(line)
            if not clean:
                continue
            parts = clean.split()
            key = parts[0].lower().replace("-", "_")
            values = parts[1:]
            if len(values) == 0:
                raise ValueError(
                    f"Missing value in input file {filename}, "
                    f"line {iline}: {line.rstrip()}"
                )
            data[key] = values
    return data

# Access keywords
def get_required(data, key):
    key = key.lower().replace("-", "_")
    if key not in data:
        raise KeyError(f"Missing required input keyword: {key}")
    return data[key]

def get_optional(data, key, default):
    key = key.lower().replace("-", "_")
    return data.get(key, default)

# Convert input dictionary to argparse.Namespace
def input_data_to_namespace(data):
    args = argparse.Namespace()

    # General paths / dimensions
    args.dir = Path(get_required(data, "base_dir")[0])
    args.ntrans = int(get_required(data, "ntransitions")[0])
    args.nsttN = int(get_required(data, "nsttsN")[0])
    args.nsttC = int(get_required(data, "nsttsC")[0])
    args.skiprows = int(get_optional(data, "skiprows", ["1"])[0])

    # Transitions
    args.transitions = {}
    if args.ntrans == 1:
        trans_name = " ".join(get_required(data, "trans_name"))
        trans_file = get_required(data, "trans_file")[0]

        args.transitions[trans_name] = trans_file

    else:
        for itrans in range(1, args.ntrans + 1):
            name_key = f"trans_name_{itrans}"
            file_key = f"trans_file_{itrans}"

            trans_name = " ".join(get_required(data, name_key))
            trans_file = get_required(data, file_key)[0]

            args.transitions[trans_name] = trans_file

    # Columns
    args.energy_col = get_optional(data, "energy_col", ["Delta_eV"])[0]
    args.Ei_col = get_optional(data, "Ei_col", ["Ei_eV"])[0]
    args.vi_col = get_optional(data, "vi_col", ["v_i"])[0]
    args.Ji_col = get_optional(data, "Ji_col", ["J_i"])[0]
    args.Omi_col = get_optional(data, "Omega_i_col", ["Omega_i"])[0]
    args.intensity_col = get_optional(
        data,
        "intensity_col",
        ["matrix_coh_sum_abs"],
    )[0]
    args.intensity_col_T = args.intensity_col + "_T"

    # Convolution parameters
    args.sigma = float(get_optional(data, "sigma", ["0.025"])[0])
    args.ngrid = int(get_optional(data, "ngrid", ["6000"])[0])
    args.deltaJ_max = float(get_optional(data, "deltaJ_max", ["inf"])[0])
    args.normalize = parse_bool(
        get_optional(data, "normalize", ["false"])[0]
    )

    # --------------------------------------------------------
    # Energy grid
    # --------------------------------------------------------

    args.use_exp_grid = parse_bool(
        get_optional(data, "use_exp_grid", ["false"])[0]
    )

    args.energy_grid_mode = get_optional(
        data,
        "energy_grid_mode",
        ["auto"],
    )[0].lower()

    if args.energy_grid_mode == "manual":
        vals = get_required(data, "energy_range")
        if len(vals) != 2:
            raise ValueError("energy_range must have two values: Emin Emax")
        args.energy_range = (
            float(vals[0]),
            float(vals[1]),
        )
    else:
        args.energy_range = None

    # --------------------------------------------------------
    # Temperatures
    # --------------------------------------------------------

    args.Tvib_values = parse_float_list(
        get_required(data, "Tvib_values")
    )

    args.Trot_values = parse_float_list(
        get_required(data, "Trot_values")
    )

    # --------------------------------------------------------
    # Experimental spectrum
    # --------------------------------------------------------

    args.read_exp = parse_bool(
        get_optional(data, "read_exp", ["false"])[0]
    )

    args.state_labels = {}
    args.exp_ranges = {}

    if args.read_exp:
        args.expdat = Path(get_required(data, "exp_esp")[0])
        args.exp_skiprows = int(
            get_optional(data, "exp_skiprows", ["3"])[0]
        )
        args.exp_energy_col = int(
            get_optional(data, "exp_energy_col", ["0"])[0]
        )
        args.exp_intensity_col = int(
            get_optional(data, "exp_intensity_col", ["3"])[0]
        )

        if args.nsttN == 1:
            state_label = " ".join(
                get_optional(data, "state_label", ["PHGS"])
            )

            vals = get_required(data, "exp_range")

            if len(vals) != 2:
                raise ValueError("exp_range must have two values: Emin Emax")

            args.state_labels[1] = state_label

            args.exp_ranges[state_label] = (
                float(vals[0]),
                float(vals[1]),
            )

        else:
            for istate in range(1, args.nsttN + 1):
                label_key = f"state_label_{istate}"
                range_key = f"exp_range_{istate}"

                state_label = " ".join(get_required(data, label_key))
                vals = get_required(data, range_key)

                if len(vals) != 2:
                    raise ValueError(
                        f"{range_key} must have two values: Emin Emax"
                    )

                args.state_labels[istate] = state_label
                args.exp_ranges[state_label] = (
                    float(vals[0]),
                    float(vals[1]),
                )
    else:
        args.expdat = None
        args.exp_skiprows = None
        args.exp_energy_col = None
        args.exp_intensity_col = None
        args.state_labels = None
        args.exp_ranges = None

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    args.output_dir = Path(
        get_optional(data, "output_dir", ["."])[0]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args
    
def main(input_file):

    data = read_input_file(input_file)
    args = input_data_to_namespace(data)

    # ========================================================
    # Read transition files
    # ========================================================

    transition_files = {
        label: args.dir / filename
        for label, filename in args.transitions.items()
    }

    transitions_df = {
        label: read_transition_output_df(path, skiprows=args.skiprows)
        for label, path in transition_files.items()
    }

    # ========================================================
    # Read experimental spectrum
    # ========================================================

    if args.read_exp:

        exp_data = np.loadtxt(args.expdat, skiprows=args.exp_skiprows)

        exp_max = get_exp_max_by_state(exp_data=exp_data,exp_ranges=args.exp_ranges,
            energy_col=args.exp_energy_col,intensity_col=args.exp_intensity_col)

    else:

        exp_data = None
        exp_max = None

    # ========================================================
    # Energy grid
    # ========================================================

    if args.energy_grid_mode == "manual":

        E_min, E_max = args.energy_range

    elif args.use_exp_grid and args.read_exp:

        E_exp = exp_data[:, args.exp_energy_col]

        E_min = E_exp.min()
        E_max = E_exp.max()

    else:

        E_min = min(
            df[args.energy_col].min()
            for df in transitions_df.values()
        )

        E_max = max(
            df[args.energy_col].max()
            for df in transitions_df.values()
        )

    E_grid = np.linspace(E_min, E_max, args.ngrid)

    # ========================================================
    # Filters
    # ========================================================

    if np.isinf(args.deltaJ_max):
        filters = None
    else:
        filters = {
            "Delta_J": lambda x: np.abs(x) <= args.deltaJ_max
        }

    # ========================================================
    # Main loop
    # ========================================================

    all_sticks_temp = {}
    all_pop_temp = {}
    all_conv_temp = {}
    all_scale_by_state = {}
    all_theory_max_by_state = {}
    all_theory_total_by_state = {}

    for Tvib in args.Tvib_values:
        for Trot in args.Trot_values:
            temp_key = (Tvib, Trot)

            all_conv_temp[temp_key] = {}
            all_sticks_temp[temp_key] = {}
            all_pop_temp[temp_key] = {}

            # ------------------------------------------------
            # Build all convolutions for this temperature
            # ------------------------------------------------
            for transition_label, df in transitions_df.items():
                sticks_temp, pop_temp = sticks_by_temp_df(df,Tvib_values=[Tvib],Trot_values=[Trot],
                    Ei_col=args.Ei_col,vi_col=args.vi_col,Ji_col=args.Ji_col,Omi_col=args.Omi_col,
                    intensity_cols=(args.intensity_col,))

                df_T = sticks_temp[temp_key]

                conv_T = build_conv_df(df_T,E_grid,energy_col=args.energy_col,
                    intensity_col=args.intensity_col_T,vi_col=args.vi_col,sigma=args.sigma,
                    filters=filters,normalize=False)

                all_sticks_temp[temp_key][transition_label] = df_T
                all_pop_temp[temp_key][transition_label] = pop_temp[temp_key]
                all_conv_temp[temp_key][transition_label] = conv_T

            # ------------------------------------------------
            # State normalization
            # ------------------------------------------------

            if args.normalize and args.read_exp:

                (all_conv_temp[temp_key],scale_by_state,theory_max_by_state,theory_total_by_state
                ) = add_state_normalization_to_convs(conv_by_transition=all_conv_temp[temp_key],
                    exp_max=exp_max)

                all_scale_by_state[temp_key] = scale_by_state
                all_theory_max_by_state[temp_key] = theory_max_by_state
                all_theory_total_by_state[temp_key] = theory_total_by_state

            # ------------------------------------------------
            # Write output files
            # ------------------------------------------------

            for transition_label, conv_T in all_conv_temp[temp_key].items():
                safe_label = safe_filename_label(transition_label)
                output_file = (args.output_dir /f"{safe_label}_Tvib{Tvib:g}_Trot{Trot:g}.dat")

                if args.normalize and args.read_exp:
                    conv_write = conv_T.copy()
                    conv_write["spec"] = conv_T["spec_state_norm"]
                    conv_write["by_vi"] = conv_T["by_vi_state_norm"]
                else:
                    conv_write = conv_T

                write_conv_file_one_transition_temperature(output_file=output_file,conv=conv_write,
                    E_grid=E_grid,transition_label=transition_label,Tvib=Tvib,Trot=Trot)

    return {
        "args": args,
        "E_grid": E_grid,
        "transitions_df": transitions_df,
        "exp_data": exp_data,
        "exp_max": exp_max,
        "all_sticks_temp": all_sticks_temp,
        "all_pop_temp": all_pop_temp,
        "all_conv_temp": all_conv_temp,
        "all_scale_by_state": all_scale_by_state,
        "all_theory_max_by_state": all_theory_max_by_state,
        "all_theory_total_by_state": all_theory_total_by_state,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")

    cli_args = parser.parse_args()

    results = main(cli_args.input_file)