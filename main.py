import argparse
import math
import re
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from tqdm import tqdm


EH_TO_CM = 219474.6
EH_TO_EV = 27.2114
BOHR_TO_ANGSTROM = 0.529177210903


# =============================================================================
# Rovibrational level object from DUO
# =============================================================================
@dataclass(frozen=True)
class DuoLevel:
    """
    Class Attributes:
    system: str - e- state identifier
    v: int - vib quantum number
    J: float - total angular momentum number
    Omega: float - J projection on the molecular axis
    Sigma: float - spin projection onto the molecular axis
    Lambda: float - e- orbital angular momentum projection onto the molecular axis
    parity: int - parity of the rovibrational level, either +1 or -1
    duo_index: int - inside DUO index identification
    energy: float - energy of the rovibrational level in cm-1 
    """
    system: str
    v: int
    J: float
    Omega: float
    Sigma: float
    Lambda: float
    parity: int
    duo_index: int
    energy_cm: float

# =============================================================================
# Coefficient object from DUO
# =============================================================================
@dataclass(frozen=True)
class CoeffComponent:
    """
    Class Attributes: 
    coeff: complex - rot level coefficient in the rovib basis
    v_basis: int - #vibrational level in the basis
    Lambda: float - e- orbital angular momentum projection onto the molecular axis
    Sigma: float - spin projection onto the molecular axis
    Omega: float - J projection onto the molecular axis
    """
    coeff: complex
    v_basis: int
    Lambda: float
    Sigma: float
    Omega: float

# =============================================================================
# Small utilities
# =============================================================================

def as_list(x):
    """
    Params: 
    x: int, list, tuple or np.ndarray
    Returns: 
    x: list
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]

# Resolve system path and file. Check with directory and prefix-directory.
def resolve_system_file(base_path, system, filename):
    """
    Locate system-specific style in directory and prefix conventions.

    Params: 
    base_path: str or Path - base directory of calculations
    system: str - specifying directory
    filename: str - file to locate
    
    Returns: 
    Path: path - to the first matching file
    
    Raises: 
    FileNotFoundError: If file cannot be found
    """

    base_path = Path(base_path)

    candidate_dir = base_path / system / filename                                   # Look at directory style -> base/system_state/filename
    if candidate_dir.exists():
        return candidate_dir

    candidate_prefix = base_path / f"{system}{filename}"                            # Look at prefix style -> base/system_statefilename -> base/system_state/filename
    if candidate_prefix.exists():
        return candidate_prefix

    raise FileNotFoundError(
        f"Could not find {filename} for system {system}. Tried:\n"
        f"  {candidate_dir}\n"
        f"  {candidate_prefix}"
    )

# =============================================================================
# System specific information
# =============================================================================
# Mapping dyson indexes to omega projections and electronic states of PH and PHm
# TODO: making it for general system or input dependent
def build_dyson_index_to_omega():
    return {
        "PHX3Sm": { 12: -1.0, 13:  0.0, 14:  1.0 },

        "PHa1D": { 1: [-2.0, 2.0] },                                                # Singlet Delta: the same Dyson file is used for both ±2 components.

        "PHMX2P": { 2: -1.5, 3: -0.5, 7:  0.5, 8:  1.5 },

        "PHMa4Sm": { 15: -1.5, 16: -0.5, 17:  0.5, 18:  1.5 },

        "PHMA2D": { 5:  -2.5, 6:  -1.5, 10:  1.5, 11:  2.5 },

        "PHM12Sm": { 4: -0.5, 9:  0.5 },
    }

# =============================================================================
# Reading DUO-formatted files
# =============================================================================
# Reading vibrational eigenfunctions: header
def is_head_eigenvib(line: str) -> bool:
    """
    Params: 
    line: str - line in eigenfunc file
    Returns:
    bool: T or F
    """

    s = line.strip()
    if not s:
        return False

    parts = s.split()
    if len(parts) < 2:
        return False

    if not re.fullmatch(r"[+-]?\d+", parts[0]):
        return False

    if not re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([Ee][+-]?\d+)?", parts[1]):
        return False

    return True

# Reading vibrational eigenfunctions: functions
def parse_duo_vib_einfun(fname, npoints: int, nvib: int):
    """
    Read vibrational eigenfunctions from a DUO output file 

    Input params: 
    fname: str or Path - Path to the DUO output file
    npoints: int - #radial-grid points per vibrational state
    nvib: int - #vibrational states to read

    Returns:
    np.ndarray - matrix with (npoints,nvib), each column containing 
                the WF of one vibrational state

    Raises
    ValueError - If a vibrational function has more than npoints
    """

    fname = Path(fname)
    vibmat = np.zeros((npoints, nvib), dtype=float)

    with open(fname, "r") as f:
        lines = f.readlines()

    ncurrpoint = 0
    ncurrvib = None

    for line in lines:                                                              #Reading the vibrational wavefunctions until the set number of vibrational levels in input
        if is_head_eigenvib(line):
            ncurrpoint = 0
            ncurrvib = int(line.split()[3])
            if ncurrvib >= nvib:
                break

        elif "End of contracted basis" in line:
            break

        else:
            if ncurrvib is None:
                continue
            if ncurrpoint >= npoints:
                raise ValueError(
                    f"Too many points while reading vib={ncurrvib} in {fname}"
                )
            vibmat[ncurrpoint, ncurrvib] = float(line)
            ncurrpoint += 1

    return vibmat

# Reading rovibrational levels energies and quantum numbers
def read_duo_levels(fname, system, nvib_max=None, j_max=None):
    """
    Reads the information of the rovibrational levels. 

    Params: 
    fname: str or Path - file containing the rovibrational levels information
    system: str - state identification
    nvib_max: int - max vib quantum number to be read
    j_max: float - max J quantum number to be read

    Returns: 
    levels: list of DuoLevel objects
    """

    fname = Path(fname)
    levels = []

    with open(fname, "r") as f:
        for iline, line in enumerate(f):
            parts = line.split()

            if len(parts) < 3:
                continue

            J = float(parts[0])
            duo_index = int(parts[1])
            energy_cm = float(parts[2])
            v = int(parts[4])
            Lambda = float(parts[5])
            Sigma = float(parts[7])
            Omega = float(parts[8])
            parity = 1 if parts[9] == "-" else 0

            if nvib_max is not None and v >= nvib_max:
                continue

            if j_max is not None and J >= j_max:
                continue

            if duo_index <= 0:
                raise ValueError(
                    f"Invalid DUO index in {fname}, line {iline}: {duo_index}"
                )

            levels.append(
                DuoLevel( system=system, v=v, J=J, Omega=Omega, Sigma=Sigma, Lambda=Lambda,   # Getting rovibrational levels energies and quantum numbers
                        parity=parity, duo_index=duo_index, energy_cm=energy_cm )
            )

    return levels

# Creating dictionary keys for rovibrational levels
def level_state_key(level):
    """
    From rovib level take coeff key
    """
    return (
        level.J,
        level.parity,
        level.duo_index,
    )

# Reading rovibrational basis coefficients in DUO format
def read_coefficients(filename, debug_first_lines=0):
    """
    Params:
    filename: str or Path - file containing rovib coeffs 
    debug_first_lines: int - general info of the file

    Returns: 
    tmp: dict of CoeffComponent
    """

    filename = Path(filename)
    tmp = defaultdict(list)
    ndebug = 0

    with open(filename, "r") as f:
        for line in f:
            parts = line.split()

            if len(parts) < 10:
                continue

            if debug_first_lines and ndebug < debug_first_lines:
                print("RAW COEFF LINE:", line.rstrip())
                for k, p in enumerate(parts):
                    print(f"  col[{k:02d}] = {p}")
                ndebug += 1

            try:
                duo_index = int(parts[0])
                J = float(parts[1])
                parity = int(parts[2])
                coeff = complex(float(parts[3]), 0.0)
                v_basis = int(parts[5])
                Lambda = float(parts[6])
                Sigma = float(parts[8])
                Omega = float(parts[9])
            except ValueError:
                continue

            key = (
                J,
                parity,
                duo_index,
            )

            tmp[key].append(
                CoeffComponent( coeff=coeff, v_basis=v_basis, Lambda=Lambda, Sigma=Sigma, Omega=Omega )   # Getting rotational components in vibrational basis for each rovibrational level
            )

    return dict(tmp)

def print_coeff_summary(coeffs, label):
    """
    Diagnostic of coeff components
    """
    norms = []
    ncomps = []
    vmins = []
    vmaxs = []

    for comps in coeffs.values():
        if not comps:
            continue

        norms.append(sum(abs(c.coeff) ** 2 for c in comps))
        ncomps.append(len(comps))
        vmins.append(min(c.v_basis for c in comps))
        vmaxs.append(max(c.v_basis for c in comps))

    if not norms:
        print(f"{label}: no coefficients found")
        return

    print(f"{label} coefficient diagnostics:")
    print(f"  keys          : {len(coeffs)}")
    print(f"  ncomp min/max : {min(ncomps)} / {max(ncomps)}")
    print(f"  norm  min/max : {min(norms):.6e} / {max(norms):.6e}")
    print(f"  v_basis min/max: {min(vmins)} / {max(vmaxs)}")


# =============================================================================
# Functions to calculate relative intensity of rovibrationally ressolved spectra
# =============================================================================

#Bra-DyN-Ket product
def bra_ket_weighted(vib_ini, vib_fin, dyson_values):
    return np.dot(vib_fin, dyson_values * vib_ini)

#Kronecker delta
def kronecker_delta(i, j):
    return 1.0 if i == j else 0.0

# Return (-1)**x, using an exact real value for integer x
def minus_one_power(x):
    if np.isclose(x, round(x)):
        return 1.0 if int(round(x)) % 2 == 0 else -1.0
    return np.exp(1j * np.pi * x)

#Factorial
def fact(n):
    if n % 1 != 0:
        return 0

    n = int(n)
    if n < 0:
        return 0

    f = 1
    for i in range(n):
        f *= i + 1
    return f

# Wigner 3j explicit expression - Cache computations for faster performance
@lru_cache(maxsize=None)
def W3_exp(J1, J2, J3, m1, m2, m3):
    K = max(0, J2 - J3 - m1, J1 - J3 + m2)
    if K % 1 != 0:
        return 0.0
    K = int(K)

    N = min(J1 + J2 - J3, J1 - m1, J2 + m2)
    if N % 1 != 0:
        return 0.0
    N = int(N)

    deltaK = kronecker_delta(m1 + m2 + m3, 0)
    if deltaK == 0.0:
        return 0.0

    m1pf = (-1) ** int(J1 - J2 - m3)

    J1MJ2mJ3 = fact(J1 + J2 - J3)
    J1mJ2MJ3 = fact(J1 - J2 + J3)
    mJ1MJ2MJ3 = fact(-J1 + J2 + J3)
    J1MJ2MJ3M1 = fact(J1 + J2 + J3 + 1)

    if J1MJ2MJ3M1 == 0:
        return 0.0

    first_root = math.sqrt(
        (J1MJ2mJ3 * J1mJ2MJ3 * mJ1MJ2MJ3) / J1MJ2MJ3M1
    )

    J1mO1 = fact(J1 - m1)
    J1MO1 = fact(J1 + m1)
    J2mO2 = fact(J2 - m2)
    J2MO2 = fact(J2 + m2)
    J3mO3 = fact(J3 - m3)
    J3MO3 = fact(J3 + m3)

    second_arg = J1mO1 * J1MO1 * J2mO2 * J2MO2 * J3mO3 * J3MO3
    if second_arg < 0:
        return 0.0

    second_root = math.sqrt(second_arg)

    third_sum = 0.0
    for k in range(K, N + 1):
        kf = fact(k)
        J1MJ2mJ3mk = fact(J1 + J2 - J3 - k)
        J1mO1mk = fact(J1 - m1 - k)
        J2MO2mk = fact(J2 + m2 - k)
        J3mJ1MO1Mk = fact(J3 - J2 + m1 + k)
        J3mJ1mO2Mk = fact(J3 - J1 - m2 + k)

        denom = (
            kf
            * J1MJ2mJ3mk
            * J1mO1mk
            * J2MO2mk
            * J3mJ1MO1Mk
            * J3mJ1mO2Mk
        )

        if denom == 0:
            continue

        third_sum += ((-1) ** k) / denom

    return deltaK * m1pf * first_root * second_root * third_sum

# Reading dyson norms files for each pair of initial and final omega states
def read_dyson_raw(d_neutral, d_cation, dyson_path, bohr_to_angstrom=True):
    dyson_path = Path(dyson_path)
    raw = {}
    r_ref = None

    for idx_i, omega_i_vals in d_neutral.items():
        for idx_f, omega_f_vals in d_cation.items():
            fname = dyson_path / f"dyson_{idx_i:02d}_{idx_f:02d}.dat"            # Reading dyson file as a function of omega combination

            if not fname.exists():
                raise FileNotFoundError(f"Missing Dyson file: {fname}")          # Checking if the dyson file exists

            r, dyson = np.loadtxt(fname, unpack=True)

            if r_ref is None:
                r_ref = r.copy()
            else:
                if len(r) != len(r_ref) or not np.allclose(r, r_ref):            # Checking if the R grid is consistent
                    raise ValueError(f"Inconsistent R grid in {fname}")

            r_use = r * BOHR_TO_ANGSTROM if bohr_to_angstrom else r.copy()       # Converting to angstrom

            raw[(idx_i, idx_f)] = { "idx_i": idx_i, "idx_f": idx_f, "omega_i_values": as_list(omega_i_vals), "omega_f_values": as_list(omega_f_vals),  # Storing the data as a dictionary for omega combination
                                "r": r_use, "dyson": np.asarray(dyson, dtype=float), "file": str(fname) }

    return raw

# Combination of dyson norms into a single magnitude for cummulative keys
def combine_dyson_by_omega(raw, mode="quadrature"):
    if mode not in ["quadrature", "sum"]:
        raise ValueError("mode must be 'quadrature' or 'sum'")

    accum = defaultdict(lambda: None)
    sources = defaultdict(list)
    r_by_key = {}

    for (idx_i, idx_f), item in raw.items():
        r = item["r"]
        dyson = item["dyson"]

        for omega_i in item["omega_i_values"]:
            for omega_f in item["omega_f_values"]:
                key = (omega_i, omega_f)

                if key not in r_by_key:
                    r_by_key[key] = r
                else:
                    if len(r) != len(r_by_key[key]) or not np.allclose(r, r_by_key[key]):     
                        raise ValueError(f"Non-matching R grids for Dyson key {key}")

                sources[key].append(item["file"])

                if accum[key] is None:                                                              # Accumulating the dyson norms:
                    accum[key] = dyson**2 if mode == "quadrature" else dyson.copy()                 # either quadrature : sum(dyson**2) 
                else:                                                                               # or sum of the dyson norms on the combination of omega values
                    accum[key] += dyson**2 if mode == "quadrature" else dyson

    combined = {}
    for key, val in accum.items():
        dyson_eff = np.sqrt(val) if mode == "quadrature" else val
        combined[key] = { "omega_i": key[0], "omega_f": key[1], "r": r_by_key[key], "dyson": dyson_eff, "sources": sources[key] } # Reducing data to a single dictionary structure

    return combined

# Interpolation of combined dyson norms as a function of X-Y atomic coordinate 
# with the information coming from DUO outputs
def make_dyson_splines(combined, spline_type="cubic", extrapolate=False):
    if spline_type not in ["cubic", "linear"]:
        raise ValueError("spline_type must be 'cubic' or 'linear'")

    splines = {}
    for key, item in combined.items():
        r = np.asarray(item["r"], dtype=float)
        dyson = np.asarray(item["dyson"], dtype=float)

        order = np.argsort(r)
        r = r[order]
        dyson = dyson[order]

        if np.any(~np.isfinite(r)) or np.any(~np.isfinite(dyson)):
            raise ValueError(f"Non-finite values in Dyson data for key {key}")

        if np.any(np.diff(r) <= 0):
            raise ValueError(f"R grid is not strictly increasing for key {key}")

        if spline_type == "cubic":
            splines[key] = CubicSpline(r,dyson,bc_type="natural",extrapolate=extrapolate)
        else:
            splines[key] = interp1d(r,dyson,kind="linear",bounds_error=not extrapolate,fill_value="extrapolate" if extrapolate else np.nan)
    return splines

# Combination of dyson functions returning the interpolated dyson norms 
# and the raw data for the combination of omega values
def read_dyson_splines(d_neutral,d_cation,dyson_path,combine_mode="quadrature",spline_type="cubic",
                    bohr_to_angstrom=True,extrapolate=False):
    
    raw = read_dyson_raw(d_neutral=d_neutral, d_cation=d_cation, dyson_path=dyson_path, bohr_to_angstrom=bohr_to_angstrom)

    combined = combine_dyson_by_omega(raw, mode=combine_mode)

    splines = make_dyson_splines(combined=combined, spline_type=spline_type, extrapolate=extrapolate)

    return splines, raw, combined

# Computing rotational factors for each pair of initial states in cache for faster performance
# Cache is implemented using lru_cache - if combination exists it is not recomputed
@lru_cache(maxsize=None)
def rotational_factor_cached(J_i, J_f, Omega_i, Omega_f, K_values_tuple):
    """
    Wigner 3j explicit expression conditions of angular momenta combination: 
    1. |J_i - J_f| <= K <= J_i + J_f
    2. (J_i + J_f + K) % 1 == 0
    3. Omega_i + Omega_f - dOmega == 0
    """

    total = 0.0

    for K in K_values_tuple:
        if not (abs(J_i - J_f) <= K <= (J_i + J_f)):
            continue

        if not np.isclose((J_i + J_f + K) % 1, 0.0):
            continue

        for dOmega in np.arange(-K, K + 1.0, 1.0):
            if not np.isclose(Omega_i + Omega_f - dOmega, 0.0):
                continue

            total += W3_exp(J_i, J_f, K, Omega_i, Omega_f, -dOmega )

    return total

def rotational_factor(J_i, J_f, Omega_i, Omega_f, K_values):
    return rotational_factor_cached(J_i, J_f, Omega_i, Omega_f, tuple(K_values))

# Relative and absolute transition energies
def transition_energy_Eh(level_i, level_f, ZPE_i_cm, ZPE_f_cm, Eelec_i_Eh, Eelec_f_Eh):
    Ei_rel = (level_i.energy_cm + ZPE_i_cm) / EH_TO_CM
    Ef_rel = (level_f.energy_cm + ZPE_f_cm) / EH_TO_CM

    Ei_abs = Eelec_i_Eh + Ei_rel
    Ef_abs = Eelec_f_Eh + Ef_rel

    return Ei_rel, Ef_rel, Ef_abs - Ei_abs

# Precomputing dyson values depending on the R grid
def precompute_dyson_values(dyson_splines, r_use):
    dyson_values = {}

    for key, spline in dyson_splines.items():
        vals = spline(r_use)

        if np.any(~np.isfinite(vals)):
            raise ValueError(
                f"Dyson spline returned non-finite values for key {key}. "
                f"R range used: {r_use.min()} - {r_use.max()}"
            )

        dyson_values[key] = vals

    return dyson_values

# Precomputing vibrational bra-ket matrices <vib_i | dyson | vib_f>
def precompute_vibrational_bk_matrices(vib_neutral_use, vib_cation_use, dyson_splines, r_use, missing_dyson="skip"):
    
    vib_bk_by_key = {}

    for keydyson, spline in dyson_splines.items():
        dyson_values = spline(r_use)

        if np.any(~np.isfinite(dyson_values)):                                              # Checking if combination of dyson norms returned non-finite values
            if missing_dyson == "error":                                                    # and it exists. If not, raise error or not depending on missing_dyson
                raise ValueError(
                    f"Dyson spline returned non-finite values for key {keydyson}. "
                    f"R range used: {r_use.min()} - {r_use.max()}"
                )
            elif missing_dyson == "skip":
                continue
            else:
                raise ValueError("missing_dyson must be 'skip' or 'error'")

        weighted_neutral = dyson_values[:, None] * vib_neutral_use
        vib_bk_by_key[keydyson] = vib_cation_use.T @ weighted_neutral

    return vib_bk_by_key

def vibrational_dyson_bra_ket(vib_ini, vib_fin, dyson_values):
    return np.dot(vib_fin, dyson_values * vib_ini)

def transition_amplitude_component_contraction(level_i,level_f,vib_bk_matrix,fc_matrix,
                                            components_i,components_f,K_values):
    
    bk_no_rotation = 0.0 + 0.0j

    coeff_sum = 0.0 + 0.0j
    coeff_abs_sum = 0.0

    bk_component_coherent = 0.0 + 0.0j
    bk_component_coherent_abs_sum = 0.0

    K_values_tuple = tuple(K_values)

    for comp_i in components_i:                                                                        # Sum over all rotational coefficients in the rovibrational basis
        for comp_f in components_f:
            coeff_factor = np.conj(comp_f.coeff) * comp_i.coeff

            coeff_sum += coeff_factor
            coeff_abs_sum += abs(coeff_factor)

            vib_dyson_comp = vib_bk_matrix[comp_f.v_basis, comp_i.v_basis]                             # Coherent rovibrational matrix element

            rot = rotational_factor_cached(level_i.J,level_f.J,comp_i.Omega,comp_f.Omega,K_values_tuple)
            phase = minus_one_power(comp_i.Omega)                                                              # Phase factor

            coherent_term = phase * rot * coeff_factor * vib_dyson_comp
            bk_component_coherent += coherent_term
            bk_component_coherent_abs_sum += abs(coherent_term)

    matrix_element_coherent = bk_component_coherent
    matrix_element_coherent_abs_sum = bk_component_coherent_abs_sum

    return (matrix_element_coherent,matrix_element_coherent_abs_sum)

def trans_amp_coherentROT(level_i,level_f,vib_bk_matrix,fc_matrix,
                                            components_i,components_f,K_values):
    
    bk_no_rotation = 0.0 + 0.0j

    coeff_sum = 0.0 + 0.0j
    coeff_abs_sum = 0.0

    bk_component_coherent = 0.0 + 0.0j
    bk_component_coherent_abs_sum = 0.0

    K_values_tuple = tuple(K_values)

    for comp_i in components_i:                                                                        # Sum over all rotational coefficients in the rovibrational basis
        for comp_f in components_f:
            coeff_factor = np.conj(comp_f.coeff) * comp_i.coeff

            coeff_sum += coeff_factor
            coeff_abs_sum += abs(coeff_factor)

            vib_dyson_comp = vib_bk_matrix[comp_f.v_basis, comp_i.v_basis]                             # Coherent rovibrational matrix element

            rot = rotational_factor_cached(level_i.J,level_f.J,comp_i.Omega,comp_f.Omega,K_values_tuple)
            phase = minus_one_power(comp_i.Omega)                                                              # Phase factor

            coherent_term = phase * rot * coeff_factor * vib_dyson_comp
            bk_component_coherent += coherent_term
            bk_component_coherent_abs_sum += abs(coherent_term)

    matrix_element_coherent_abs_sum = bk_component_coherent_abs_sum
    matrix_element_coherent = bk_component_coherent

    return (matrix_element_coherent,matrix_element_coherent_abs_sum)

# Building the transition table and gathering data
def build_transition_table(levels_neutral,levels_cation,vib_neutral,vib_cation,coeff_neutral,coeff_cation,dyson_splines,
                        rvals,mask,ZPE_neutral_cm,ZPE_cation_cm,Eelec_neutral_Eh,Eelec_cation_Eh,K_values,
                        missing_dyson="skip",missing_coeff="error",min_intensity=0.0,):
    
    rows = []

    r_use = rvals[mask]
    vib_neutral_use = vib_neutral[mask, :]
    vib_cation_use = vib_cation[mask, :]

    fc_matrix = vib_cation_use.T @ vib_neutral_use

    vib_bk_by_key = precompute_vibrational_bk_matrices(vib_neutral_use=vib_neutral_use, vib_cation_use=vib_cation_use,          # Vibrational matrix elements <vib_cation | dyson | vib_neutral>
                                                    dyson_splines=dyson_splines, r_use=r_use, missing_dyson=missing_dyson)

    for level_i in tqdm(levels_neutral, desc="Neutral levels"):                                        # Looping over all neutral levels
        key_i = level_state_key(level_i)
        chi_i = vib_neutral_use[:, level_i.v]

        if key_i not in coeff_neutral:                                                                 # Checking if coefficient exists
            if missing_coeff == "error":
                raise KeyError(f"Missing neutral coefficient key: {key_i}")
            if missing_coeff == "skip":
                continue
            raise ValueError("missing_coeff must be 'error' or 'skip'")

        components_i = coeff_neutral[key_i]

        for level_f in levels_cation:                                                                  # Looping over all cation levels
            key_f = level_state_key(level_f)
            chi_f = vib_cation_use[:, level_f.v]

            if key_f not in coeff_cation:                                                              # Checking if coefficient exists
                if missing_coeff == "error":
                    raise KeyError(f"Missing cation coefficient key: {key_f}")
                if missing_coeff == "skip":
                    continue
                raise ValueError("missing_coeff must be 'error' or 'skip'")

            components_f = coeff_cation[key_f]

            Ei_rel_Eh, Ef_rel_Eh, DeltaE_Eh = transition_energy_Eh(level_i=level_i, level_f=level_f,                     # Calculating transition energy
                                                                ZPE_i_cm=ZPE_neutral_cm, ZPE_f_cm=ZPE_cation_cm, 
                                                                Eelec_i_Eh=Eelec_neutral_Eh, Eelec_f_Eh=Eelec_cation_Eh)

            if DeltaE_Eh <= 0.0:
                continue

            keydyson = (level_i.Omega, level_f.Omega)                                                  # Dyson key

            if keydyson not in vib_bk_by_key:                                                          # Checking if combination of dyson norms exists
                if missing_dyson == "error":
                    raise KeyError(
                        f"Missing Dyson key {keydyson} for transition "
                        f"i={level_i.duo_index}, f={level_f.duo_index}"
                    )
                elif missing_dyson == "skip":
                    continue
                else:
                    raise ValueError("missing_dyson must be 'skip' or 'error'")

            vib_bk_matrix = vib_bk_by_key[keydyson]

            # (matrix_element_coherent,matrix_element_coherent_abs_sum) = transition_amplitude_component_contraction(level_i=level_i, level_f=level_f, vib_bk_matrix=vib_bk_matrix, fc_matrix=fc_matrix, 
            #                                             components_i=components_i, components_f=components_f, K_values=K_values)

            (matrix_element_coherent,matrix_element_coherent_abs_sum) = trans_amp_coherentROT(level_i=level_i, level_f=level_f, vib_bk_matrix=vib_bk_matrix, 
                                                                        fc_matrix=fc_matrix, components_i=components_i, components_f=components_f, K_values=K_values)

            rows.append(
                { "v_i": level_i.v, "J_i": level_i.J, "Omega_i": level_i.Omega, "Sigma_i": level_i.Sigma, "Lambda_i": level_i.Lambda, "parity_i": level_i.parity, "index_i": level_i.duo_index,
                "v_f": level_f.v, "J_f": level_f.J, "Omega_f": level_f.Omega, "Sigma_f": level_f.Sigma, "Lambda_f": level_f.Lambda, "parity_f": level_f.parity, "index_f": level_f.duo_index,
                "Ei_eV": Ei_rel_Eh * EH_TO_EV,"Ef_eV": Ef_rel_Eh * EH_TO_EV,"DeltaE_eV": DeltaE_Eh * EH_TO_EV,
                "matrix_coh_real": np.real(matrix_element_coherent), "matrix_coh_imag": np.imag(matrix_element_coherent), "matrix_coh_abs": np.abs(matrix_element_coherent),
                "matrix_coh_sum_abs": matrix_element_coherent_abs_sum,
                "Delta_v": level_f.v - level_i.v, "Delta_J": level_f.J - level_i.J, "Delta_Omega": level_f.Omega - level_i.Omega,
                }
            )

    df = pd.DataFrame(rows)                                                                            # Creating dataframe from rows to easier analysis

    if len(df) == 0:
        raise RuntimeError("No transitions were generated.")

    return df


# =============================================================================
# Write transition table
# =============================================================================
def dump_transition_table(df, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    preferred_cols = [
        "v_i", "J_i", "Omega_i", "Sigma_i", "Lambda_i", "parity_i", "index_i",
        "v_f", "J_f", "Omega_f", "Sigma_f", "Lambda_f", "parity_f", "index_f",
        "Ei_eV", "Ef_eV", "DeltaE_eV",
        "matrix_coh_real", "matrix_coh_imag", "matrix_coh_abs","matrix_coh_sum_abs",
        "Delta_v", "Delta_J", "Delta_Omega",
    ]

    # Using datafram and preferred columns
    cols = [col for col in preferred_cols if col in df.columns]
    extra = [col for col in df.columns if col not in cols]
    df_out = df[cols + extra]

    df_out.to_csv(filename, sep=" ", index=False, float_format="%.10e")

    print(f"Wrote: {filename}")
    print(f"# transitions: {len(df_out)}")


# =============================================================================
# Argument parser and main
# =============================================================================

def build_K_values(DJ_values):
    """
    Expands maximum DJ/K values into the allowed angular ranks.

    Example:
        -DJ 1.5      -> [0.5, 1.5]
        -DJ 2.5      -> [0.5, 1.5, 2.5]
        -DJ 0.5 1.5  -> [0.5, 1.5]
    """

    K_set = set()

    for DJ in DJ_values:
        if DJ < 0.5:
            raise ValueError(f"DJ must be >= 0.5, got {DJ}")

        nsteps = int(round(DJ - 0.5))

        if not np.isclose(0.5 + nsteps, DJ):
            raise ValueError(
                f"DJ={DJ} is not compatible with half-integer sequence "
                "0.5, 1.5, 2.5, ..."
            )

        for n in range(nsteps + 1):
            K_set.add(0.5 + n)

    return sorted(K_set)

#============================================================================
# Input file parser
#============================================================================

# Accounting for comments
def strip_inline_comment(line):
    stripped = line.strip()

    if not stripped:
        return ""

    if stripped.startswith("#") or stripped.startswith("!"):
        return ""

    clean = line.split("#", 1)[0]
    clean = clean.split("!", 1)[0]

    return clean.strip()

# Parsing boolean values
def parse_bool(value):
    value = str(value).strip().lower()
    if value in ["true", "t", "yes", "y", "1", ".true."]:
        return True
    if value in ["false", "f", "no", "n", "0", ".false."]:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")

# Parsing input file
def read_input_file(filename):
    """
    Reads a simple whitespace-separated input file.

    Example:
        ZPE 1187.278444217386 1177.879470966726
        Etot -341.492406252 -341.125731257
        nVJtot 36 81
        DJ 10.5
        Nvib 10 20
        NJ 15 25
        maskR 0.793755 1.8
        npts 10001
        path ./
        dsys PHGS/ PHMGS/
        dypath ./Dyson_Comp/
        states PHX3Sm PHMX2P
        output PHGSPHMGS.out
        dyson_mode quadrature
        spline cubic
        extrapolate false
        missing_dyson skip
        missing_coeff error
        min_intensity 0.0
    """

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
                raise ValueError(f"Missing value in input file {filename}, line {iline}: {line.rstrip()}")

            data[key] = values

    return data

# Accessing and uniforming keywords of input
def get_required(data, key):
    key = key.lower().replace("-", "_")
    if key not in data:
        raise KeyError(f"Missing required input keyword: {key}")
    return data[key]

def get_optional(data, key, default):
    key = key.lower().replace("-", "_")
    return data.get(key, default)

def input_data_to_namespace(data):
    args = argparse.Namespace()

    args.ZPE = [float(x) for x in get_required(data, "ZPE")]
    args.Etot = [float(x) for x in get_required(data, "Etot")]
    args.nVJtot = [int(x) for x in get_required(data, "nVJtot")]
    args.DJ = [float(x) for x in get_required(data, "DJ")]
    args.numvib = [int(x) for x in get_required(data, "Nvib")]
    args.numJ = [int(x) for x in get_optional(data, "NJ", [])] or None
    args.maskR = [float(x) for x in get_required(data, "maskR")]
    args.npts = int(get_required(data, "npts")[0])
    args.path = get_required(data, "path")[0]
    args.dsys = get_required(data, "dsys")
    args.pathtody = get_required(data, "dypath")[0]
    args.states = get_required(data, "states")
    args.output = get_optional(data, "output", [None])[0]

    args.dyson_mode = get_optional(data, "dyson_mode", ["quadrature"])[0]
    args.spline = get_optional(data, "spline", ["cubic"])[0]
    args.extrapolate = parse_bool(get_optional(data, "extrapolate", ["false"])[0])
    args.missing_dyson = get_optional(data, "missing_dyson", ["skip"])[0]
    args.missing_coeff = get_optional(data, "missing_coeff", ["error"])[0]
    args.min_intensity = float(get_optional(data, "min_intensity", ["0.0"])[0])

    # Checking correct number of parameters in file. 
    if len(args.ZPE) != 2:
        raise ValueError("ZPE must have exactly 2 values: neutral cation")
    if len(args.Etot) != 2:
        raise ValueError("Etot must have exactly 2 values: neutral cation")
    if len(args.nVJtot) != 2:
        raise ValueError("nVJtot must have exactly 2 values")
    if len(args.numvib) != 2:
        raise ValueError("Nvib must have exactly 2 values: neutral cation")
    if args.numJ is not None and len(args.numJ) != 2:
        raise ValueError("NJ must have exactly 2 values: neutral cation")
    if len(args.maskR) != 2:
        raise ValueError("maskR must have exactly 2 values: rmin rmax")
    if len(args.dsys) != 2:
        raise ValueError("dsys must have exactly 2 values: neutral cation")
    if len(args.states) != 2:
        raise ValueError("states must have exactly 2 values: neutral cation")

    if args.dyson_mode not in ["quadrature", "sum"]:
        raise ValueError("dyson_mode must be 'quadrature' or 'sum'")
    if args.spline not in ["cubic", "linear"]:
        raise ValueError("spline must be 'cubic' or 'linear'")
    if args.missing_dyson not in ["skip", "error"]:
        raise ValueError("missing_dyson must be 'skip' or 'error'")
    if args.missing_coeff not in ["skip", "error"]:
        raise ValueError("missing_coeff must be 'skip' or 'error'")

    return args

# Giving input file to argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build raw rovibronic photoionization transition table."
    )

    parser.add_argument("-i", "--input", type=str, default="photoion.inp", help="Input file. Default: photoion.inp")

    parsed = parser.parse_args()
    input_data = read_input_file(parsed.input)
    return input_data_to_namespace(input_data)

# Main
def main():

    # Reading input arg and information
    args = parse_args()

    base_path = Path(args.path)
    dyson_path = Path(args.pathtody)

    system_neutral, system_cation = args.dsys
    state_neutral, state_cation = args.states

    nvib_total, _ = args.nVJtot
    nvib_neutral, nvib_cation = args.numvib
    jmax_neutral, jmax_cation = (args.numJ if args.numJ is not None else (None, None))

    ZPE_neutral_cm, ZPE_cation_cm = args.ZPE
    Eelec_neutral_Eh, Eelec_cation_Eh = args.Etot
    rmin, rmax = args.maskR
    K_values = build_K_values(args.DJ)

    # Omega dyson combination generation - TODO: generalize it to all systems
    dyson_index_to_omega = build_dyson_index_to_omega()
    if state_neutral not in dyson_index_to_omega:
        raise KeyError(
            f"Unknown neutral state {state_neutral}. "
            f"Available: {list(dyson_index_to_omega)}"
        )
    if state_cation not in dyson_index_to_omega:
        raise KeyError(
            f"Unknown cation state {state_cation}. "
            f"Available: {list(dyson_index_to_omega)}"
        )

    # Output file
    if args.output is None:
        k_tag = "_K" + "-".join(f"{k:g}" for k in K_values)
        output_file = base_path / f"{state_neutral}_{state_cation}{k_tag}.dat"
    else:
        output_file = Path(args.output)

    # Reading vibrational eigenfunctions
    vib_neutral_file = resolve_system_file(base_path, system_neutral, "vibeigenvect_vib.chk")
    vib_cation_file = resolve_system_file(base_path, system_cation, "vibeigenvect_vib.chk")
    vib_neutral = parse_duo_vib_einfun(vib_neutral_file, args.npts, nvib_total)
    vib_cation = parse_duo_vib_einfun(vib_cation_file, args.npts, nvib_total)

    # Reading rovibrational levels
    levels_neutral_file = resolve_system_file(base_path, system_neutral, "rovibronic_energies.dat")
    levels_cation_file = resolve_system_file(base_path, system_cation, "rovibronic_energies.dat")
    levels_neutral = read_duo_levels(levels_neutral_file, system=system_neutral, nvib_max=nvib_neutral, j_max=jmax_neutral)
    levels_cation = read_duo_levels(levels_cation_file, system=system_cation, nvib_max=nvib_cation, j_max=jmax_cation)
    
    # Checking for empty rovibrational levels
    if len(levels_neutral) == 0:
        raise RuntimeError("No neutral DUO levels read.")
    if len(levels_cation) == 0:
        raise RuntimeError("No cation DUO levels read.")

    # Reading rovibrational coefficients
    coeff_neutral_file = resolve_system_file(base_path, system_neutral, "vibeigenvect_vectors.chk")
    coeff_cation_file = resolve_system_file(base_path, system_cation, "vibeigenvect_vectors.chk")
    coeff_neutral = read_coefficients(coeff_neutral_file)
    coeff_cation = read_coefficients(coeff_cation_file)

    # Reading R-grid
    rgrid_file = base_path / "Dipole_moment_functions.dat"
    if not rgrid_file.exists():
        raise FileNotFoundError(f"Missing R-grid file: {rgrid_file}")

    # Reading dyson values and creating splines
    rvals = np.loadtxt(rgrid_file, usecols=0)
    mask = (rvals > rmin) & (rvals < rmax)
    if np.sum(mask) == 0:
        raise RuntimeError(
            f"Empty R mask. Requested range: {rmin} - {rmax}. "
            f"Available range: {np.min(rvals)} - {np.max(rvals)}"
        )

    dyson_splines, _, _ = read_dyson_splines(d_neutral=dyson_index_to_omega[state_neutral], d_cation=dyson_index_to_omega[state_cation], 
                                            dyson_path=dyson_path, combine_mode=args.dyson_mode, spline_type=args.spline, bohr_to_angstrom=True, 
                                            extrapolate=args.extrapolate)

    # Building transition table
    df = build_transition_table(levels_neutral=levels_neutral, levels_cation=levels_cation, vib_neutral=vib_neutral, vib_cation=vib_cation, 
    coeff_neutral=coeff_neutral, coeff_cation=coeff_cation, dyson_splines=dyson_splines, rvals=rvals, mask=mask, 
    ZPE_neutral_cm=ZPE_neutral_cm, ZPE_cation_cm=ZPE_cation_cm, Eelec_neutral_Eh=Eelec_neutral_Eh, Eelec_cation_Eh=Eelec_cation_Eh, 
    K_values=K_values, missing_dyson=args.missing_dyson, missing_coeff=args.missing_coeff, min_intensity=args.min_intensity)

    dump_transition_table(df, output_file)

if __name__ == "__main__":
    main()