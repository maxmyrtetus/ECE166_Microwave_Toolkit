
"""
ECE166 Competency Project 
---------------------------------------

Goal: Demonstrate the *same* microwave-circuits skills that show up on ECE 166
style-questions (transmission lines, Smith chart movement, S-parameters, power waves,
couplers/hybrids, Friis noise, LNA matching/stability) in a single Python file.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Small helpers / formatting
# ---------------------------

def cpolar(z: complex) -> Tuple[float, float]:
    """Return (magnitude, angle_deg)."""
    mag = abs(z)
    ang = math.degrees(math.atan2(z.imag, z.real))
    return mag, ang


def fmt_complex(z: complex, nd: int = 4) -> str:
    """Pretty print complex number a+jb."""
    a = round(z.real, nd)
    b = round(z.imag, nd)
    sign = "+" if b >= 0 else "-"
    return f"{a} {sign} j{abs(b)}"


def db20(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-300))


def db10(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-300))


# -------------------------------------
# Smith chart math + a simple plotter
# -------------------------------------

def gamma_from_z(z_norm: complex) -> complex:
    """Reflection coefficient from normalized impedance z=Z/Z0."""
    return (z_norm - 1) / (z_norm + 1)


def z_from_gamma(gamma: complex) -> complex:
    """Normalized impedance z=Z/Z0 from reflection coefficient."""
    if abs(1 - gamma) < 1e-15:
        return complex(np.inf)
    return (1 + gamma) / (1 - gamma)


def draw_smith_chart(ax: plt.Axes,
                     r_list: Iterable[float] = (0, 0.2, 0.5, 1, 2, 5),
                     x_list: Iterable[float] = (-5, -2, -1, -0.5, -0.2, 0.2, 0.5, 1, 2, 5),
                     npts: int = 400) -> None:
    """
    Draw a *basic* Smith chart grid in the Gamma plane (unit circle).
    """
    # Unit circle
    th = np.linspace(0, 2*np.pi, npts)
    ax.plot(np.cos(th), np.sin(th), linewidth=1)

    # Axes
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)

    # Constant resistance circles
    xs = np.linspace(-50, 50, npts)  # sweep reactance
    for r in r_list:
        z = r + 1j*xs
        g = (z - 1)/(z + 1)
        ax.plot(g.real, g.imag, linewidth=0.3)

    # Constant reactance arcs
    rs = np.linspace(0, 50, npts)  # sweep resistance
    for x in x_list:
        z = rs + 1j*x
        g = (z - 1)/(z + 1)
        ax.plot(g.real, g.imag, linewidth=0.3)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Re{Γ}")
    ax.set_ylabel("Im{Γ}")
    ax.set_title("Smith Chart (Γ-plane)")


def smith_plot_points(ax: plt.Axes, gammas: List[complex], labels: List[str]) -> None:
    for g, lab in zip(gammas, labels):
        ax.plot([g.real], [g.imag], marker="o")
        ax.text(g.real + 0.02, g.imag + 0.02, lab)


# -----------------------------------------
# Transmission line propagation (ECE 166)
# -----------------------------------------

def propagate_gamma(gamma_load: complex,
                    length_wl: float,
                    alpha_nepers_per_wl: float = 0.0) -> complex:
    """
    Move a reflection coefficient "toward the generator" by length_wl (in wavelengths).

    Lossless line: alpha=0 -> Γ(z) rotates around origin.
    Lossy line: |Γ| shrinks by exp(-2*alpha*length).
    """
    beta = 2 * math.pi  # rad per wavelength
    prop = np.exp(-2 * (alpha_nepers_per_wl + 1j*beta) * length_wl)
    return gamma_load * prop


def tline_zin(Z0: float,
              ZL: complex,
              length_wl: float,
              alpha_nepers_per_wl: float = 0.0) -> complex:
    """
    Input impedance of a line of length length_wl terminated in ZL.

    Uses Γ-propagation (works for lossy/lossless uniformly).
    """
    zL = ZL / Z0
    gamma_L = gamma_from_z(zL)
    gamma_in = propagate_gamma(gamma_L, length_wl=length_wl, alpha_nepers_per_wl=alpha_nepers_per_wl)
    z_in = z_from_gamma(gamma_in)
    return z_in * Z0


def find_purely_resistive_points(Z0: float,
                                 ZL: complex,
                                 search_length_wl: float,
                                 ngrid: int = 5000,
                                 alpha_nepers_per_wl: float = 0.0,
                                 tol_imag_ohm: float = 1e-2) -> List[Tuple[float, float]]:
    """
    Find distances (in wavelengths) from the LOAD where Zin is (approximately) purely resistive.

    Returns list of (l_wl, R_ohm).
    """
    ls = np.linspace(0, search_length_wl, ngrid)
    vals = []
    for l in ls:
        Zin = tline_zin(Z0, ZL, l, alpha_nepers_per_wl=alpha_nepers_per_wl)
        if abs(Zin.imag) < tol_imag_ohm and np.isfinite(Zin.real):
            vals.append((float(l), float(Zin.real)))
    # De-duplicate clusters by simple spacing rule
    out: List[Tuple[float, float]] = []
    min_sep = search_length_wl / 50.0 if search_length_wl > 0 else 0.0
    for l, R in vals:
        if not out or abs(l - out[-1][0]) > min_sep:
            out.append((l, R))
    return out


# -----------------------------------------
# Problem 1: TL + shunt stub style workflow
# -----------------------------------------

@dataclasses.dataclass
class TLStubProblemConfig:
    Z0_main: float = 50.0
    ZL: complex = 20 - 1j*20

    # Main line segmentation (in wavelengths, moving from LOAD toward generator)
    l_load_to_A: float = 0.15
    l_A_to_B: float = 0.25
    l_B_to_in: float = 0.50

    # Shunt stub at node A
    Z0_stub: float = 50.0
    stub_type: str = "open"  # "open" or "short"
    l_stub: float = 0.07

    # Excitation
    V1_plus_Vpk: float = 10.0  # incident wave amplitude at the INPUT reference plane
    assume_lossless: bool = True

    # For part (d)-style loss: specify e^{-alpha * L_total} (one-way), for a lossy run
    # Set to 1.0 for lossless.
    one_way_attenuation: float = 1.0  # e^{-alpha * L_total}


def stub_input_impedance(Z0: float, length_wl: float, stub_type: str) -> complex:
    """
    Input impedance of an OPEN or SHORT-circuited lossless stub.

    Open stub:  Z_in = -j Z0 cot(beta l)
    Short stub: Z_in =  j Z0 tan(beta l)
    """
    beta_l = 2 * math.pi * length_wl
    # Avoid singularities
    eps = 1e-12
    if stub_type.lower() == "open":
        # cot(x) = cos/sin
        s = math.sin(beta_l)
        c = math.cos(beta_l)
        cot = c / (s if abs(s) > eps else (eps if s >= 0 else -eps))
        return -1j * Z0 * cot
    if stub_type.lower() == "short":
        t = math.tan(beta_l)
        return 1j * Z0 * t
    raise ValueError("stub_type must be 'open' or 'short'")


def run_problem_1(cfg: TLStubProblemConfig, outdir: str) -> None:
    print("\n" + "="*72)
    print("Problem 1-style: Transmission line + shunt stub computations")
    print("="*72)

    Z0 = cfg.Z0_main

    # Convert one-way attenuation e^{-alpha * L_total} into alpha (Nepers per wavelength)
    L_total = cfg.l_load_to_A + cfg.l_A_to_B + cfg.l_B_to_in
    if cfg.one_way_attenuation <= 0:
        raise ValueError("one_way_attenuation must be > 0")
    if abs(cfg.one_way_attenuation - 1.0) < 1e-12:
        alpha = 0.0
    else:
        alpha = -math.log(cfg.one_way_attenuation) / max(L_total, 1e-12)

    if cfg.assume_lossless:
        alpha_used = 0.0
    else:
        alpha_used = alpha

    # 1) ZL
    ZL = cfg.ZL
    zL = ZL / Z0
    gL = gamma_from_z(zL)

    # 2) Move from load to point A on main line (ignoring stub for this movement)
    ZA_from_load = tline_zin(Z0, ZL, cfg.l_load_to_A, alpha_nepers_per_wl=alpha_used)
    zA_from_load = ZA_from_load / Z0
    gA_from_load = gamma_from_z(zA_from_load)

    # 3) Stub input impedance at A (ZS)
    ZS = stub_input_impedance(cfg.Z0_stub, cfg.l_stub, cfg.stub_type)

    # 4) Combine at A (shunt): Z_A_total = (ZA_from_load || ZS)
    ZA_total = 1 / (1/ZA_from_load + 1/ZS)
    zA_total = ZA_total / Z0
    gA_total = gamma_from_z(zA_total)

    # 5) Move from A to B, then B to input
    ZB = tline_zin(Z0, ZA_total, cfg.l_A_to_B, alpha_nepers_per_wl=alpha_used)
    Zin = tline_zin(Z0, ZB, cfg.l_B_to_in, alpha_nepers_per_wl=alpha_used)

    g_in = gamma_from_z(Zin / Z0)

    # Power delivered into network at the input reference plane
    P_inc = (cfg.V1_plus_Vpk ** 2) / (2 * Z0)  # assuming V+ is Vpk
    P_del = P_inc * (1 - abs(g_in)**2)  # delivered into the network
    if cfg.assume_lossless:
        P_load = P_del
    else:
        P_load = float("nan")  # lossy case: can't say without modeling all losses
    print(f"Z0(main) = {Z0} Ω")
    print(f"ZL       = {fmt_complex(ZL)} Ω")
    print(f"ΓL       = {fmt_complex(gL)}  |ΓL|={cpolar(gL)[0]:.4f}, ∠={cpolar(gL)[1]:.2f}°")
    print(f"At point A (from load only): ZA_load = {fmt_complex(ZA_from_load)} Ω")
    print(f"Stub: type={cfg.stub_type}, l_stub={cfg.l_stub}λ, ZS = {fmt_complex(ZS)} Ω")
    print(f"Combined at A (shunt): ZA = {fmt_complex(ZA_total)} Ω")
    print(f"Point B: ZB = {fmt_complex(ZB)} Ω")
    print(f"Input:  Zin = {fmt_complex(Zin)} Ω")
    print(f"Γin      = {fmt_complex(g_in)}  |Γin|={cpolar(g_in)[0]:.4f}, ∠={cpolar(g_in)[1]:.2f}°")

    print(f"\nIncident power (from V1+): P_inc = {P_inc:.6g} W")
    print(f"Delivered into network:     P_del = {P_del:.6g} W")
    if cfg.assume_lossless:
        print(f"Lossless assumption => Power to load: P_load = {P_load:.6g} W")

    # "Where does load look purely resistive?" (search along first segment from the load)
    resistive_pts = find_purely_resistive_points(
        Z0=Z0, ZL=ZL, search_length_wl=cfg.l_load_to_A, alpha_nepers_per_wl=alpha_used
    )
    if resistive_pts:
        print("\nPurely-resistive points along the line segment from LOAD to A:")
        for l_wl, R_ohm in resistive_pts:
            if cfg.assume_lossless:
                Vpk = math.sqrt(max(0.0, 2 * P_load * R_ohm))
                print(f"  l = {l_wl:.4f} λ : R ≈ {R_ohm:.3f} Ω  =>  V_R,pk ≈ {Vpk:.4f} V")
            else:
                print(f"  l = {l_wl:.4f} λ : R ≈ {R_ohm:.3f} Ω")
    else:
        print("\n(No purely-resistive points found in the searched segment — adjust tolerance or length.)")

    # Smith chart plot of the requested points
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_smith_chart(ax)
    smith_plot_points(
        ax,
        gammas=[gL, gA_from_load, gamma_from_z(ZS / cfg.Z0_stub), gA_total, gamma_from_z(ZB / Z0), g_in],
        labels=["ZL", "ZA(load)", "ZS(stub)", "ZA(total)", "ZB", "Zin"],
    )
    fig.tight_layout()
    figpath = os.path.join(outdir, "problem1_smith_points.png")
    fig.savefig(figpath, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {figpath}")


# ------------------------------------------------------------
# Problem 2: 3-port RC-ish network used for S-parameter sweep
# ------------------------------------------------------------

@dataclasses.dataclass
class ThreePortRCConfig:
    """
    A symmetric 3-port network: a series capacitor from port1 to an internal node, and
    two equal resistors from that node to port2 and port3.

    Ports are referenced to Z0.
    """
    Z0: float = 50.0
    R: float = 50.0
    f_ref_hz: float = 1e9  # choose so that Xc = -Z0 at f_ref
    C: Optional[float] = None  # if None, choose C so that |Xc|=Z0 at f_ref

    # Sweep
    f_min_hz: float = 1e6
    f_max_hz: float = 1e12
    n_sweep: int = 400


def threeport_Y_matrix(freq_hz: float, cfg: ThreePortRCConfig) -> np.ndarray:
    """
    Build the 3x3 Y-parameter matrix (I = Y V) for the 3-port after eliminating
    the single internal node via KCL.
    """
    Z0 = cfg.Z0
    R = cfg.R
    if cfg.C is None:
        C = 1 / (2 * math.pi * cfg.f_ref_hz * Z0)  # makes Xc = -Z0 at f_ref
    else:
        C = cfg.C

    w = 2 * math.pi * max(freq_hz, 0.0)
    Yc = 1j * w * C
    Yr = 1 / R

    # For port voltages V1,V2,V3 and internal node Va:
    #   Va = (Yc V1 + Yr V2 + Yr V3) / (Yc + 2Yr)
    #   I1 = Yc (V1 - Va), I2 = Yr (V2 - Va), I3 = Yr (V3 - Va)
    #
    # Build Y numerically by exciting one port voltage at a time (basis vectors).
    Y = np.zeros((3, 3), dtype=complex)
    Vbasis = np.eye(3, dtype=complex)

    denom = (Yc + 2 * Yr)
    for k in range(3):
        V1, V2, V3 = Vbasis[:, k]
        if abs(denom) < 1e-30:
            Va = 0.0  # at DC, Yc=0 => internal node is floating from port1
        else:
            Va = (Yc * V1 + Yr * V2 + Yr * V3) / denom
        I1 = Yc * (V1 - Va)
        I2 = Yr * (V2 - Va)
        I3 = Yr * (V3 - Va)
        Y[:, k] = [I1, I2, I3]
    return Y


def Y_to_S(Y: np.ndarray, Z0: float) -> np.ndarray:
    """Convert Y-parameters to S for equal reference impedance Z0."""
    I = np.eye(Y.shape[0], dtype=complex)
    return (I - Z0 * Y) @ np.linalg.inv(I + Z0 * Y)


def run_problem_2(cfg: ThreePortRCConfig, outdir: str) -> None:
    print("\n" + "="*72)
    print("Problem 2-style: 3-port S-parameters vs frequency + time response")
    print("="*72)

    Z0 = cfg.Z0
    if cfg.C is None:
        C = 1 / (2 * math.pi * cfg.f_ref_hz * Z0)
    else:
        C = cfg.C

    print(f"Z0 = {Z0} Ω, R = {cfg.R} Ω")
    print(f"Chosen C = {C:.6g} F (so that Xc=-Z0 at f_ref={cfg.f_ref_hz:.3g} Hz)" if cfg.C is None
          else f"Using user-specified C = {C:.6g} F")

    # Evaluate S at f_ref (should match the clean closed forms seen in solutions)
    Yref = threeport_Y_matrix(cfg.f_ref_hz, cfg)
    Sref = Y_to_S(Yref, Z0)
    print("\nS(f_ref) =")
    for i in range(3):
        row = "  [" + ", ".join(fmt_complex(Sref[i, j]) for j in range(3)) + "]"
        print(row)

    # Power calculation for a1=1, a2=a3=0 (all ports matched)
    a = np.array([1+0j, 0+0j, 0+0j])
    b = Sref @ a
    P_in = float(np.sum(np.abs(a)**2))
    P_out = float(np.sum(np.abs(b)**2))
    P_diss = P_in - P_out  # power dissipated inside network (dimensionless with Z0 normalization)
    print("\nFor a1=1, a2=a3=0 (matched terminations):")
    print(f"  b = {b}")
    print(f"  Σ|a|^2 = {P_in:.6g},  Σ|b|^2 = {P_out:.6g},  dissipated = {P_diss:.6g}")

    # Frequency sweep for S11, S21, S22 on a Smith chart
    f_sweep = np.logspace(math.log10(cfg.f_min_hz), math.log10(cfg.f_max_hz), cfg.n_sweep)
    gam_S11 = []
    gam_S21 = []  # NOTE: S21 isn't a reflection coefficient, but we can still plot it in the complex plane
    gam_S22 = []
    for f in f_sweep:
        Y = threeport_Y_matrix(f, cfg)
        S = Y_to_S(Y, Z0)
        gam_S11.append(S[0, 0])
        gam_S21.append(S[1, 0])
        gam_S22.append(S[1, 1])

    # Smith chart plot (S11, S21, S22)
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_smith_chart(ax)
    ax.plot(np.real(gam_S11), np.imag(gam_S11), linewidth=1, label="S11(f)")
    ax.plot(np.real(gam_S21), np.imag(gam_S21), linewidth=1, label="S21(f)")
    ax.plot(np.real(gam_S22), np.imag(gam_S22), linewidth=1, label="S22(f)")
    ax.legend()
    fig.tight_layout()
    figpath = os.path.join(outdir, "problem2_s11_s22_smith.png")
    fig.savefig(figpath, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {figpath}")

    # Complex-plane plot of S21(f)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.real(gam_S21), np.imag(gam_S21), linewidth=1)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Re{S21}")
    ax.set_ylabel("Im{S21}")
    ax.set_title("S21(f) in complex plane")
    fig.tight_layout()
    figpath = os.path.join(outdir, "problem2_s21_complex.png")
    fig.savefig(figpath, dpi=200)
    plt.close(fig)
    print(f"Saved: {figpath}")

    # Time constant for step response (matched terminations)
    # Equivalent resistance seen by C (one side sees Z0, other side sees two branches (R+Z0) in parallel)
    Req_other_side = (cfg.R + Z0) / 2.0
    Req_total = Z0 + Req_other_side
    tau = Req_total * C

    print(f"\nTime constant estimate (matched ports):")
    print(f"  Req_other_side = (R+Z0)/2 = {Req_other_side:.4g} Ω")
    print(f"  Req_total      = Z0 + Req_other_side = {Req_total:.4g} Ω")
    print(f"  tau = Req_total * C = {tau:.6g} s")

    # Step response for b1(t), b2(t) with a1 = u(t).
    # Logic:
    #   at t=0+  capacitor ~ short  => use S(high freq)
    #   at t→∞   capacitor ~ open   => use S(0)
    # and a single-pole exponential with time constant tau.
    S_low = Y_to_S(threeport_Y_matrix(0.0, cfg), Z0)                 # DC
    S_high = Y_to_S(threeport_Y_matrix(cfg.f_max_hz, cfg), Z0)       # "very high f" approximation
    b1_0 = (S_high @ a)[0]
    b2_0 = (S_high @ a)[1]
    b1_inf = (S_low @ a)[0]
    b2_inf = (S_low @ a)[1]

    t = np.linspace(0, 6*tau, 600)
    b1_t = b1_inf + (b1_0 - b1_inf) * np.exp(-t / tau)
    b2_t = b2_inf + (b2_0 - b2_inf) * np.exp(-t / tau)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, np.real(b1_t), label="Re{b1(t)}")
    ax.plot(t, np.real(b2_t), label="Re{b2(t)}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("amplitude (real part)")
    ax.set_title("Step response with a1=u(t) (matched ports)")
    ax.legend()
    fig.tight_layout()
    figpath = os.path.join(outdir, "problem2_step_response.png")
    fig.savefig(figpath, dpi=200)
    plt.close(fig)
    print(f"Saved: {figpath}")


# -----------------------------------------
# Problem 3: Friis gain + noise temperature
# -----------------------------------------

@dataclasses.dataclass
class FriisChainConfig:
    """
    Simple cascade of blocks, each with gain (linear) and noise factor F (linear).
    """
    blocks: List[Tuple[str, float, float]] = dataclasses.field(default_factory=lambda: [
        ("Amp1", 10**(10/10), 10**(2/10)),   # 10 dB gain, 2 dB NF
        ("Amp2", 10**(20/10), 10**(3/10)),   # 20 dB gain, 3 dB NF
    ])
    source_temp_K: float = 290.0
    bandwidth_hz: float = 20e6


def friis_noise_factor(blocks: List[Tuple[str, float, float]]) -> Tuple[float, float]:
    """
    Return (F_total, G_total) for cascade.
    blocks: (name, G_linear, F_linear)
    """
    F_total = 0.0
    G_prev = 1.0
    for i, (_, G, F) in enumerate(blocks):
        if i == 0:
            F_total = F
        else:
            F_total += (F - 1) / G_prev
        G_prev *= G
    G_total = G_prev
    return F_total, G_total


def run_problem_3(cfg: FriisChainConfig) -> None:
    print("\n" + "="*72)
    print("Problem 3-style: Friis gain, NF, equivalent noise temperature, output noise power")
    print("="*72)

    Ftot, Gtot = friis_noise_factor(cfg.blocks)
    NFtot_db = db10(Ftot)
    Gtot_db = db10(Gtot)
    T0 = cfg.source_temp_K
    Teq = (Ftot - 1) * T0

    print("Blocks:")
    for name, G, F in cfg.blocks:
        print(f"  {name:>6s}: G={db10(G):.2f} dB,  F={db10(F):.2f} dB")
    print(f"\nTotal gain: {Gtot_db:.3f} dB (linear {Gtot:.6g})")
    print(f"Total NF:   {NFtot_db:.3f} dB (F={Ftot:.6g})")
    print(f"Equivalent noise temperature Teq = (F-1)T0 = {Teq:.3f} K")

    # Output noise power (k * Tsys * B * Gtotal)
    kB = 1.380649e-23
    Tsys = T0 + Teq
    Pn_out = kB * Tsys * cfg.bandwidth_hz * Gtot
    print(f"\nAssuming bandwidth B={cfg.bandwidth_hz:.3g} Hz:")
    print(f"  Tsys = T0 + Teq = {Tsys:.3f} K")
    print(f"  Output noise power ≈ k*Tsys*B*G = {Pn_out:.6g} W = {10*math.log10(Pn_out/1e-3):.3f} dBm")


# --------------------------------------------------
# Problem 4: Couplers / hybrids / Wilkinson patterns
# --------------------------------------------------

def S_3db_hybrid() -> np.ndarray:
    """
    Ideal 90° 3-dB hybrid:

    - Excite port 1: outputs at ports 2 and 4 with a 90° phase difference
    - Port 3 is isolated from port 1
    - All ports matched
:
      S11=S22=S33=S44=0,  S21=-j/√2,  S41=-1/√2,  S31=0
    """
    j = 1j
    s = 1 / math.sqrt(2)
    S = np.array([
        [0, -j*s, 0, -s],
        [-j*s, 0, -s, 0],
        [0, -s, 0, -j*s],
        [-s, 0, -j*s, 0],
    ], dtype=complex)
    return S


def S_coupled_line(c: float) -> np.ndarray:
    """
    Ideal matched 4-port directional coupler with coupling coefficient c in [0,1].

    Conventions:
      - Port 1 input couples to port 2 (through) and port 3 (coupled)
      - Port 4 is isolated from port 1

      S11=0, S21=-j*sqrt(1-c^2), S31=c, S41=0
    """
    if not (0 <= c <= 1):
        raise ValueError("c must be in [0,1]")
    t = math.sqrt(1 - c**2)
    j = 1j
    S = np.array([
        [0, -j*t, c, 0],
        [-j*t, 0, 0, c],
        [c, 0, 0, -j*t],
        [0, c, -j*t, 0],
    ], dtype=complex)
    return S


def S_wilkinson_ideal() -> np.ndarray:
    """
    Ideal equal-split Wilkinson divider *as a 3-port* (matched, isolated outputs).

    Conventions used here:
      - S11=S22=S33=0  (all ports matched)
      - S21=S31=-j/√2  (equal split with a -90° phase)
      - S23=S32=0      (output ports isolated)

    Note: A 3-port that is matched + isolated cannot be lossless (it must dissipate power
    in the internal resistor for certain excitations). That is why we compute the resistor power as:
        P_diss = Σ|a|^2 - Σ|b|^2
    """
    j = 1j
    s = 1 / math.sqrt(2)
    S = np.array([
        [0, -j*s, -j*s],
        [-j*s, 0, 0],
        [-j*s, 0, 0],
    ], dtype=complex)
    return S


def run_problem_4_demo() -> None:
    print("\n" + "="*72)
    print("Problem 4-style: Coupler / hybrid / Wilkinson quick checks")
    print("="*72)

    # Example A: 3-dB hybrid with a1=1
    Sh = S_3db_hybrid()
    a = np.array([1+0j, 0+0j, 0+0j, 0+0j])
    b = Sh @ a
    print("\n3-dB Hybrid example (a1=1):")
    print(f"  b = {b}")
    print(f"  |b2|={abs(b[1]):.4f}, |b4|={abs(b[3]):.4f}, b3 (isolated)={b[2]}")

    # Example B: Coupled-line coupler sanity check
    c = 0.6
    Sc = S_coupled_line(c)
    b2 = (Sc @ a)[1]
    b3 = (Sc @ a)[2]
    print("\nCoupled-line coupler example (c=0.6, a1=1):")
    print(f"  through (b2) = {fmt_complex(b2)}  |b2|={abs(b2):.4f}")
    print(f"  coupled (b3) = {fmt_complex(b3)}  |b3|={abs(b3):.4f}")

    # Example C: Wilkinson: compute dissipated power for a1=1, others 0
    Sw = S_wilkinson_ideal()
    a3 = np.array([1+0j, 0+0j, 0+0j])
    b3 = Sw @ a3
    P_in = float(np.sum(np.abs(a3)**2))
    P_out = float(np.sum(np.abs(b3)**2))
    P_diss = P_in - P_out
    print("\nWilkinson (3-port ideal model), a1=1:")
    print(f"  b = {b3}")
    print(f"  Σ|a|^2={P_in:.4f}, Σ|b|^2={P_out:.4f} => dissipated={P_diss:.4f} (power in internal resistor)")


    # Example D: Wilkinson + amplifier (power dissipated in the Wilkinson resistor for an unbalanced case)
    #
    # Ideal amplifier S-parameters (typical in exam problems):
    #   S11=0.2, S21=10, S12=0, S22=0  (matched output, unilateral)
    S11_amp = 0.2 + 0j
    S21_amp = 10.0 + 0j
    S12_amp = 0.0 + 0j
    S22_amp = 0.0 + 0j

    # a1=1 at Wilkinson port 1
    a1 = 1 + 0j

    # Divider waves from port1 excitation (outputs are isolated/matched in this ideal model)
    b2_wk = Sw[1, 0] * a1
    b3_wk = Sw[2, 0] * a1

    # Amplifier produces reflected wave at its input (back into Wilkinson port2)
    a2 = S11_amp * b2_wk + S12_amp * b3_wk

    # Amplifier produces a wave at its output (back into Wilkinson port3)
    a3 = S21_amp * b2_wk + S22_amp * b3_wk

    a_vec = np.array([a1, a2, a3], dtype=complex)
    b_vec = Sw @ a_vec
    b1 = b_vec[0]

    P_in = float(np.sum(np.abs(a_vec) ** 2))
    P_out = float(np.sum(np.abs(b_vec) ** 2))
    P_diss = P_in - P_out

    print("\nWilkinson + amplifier example (a1=1, S11_amp=0.2, S21_amp=10):")
    print(f"  a = {a_vec}")
    print(f"  b = {b_vec}")
    print(f"  b1 = {fmt_complex(b1)}  |b1|={abs(b1):.4f}")
    print(f"  Resistor power = Σ|a|^2 - Σ|b|^2 = {P_diss:.4f} (normalized, |a1|^2=1)")


# -----------------------------------------------------
# Problem 5: LNA basics (unilateral, stability, match)
# -----------------------------------------------------

@dataclasses.dataclass
class LNAConfig:
    Z0: float = 50.0
    f0_hz: float = 6e9

    # Example transistor S-parameters at f0 
    S11: complex = 0.2 * np.exp(1j * math.radians(-30))
    S21: complex = 8.0 * np.exp(1j * math.radians(80))    # gain
    S12: complex = 0.03 * np.exp(1j * math.radians(10))
    S22: complex = 0.4 * np.exp(1j * math.radians(-90))

    # Optimum source reflection coefficient for minimum NF (Γopt)
    Gamma_opt: complex = 0.62 * np.exp(1j * math.radians(20))

    # Stability check frequency (often 1 GHz in finals)
    f_stab_hz: float = 1e9
    # If don't have S-params at f_stab, reuse f0 values as a placeholder
    reuse_f0_for_stability: bool = True


def unilateral_figure_of_merit(S11: complex, S21: complex, S12: complex, S22: complex) -> float:
    """
    Common 'unilateral figure of merit' used in RF design classes:
        U = |S12||S21||S11||S22| / ((1-|S11|^2)(1-|S22|^2))
    """
    num = abs(S12) * abs(S21) * abs(S11) * abs(S22)
    den = (1 - abs(S11)**2) * (1 - abs(S22)**2)
    return num / den if den > 0 else float("inf")


def stability_factors(S11: complex, S21: complex, S12: complex, S22: complex) -> Tuple[complex, float]:
    """
    Return (Delta, K) where:
        Δ = S11*S22 - S12*S21
        K = (1 - |S11|^2 - |S22|^2 + |Δ|^2) / (2 |S12 S21|)
    Unconditional stability if K>1 and |Δ|<1.
    """
    Delta = S11 * S22 - S12 * S21
    denom = 2 * abs(S12 * S21)
    if denom < 1e-30:
        K = float("inf")
    else:
        K = (1 - abs(S11)**2 - abs(S22)**2 + abs(Delta)**2) / denom
    return Delta, K


def gamma_out(S11: complex, S21: complex, S12: complex, S22: complex, gamma_S: complex) -> complex:
    """Γ_out looking into port 2 with source reflection Γ_S at port 1."""
    return S22 + (S12 * S21 * gamma_S) / (1 - S11 * gamma_S)


def gamma_in(S11: complex, S21: complex, S12: complex, S22: complex, gamma_L: complex) -> complex:
    """Γ_in looking into port 1 with load reflection Γ_L at port 2."""
    return S11 + (S12 * S21 * gamma_L) / (1 - S22 * gamma_L)




def run_problem_5(cfg: LNAConfig, outdir: str) -> None:
    print("\n" + "="*72)
    print("Problem 5-style: Unilateral check, stability, Γopt match, Γout/ΓL, S11Amp")
    print("="*72)

    Z0 = cfg.Z0
    S11, S21, S12, S22 = cfg.S11, cfg.S21, cfg.S12, cfg.S22

    U = unilateral_figure_of_merit(S11, S21, S12, S22)
    print(f"Unilateral figure-of-merit U = {U:.4g}  ({'OK' if U < 0.1 else 'not very unilateral'})")

    Delta, K = stability_factors(S11, S21, S12, S22)
    print(f"At f0={cfg.f0_hz:.3g} Hz:")
    print(f"  Δ = {fmt_complex(Delta)}  |Δ|={abs(Delta):.4f}")
    print(f"  K = {K:.4g}  => {'unconditionally stable' if (K>1 and abs(Delta)<1) else 'potentially unstable'}")

    # Noise-match: ΓS = Γopt
    gamma_S = cfg.Gamma_opt
    Zs_opt = Z0 * (1 + gamma_S) / (1 - gamma_S)
    print(f"\nΓopt = {fmt_complex(gamma_S)}  (|Γ|={abs(gamma_S):.3f}, ∠={cpolar(gamma_S)[1]:.2f}°)")
    print(f"Equivalent source impedance for min NF: Zs = {fmt_complex(Zs_opt)} Ω")

    # Output side: assume output is conjugately matched to Γout
    gout = gamma_out(S11, S21, S12, S22, gamma_S)
    gL = np.conjugate(gout)
    print(f"\nΓout (with ΓS=Γopt) = {fmt_complex(gout)}  (|Γ|={abs(gout):.3f})")
    print(f"Assuming conjugate output match: ΓL = Γout* = {fmt_complex(gL)}")

    gin = gamma_in(S11, S21, S12, S22, gL)
    Zin_trans = Z0 * (1 + gin) / (1 - gin)
    print(f"\nΓin (with ΓL matched) = {fmt_complex(gin)}  (|Γ|={abs(gin):.3f})")
    print(f"Input impedance of transistor under output match: Zin = {fmt_complex(Zin_trans)} Ω")

    # Design a simple LC L-match from 50Ω to Zs_opt at f0
    # (This is the "Ms for min NF" step.)
    print("\nDesigning a simple L-match from 50Ω to Zs (min NF target) ...")
    # Keep this practical: we use a standard approach:
    # - cancel imag part with one series element
    # - match resistances with series+shunt
    try:
        # We'll implement a very small brute-force approach to avoid sign mistakes:
        Lmatch = design_L_match_bruteforce(Z0=Z0, Z_target=Zs_opt, f0=cfg.f0_hz)
        print("Selected L-match topology:", Lmatch["topology"])
        for part in Lmatch["parts"]:
            print(f"  {part}")
        # Compute overall S11Amp: reflection at the source plane looking into Ms+transistor (with ΓL matched)
        g_total = input_reflection_through_Lmatch(
            Z0=Z0,
            Z_load=Zin_trans,
            f0=cfg.f0_hz,
            Lmatch=Lmatch,
        )
        print(f"\nOverall input reflection (S11Amp) ≈ Γ = {fmt_complex(g_total)}  |Γ|={abs(g_total):.3f}, ∠={cpolar(g_total)[1]:.2f}°")

    except Exception as e:
        print("L-match synthesis failed:", e)

    # Optional: plot Γopt point on Smith chart
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_smith_chart(ax)
    smith_plot_points(ax, [gamma_S, gin], ["Γopt", "Γin(transistor)"])
    fig.tight_layout()
    figpath = os.path.join(outdir, "problem5_gamma_points.png")
    fig.savefig(figpath, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {figpath}")

    # Stability at 1 GHz (if requested)
    if cfg.f_stab_hz is not None:
        if cfg.reuse_f0_for_stability:
            S11s, S21s, S12s, S22s = S11, S21, S12, S22
            note = "(reusing f0 S-params as placeholder)"
        else:
            # If have real data at f_stab, set it here.
            S11s, S21s, S12s, S22s = S11, S21, S12, S22
            note = "(no separate data provided)"
        Delta_s, K_s = stability_factors(S11s, S21s, S12s, S22s)
        print(f"\nStability check at f={cfg.f_stab_hz:.3g} Hz {note}:")
        print(f"  |Δ|={abs(Delta_s):.4f}, K={K_s:.4g} => {'stable' if (K_s>1 and abs(Delta_s)<1) else 'not unconditionally stable'}")


# ------------------------------------------------------------------
# Practical L-match synthesis 
# ------------------------------------------------------------------

def series_Z_of_part(part: Dict, w: float) -> complex:
    """Impedance of a series component dict at angular frequency w."""
    kind = part["kind"]
    val = part["value"]
    if kind == "L":
        return 1j*w*val
    if kind == "C":
        return -1j/(w*val)
    raise ValueError("unknown part kind")


def shunt_Y_of_part(part: Dict, w: float) -> complex:
    """Admittance of a shunt component dict at angular frequency w."""
    kind = part["kind"]
    val = part["value"]
    if kind == "L":
        Z = 1j*w*val
        return 1/Z
    if kind == "C":
        Z = -1j/(w*val)
        return 1/Z
    raise ValueError("unknown part kind")


def apply_Lmatch(Z_load: complex, Lmatch: Dict, w: float) -> complex:
    """
    Given a load impedance at the *right* side of the match network, compute
    the input impedance seen at the *left* side (source side).
    """
    Z = Z_load
    for elem in reversed(Lmatch["elements"]):
        if elem["position"] == "series":
            Z = Z + series_Z_of_part(elem, w)
        elif elem["position"] == "shunt":
            Y = 1/Z + shunt_Y_of_part(elem, w)
            Z = 1/Y
        else:
            raise ValueError("bad position")
    return Z


def input_reflection_through_Lmatch(Z0: float, Z_load: complex, f0: float, Lmatch: Dict) -> complex:
    """Return Γ seen looking into the L-match when terminated by Z_load."""
    w = 2*math.pi*f0
    Zin = apply_Lmatch(Z_load, Lmatch, w)
    return (Zin - Z0) / (Zin + Z0)


def design_L_match_bruteforce(Z0: float, Z_target: complex, f0: float) -> Dict:
    """
    Small brute-force L-match search:
      Try 4 classic 2-element L topologies (series+shunt orders and L/C choices)
      and pick the one that best matches Z0 to Z_target at f0.
    """
    w = 2*math.pi*f0

    # Candidate element magnitude grids (reactance magnitudes) in ohms
    X_grid = np.logspace(math.log10(0.5), math.log10(500), 400)  # 0.5Ω .. 500Ω

    # Build candidate element dicts for a given reactance magnitude
    def series_elem(kind: str, Xmag: float) -> Dict:
        if kind == "L":
            return {"position": "series", "kind": "L", "value": Xmag / w}
        else:
            return {"position": "series", "kind": "C", "value": 1/(w*Xmag)}

    def shunt_elem(kind: str, Xmag: float) -> Dict:
        # Shunt element is easier specified by its *reactance magnitude* at f0.
        if kind == "L":
            return {"position": "shunt", "kind": "L", "value": Xmag / w}
        else:
            return {"position": "shunt", "kind": "C", "value": 1/(w*Xmag)}

    topologies = [
        ("series_then_shunt", ["series", "shunt"]),
        ("shunt_then_series", ["shunt", "series"]),
    ]

    # We'll test 4 combos: series(L/C) and shunt(L/C)
    kinds = ["L", "C"]

    best = None
    best_err = float("inf")

    for topo_name, positions in topologies:
        for k_series in kinds:
            for k_shunt in kinds:
                # grid search X magnitudes
                for Xs in X_grid[::2]:   # reduce combos a bit
                    for Xp in X_grid[::2]:
                        elems = []
                        for pos in positions:
                            if pos == "series":
                                elems.append(series_elem(k_series, Xs))
                            else:
                                elems.append(shunt_elem(k_shunt, Xp))
                        cand = {"topology": topo_name, "elements": elems}

                        Zin = apply_Lmatch(Z_target, cand, w)
                        err = abs(Zin - Z0)
                        if err < best_err:
                            best_err = err
                            best = cand

    if best is None:
        raise RuntimeError("No match found")

    # Convert to a friendlier print format
    parts = []
    for e in best["elements"]:
        if e["kind"] == "L":
            parts.append(f"{e['position']:>6s} L = {e['value']:.6g} H")
        else:
            parts.append(f"{e['position']:>6s} C = {e['value']:.6g} F")

    return {"topology": best["topology"], "elements": best["elements"], "parts": parts, "match_error_ohm": best_err}


# -----------------
# Main entry point
# -----------------

def main() -> None:
    outdir = "ece166_outputs"
    os.makedirs(outdir, exist_ok=True)

    # Problem 1 demo
    cfg1 = TLStubProblemConfig(
        Z0_main=50.0,
        ZL=20 - 1j*20,
        l_load_to_A=0.15,
        l_A_to_B=0.25,
        l_B_to_in=0.50,
        Z0_stub=50.0,
        stub_type="open",
        l_stub=0.07,
        V1_plus_Vpk=10.0,
        assume_lossless=True,
        one_way_attenuation=1.0,  # set to 0.71 and assume_lossless=False to mimic the "lossy line" part
    )
    run_problem_1(cfg1, outdir=outdir)

    # Problem 2 demo
    cfg2 = ThreePortRCConfig(
        Z0=50.0,
        R=50.0,
        f_ref_hz=1e9,   # makes Zc=-jZ0 at f_ref if C is None
        C=None,
        f_min_hz=1e6,
        f_max_hz=1e12,
        n_sweep=400,
    )
    run_problem_2(cfg2, outdir=outdir)

    # Problem 3 demo
    cfg3 = FriisChainConfig(
        blocks=[
            ("Amp1", 10**(10/10), 10**(2/10)),
            ("Att",  10**(-3/10), 10**(3/10)),  # example attenuator: -3 dB gain, NF=3 dB
            ("Amp2", 10**(20/10), 10**(3/10)),
        ],
        source_temp_K=290.0,
        bandwidth_hz=20e6,
    )
    run_problem_3(cfg3)

    # Problem 4 quick demo
    run_problem_4_demo()

    # Problem 5 demo
    cfg5 = LNAConfig()
    run_problem_5(cfg5, outdir=outdir)

    print("\nDone. Check the ./ece166_outputs/ folder for plots.")


if __name__ == "__main__":
    main()
