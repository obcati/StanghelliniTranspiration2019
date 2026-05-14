"""
Stanghellini Transpiration Model

dilopezv@unal.edu.co
=================================
Reference
---------
Stanghellini, C., van 't Ooster, A., & Heuvelink, E. (2024).
    Greenhouse horticulture: Technology for optimal crop production (2nd ed.).
    Brill Nijhoff. https://doi.org/10.1163/9789004697041

Model Description
-----------------
Canopy transpiration per unit ground area E [g m⁻² s⁻¹]:

    E = 2·LAI / [(1+ε)·r_b + r_s]  ·  [VPD  +  (ε·r_b / 2·LAI) · (R_n / L)]

Symbols
    LAI   m² m⁻²   Leaf Area Index
    ε     –        Latent-to-sensible heat ratio of saturated air:
                       ε = 0.7156 · exp(0.0533 · Tₐ)
    VPD   g m⁻³   Absolute vapour pressure deficit
    L     J g⁻¹   Latent heat of vaporisation:  L = 2501 − 2.361·Tₐ
    r_b   s m⁻¹   Leaf boundary-layer resistance (fixed = 200 s m⁻¹)
    r_s   s m⁻¹   Stomatal resistance:
                       r_s = 82·[1 + 6.95·exp(−0.4·I_sun/LAI)]·[1 + 0.023·(Tₐ+20)²]
    R_n   W m⁻²   Net crop radiation:
                       R_n = 0.86·(1 + exp(−0.7·LAI))·I_sun

Note: E [g m⁻² s⁻¹] × 3.6 = E [mm h⁻¹]  (water density 1000 kg m⁻³).

The model is evaluated for LAI ∈ {0.5, 1, 2, 3, 4}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Physical / thermodynamic constants
# ─────────────────────────────────────────────────────────────────────────────
M_WATER = 18.016      # Molar mass of water                       [g mol⁻¹]
R_GAS   = 8.314       # Universal gas constant                    [J mol⁻¹ K⁻¹]

# ─────────────────────────────────────────────────────────────────────────────
# Model parameters  (Stanghellini et al. 2024, greenhouse tomato)
# ─────────────────────────────────────────────────────────────────────────────
R_B = 200.0           # Leaf boundary-layer resistance (fixed)    [s m⁻¹]

# LAI values to evaluate
LAI_VALUES = [0.5, 1.0, 2.0, 3.0, 4.0]   # [m² m⁻²]


# ─────────────────────────────────────────────────────────────────────────────
# Thermodynamic helpers
# ─────────────────────────────────────────────────────────────────────────────

def sat_vapour_pressure(T_C: np.ndarray | float) -> np.ndarray | float:
    """Saturation vapour pressure e_s [Pa] — Buck (1981)."""
    return 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))


def vpd_absolute(T_C: np.ndarray | float,
                 RH:  np.ndarray | float) -> np.ndarray | float:
    """
    Absolute vapour pressure deficit VPD [g m⁻³].

    Derived from the ideal gas law:
        VPD_abs = VPD_Pa · M_water / (R · T_K)
    """
    e_s   = sat_vapour_pressure(T_C)
    VPD_Pa = e_s * (1.0 - RH)                        # [Pa]
    T_K   = T_C + 273.15
    return VPD_Pa * M_WATER / (R_GAS * T_K)           # [g m⁻³]


def latent_heat_g(T_C: np.ndarray | float) -> np.ndarray | float:
    """Latent heat of vaporisation L [J g⁻¹], temperature-dependent."""
    return 2501.0 - 2.361 * T_C


# ─────────────────────────────────────────────────────────────────────────────
# Stanghellini et al. (2024) model sub-functions
# ─────────────────────────────────────────────────────────────────────────────

def epsilon_ratio(Ta: np.ndarray | float) -> np.ndarray | float:
    """
    ε — ratio of the latent to sensible heat content of saturated air
    for a 1 °C change in temperature (Stanghellini et al. 2024).

        ε = 0.7156 · exp(0.0533 · Tₐ)
    """
    return 0.7156 * np.exp(0.0533 * Ta)


def net_radiation(I_sun: np.ndarray | float,
                 LAI: float) -> np.ndarray | float:
    """
    Net radiation of the crop R_n [W m⁻²] (Stanghellini et al. 2024).

        R_n = 0.86 · (1 + exp(−0.7·LAI)) · I_sun
    """
    return 0.86 * (1.0 + np.exp(-0.7 * LAI)) * I_sun


def stomatal_resistance(I_sun: np.ndarray | float,
                        Ta:    np.ndarray | float,
                        LAI:   float) -> np.ndarray | float:
    """
    Stomatal resistance r_s [s m⁻¹] (Stanghellini et al. 2024).

        r_s = 82 · [1 + 6.95·exp(−0.4·I_sun/LAI)] · [1 + 0.023·(Tₐ-20)²]

    The first factor accounts for radiation (I_sun/LAI = mean irradiance
    per unit leaf area); the second accounts for temperature.
    """
    f_rad  = 1.0 + 6.95 * np.exp(-0.4 * I_sun / np.maximum(LAI, 1e-6))
    f_temp = 1.0 + 0.023 * (Ta - 20.0) ** 2
    return 82.0 * f_rad * f_temp


# ─────────────────────────────────────────────────────────────────────────────
# Main transpiration function
# ─────────────────────────────────────────────────────────────────────────────

def stanghellini_transpiration(
    Ta:    np.ndarray | float,
    RH:    np.ndarray | float,
    I_sun: np.ndarray | float,
    LAI:   float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Canopy transpiration rate E [g m⁻² s⁻¹] — Stanghellini et al. (2024).

        E = 2·LAI / [(1+ε)·r_b + r_s]  ·  [VPD + (ε·r_b / 2·LAI)·(R_n/L)]

    Parameters
    ----------
    Ta    : air temperature              [°C]
    RH    : relative humidity (0–1)      [-]
    I_sun : global solar radiation       [W m⁻²]
    LAI   : Leaf Area Index              [m² m⁻²]

    Returns
    -------
    E   : transpiration rate             [g m⁻² s⁻¹]
    r_s : stomatal resistance            [s m⁻¹]
    R_n : net crop radiation             [W m⁻²]
    """
    eps  = epsilon_ratio(Ta)                          # [-]
    L    = latent_heat_g(Ta)                          # [J g⁻¹]
    VPD  = vpd_absolute(Ta, RH)                       # [g m⁻³]
    R_n  = net_radiation(I_sun, LAI)                  # [W m⁻²]
    r_s  = stomatal_resistance(I_sun, Ta, LAI)        # [s m⁻¹]

    # R_n/L : [W m⁻²] / [J g⁻¹] = [J s⁻¹ m⁻²] / [J g⁻¹] = [g m⁻² s⁻¹]
    rad_term = (eps * R_B / (2.0 * LAI)) * (R_n / L)  # [g m⁻³]

    numerator   = 2.0 * LAI * (VPD + rad_term)        # [m² m⁻²] × [g m⁻³]
    denominator = (1.0 + eps) * R_B + r_s              # [s m⁻¹]

    E = numerator / denominator                        # [g m⁻² s⁻¹]
    E = np.maximum(E, 0.0)                             # transpiration ≥ 0

    return E, r_s, R_n


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_greenhouse_data(filepath: Path) -> pd.DataFrame:
    """
    Load the greenhouse weather CSV.

    Handles European decimal-comma notation (e.g. "12,3" → 12.3) and
    parses the US-style timestamp (M/D/YY H:MM).
    """
    df = pd.read_csv(filepath, dtype=str)

    # Parse timestamp  (M/D/YY H:MM  — US format, e.g. "3/19/25 0:00")
    df["datetime"] = pd.to_datetime(
        df["Timestamp"], format="%m/%d/%y %H:%M", errors="coerce"
    )

    # Convert European comma decimals to float for all numeric columns
    # Exclude Timestamp and the newly-created datetime column
    skip = {"Timestamp", "datetime"}
    numeric_cols = [c for c in df.columns if c not in skip]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Rename columns to short, code-friendly names
    df = df.rename(
        columns={
            "°C Air Temperature":       "T_air",
            "RH Relative Humidity":     "RH",
            "kPa Atmospheric Pressure": "P_kPa",
            "W/m² Solar Radiation":     "I_global",
            "µmol·m⁻²·s⁻¹ PPFD":       "PPFD",
            "m/s Wind Speed":           "u",
        }
    )

    required = ["datetime", "T_air", "RH", "P_kPa", "I_global", "u"]
    df = df.dropna(subset=required).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid records found after parsing. "
            "Check that the CSV column names match those expected by the loader."
        )

    # Guard against physically implausible sensor values
    df = df[
        (df["T_air"].between(-10, 60))
        & (df["RH"].between(0.0, 1.0))
        & (df["P_kPa"] > 0)
        & (df["I_global"] >= 0)
        & (df["u"] >= 0)
    ].reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Run model — multiple LAI values
# ─────────────────────────────────────────────────────────────────────────────

def run_model(
    df: pd.DataFrame,
    lai_values: list[float] = LAI_VALUES,
) -> dict[float, pd.DataFrame]:
    """
    Apply the Stanghellini et al. (2024) model for each LAI in *lai_values*.

    Returns a dict  {LAI_value: result_DataFrame}  where each DataFrame
    contains the base weather columns plus:
        VPD_g_m3  : absolute vapour pressure deficit    [g m⁻³]
        R_n       : net crop radiation                  [W m⁻²]
        r_s       : stomatal resistance                 [s m⁻¹]
        E_g_m2_s  : transpiration rate                  [g m⁻² s⁻¹]
        E_mm_h    : transpiration rate (water depth)    [mm h⁻¹]
    """
    results: dict[float, pd.DataFrame] = {}
    for lai in lai_values:
        out = df.copy()
        out["VPD_g_m3"] = vpd_absolute(df["T_air"].values, df["RH"].values)

        E, r_s, R_n = stanghellini_transpiration(
            Ta    = df["T_air"].values,
            RH    = df["RH"].values,
            I_sun = df["I_global"].values,
            LAI   = lai,
        )
        out["R_n"]      = R_n
        out["r_s"]      = r_s
        out["E_g_m2_s"] = E
        out["E_mm_h"]   = E * 3.6    # g m⁻² s⁻¹ → mm h⁻¹
        results[lai] = out
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette for LAI values
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_timeseries(
    results: dict[float, pd.DataFrame],
    out_dir: Path,
    dt_min: float,
) -> None:
    """
    Four-panel time-series figure with one line per LAI value:
      1. Transpiration rate  [mm h⁻¹]
      2. Stomatal resistance [s m⁻¹]
      3. Net crop radiation  [W m⁻²]
      4. VPD & air temperature (shared driving variables)
    """
    lai_list = sorted(results.keys())
    colors   = _COLORS[: len(lai_list)]
    fmt      = mdates.DateFormatter("%d %b\n%H:%M")

    # Use the first result for the shared climate columns
    base = next(iter(results.values()))

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)

    # Panel 1 – Transpiration
    ax = axes[0]
    for lai, col in zip(lai_list, colors):
        df = results[lai]
        ax.plot(df["datetime"], df["E_mm_h"], lw=0.9, color=col,
                label=f"LAI = {lai}")
    ax.set_ylabel("Transpiration [mm h⁻¹]")
    ax.set_title(
        "Stanghellini et al. (2024) Greenhouse Canopy Transpiration\n"
        "r_b = 200 s m⁻¹"
    )
    ax.legend(fontsize=8, ncol=len(lai_list))
    ax.grid(True, alpha=0.3)

    # Panel 2 – Stomatal resistance
    ax = axes[1]
    for lai, col in zip(lai_list, colors):
        df = results[lai]
        ax.plot(df["datetime"], df["r_s"], lw=0.9, color=col,
                label=f"LAI = {lai}")
    ax.set_ylabel("r_s [s m⁻¹]")
    ax.legend(fontsize=8, ncol=len(lai_list))
    ax.grid(True, alpha=0.3)

    # Panel 3 – Net radiation (differs by LAI)
    ax = axes[2]
    for lai, col in zip(lai_list, colors):
        df = results[lai]
        ax.plot(df["datetime"], df["R_n"], lw=0.9, color=col,
                label=f"LAI = {lai}")
    ax.set_ylabel("R_n [W m⁻²]")
    ax.legend(fontsize=8, ncol=len(lai_list))
    ax.grid(True, alpha=0.3)

    # Panel 4 – Climate drivers (same for all LAI)
    ax  = axes[3]
    ax2 = ax.twinx()
    ax.plot(base["datetime"], base["T_air"], color="tomato",
            lw=0.8, label="T_air [°C]")
    ax2.plot(base["datetime"], base["VPD_g_m3"], color="mediumpurple",
             lw=0.7, alpha=0.8, label="VPD [g m⁻³]")
    ax.set_ylabel("Air temperature [°C]")
    ax2.set_ylabel("VPD [g m⁻³]", color="mediumpurple")
    ax2.tick_params(axis="y", colors="mediumpurple")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(fmt)

    fig.tight_layout()
    out_path = out_dir / "transpiration_timeseries.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


def plot_daily_totals(
    results: dict[float, pd.DataFrame],
    out_dir: Path,
    dt_min: float,
) -> None:
    """Grouped bar chart of daily total transpiration per LAI [mm day⁻¹]."""
    lai_list = sorted(results.keys())
    colors   = _COLORS[: len(lai_list)]
    h_step   = dt_min / 60.0   # hours per timestep

    daily: dict[float, pd.Series] = {}
    for lai in lai_list:
        daily[lai] = (
            results[lai]
            .set_index("datetime")["E_mm_h"]
            .resample("D")
            .sum() * h_step
        )

    dates  = daily[lai_list[0]].index
    n_days = len(dates)
    n_lai  = len(lai_list)
    width  = 0.8 / n_lai
    x      = np.arange(n_days)

    fig, ax = plt.subplots(figsize=(max(10, n_days * 0.6), 5))
    for i, (lai, col) in enumerate(zip(lai_list, colors)):
        offset = (i - n_lai / 2.0 + 0.5) * width
        ax.bar(x + offset, daily[lai].values, width=width * 0.95,
               color=col, alpha=0.85, label=f"LAI = {lai}")

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%d %b") for d in dates], rotation=45, ha="right")
    ax.set_ylabel("Transpiration [mm day⁻¹]")
    ax.set_title("Daily Total Transpiration by LAI — Stanghellini et al. (2024)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "daily_transpiration.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


def plot_lai_sensitivity(
    results: dict[float, pd.DataFrame],
    out_dir: Path,
    dt_min: float,
) -> None:
    """
    Summary scatter / line plot: total transpiration and mean daytime
    r_s vs LAI, to visualise the sensitivity of the model to canopy density.
    """
    lai_list   = sorted(results.keys())
    h_step     = dt_min / 60.0
    totals     = []
    mean_rs    = []
    mean_E_day = []

    for lai in lai_list:
        df  = results[lai]
        totals.append((df["E_mm_h"] * h_step).sum())
        mean_rs.append(df["r_s"].mean())
        # daytime mean (radiation > 5 W m⁻²)
        day = df[df["I_global"] > 5.0]
        mean_E_day.append(day["E_mm_h"].mean() if not day.empty else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(lai_list, totals, "o-", color="steelblue", lw=1.5, ms=7)
    ax.set_xlabel("LAI [m² m⁻²]")
    ax.set_ylabel("Total transpiration [mm]")
    ax.set_title("Total Transpiration vs LAI")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(lai_list, mean_E_day, "o-", color="goldenrod", lw=1.5, ms=7)
    ax.set_xlabel("LAI [m² m⁻²]")
    ax.set_ylabel("Mean daytime E [mm h⁻¹]")
    ax.set_title("Daytime Mean Transpiration Rate vs LAI")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(lai_list, mean_rs, "o-", color="darkgreen", lw=1.5, ms=7)
    ax.set_xlabel("LAI [m² m⁻²]")
    ax.set_ylabel("Mean r_s [s m⁻¹]")
    ax.set_title("Mean Stomatal Resistance vs LAI")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Stanghellini et al. (2024) Model — LAI Sensitivity",
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    out_path = out_dir / "lai_sensitivity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    base_dir = Path(__file__).parent
    in_path  = base_dir / "input" / "weather_data_greenhouse.csv"
    out_dir  = base_dir / "output"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading data from: {in_path}")
    df = load_greenhouse_data(in_path)
    print(f"  Loaded {len(df)} records  "
          f"({df['datetime'].min()} → {df['datetime'].max()})")

    # Detect timestep [min]
    dt_min = (
        df["datetime"].diff().dropna().dt.total_seconds().median() / 60
    )
    h_step = dt_min / 60.0

    print(f"Running Stanghellini et al. (2024) model for LAI ∈ {LAI_VALUES} …")
    results = run_model(df, LAI_VALUES)

    # ── Save CSV results (one file per LAI) ───────────────────────────────
    print("\n─── Results by LAI ─────────────────────────────────────────────")
    print(f"  {'LAI':>6}  {'Total E [mm]':>14}  {'Mean E [mm/h]':>15}  "
          f"{'Max E [mm/h]':>13}  {'Mean r_s [s/m]':>14}")
    print("  " + "-" * 70)
    for lai in sorted(results.keys()):
        res  = results[lai]
        tot  = (res["E_mm_h"] * h_step).sum()
        mean = res["E_mm_h"].mean()
        maxv = res["E_mm_h"].max()
        mrs  = res["r_s"].mean()
        print(f"  {lai:>6.1f}  {tot:>14.2f}  {mean:>15.4f}  {maxv:>13.4f}  {mrs:>14.1f}")

        csv_path = out_dir / f"transpiration_LAI{lai:.1f}.csv"
        res.to_csv(csv_path, index=False)
    print("  " + "-" * 70)

    # Also save a combined wide-format CSV (E_mm_h per LAI as columns)
    wide = df[["datetime", "T_air", "RH", "I_global"]].copy()
    for lai in sorted(results.keys()):
        wide[f"E_mm_h_LAI{lai:.1f}"] = results[lai]["E_mm_h"].values
        wide[f"r_s_LAI{lai:.1f}"]    = results[lai]["r_s"].values
    wide.to_csv(out_dir / "transpiration_all_LAI.csv", index=False)
    print(f"\n  Combined CSV saved → {out_dir / 'transpiration_all_LAI.csv'}")

    # ── Figures ───────────────────────────────────────────────────────────
    plot_timeseries(results, out_dir, dt_min)
    plot_daily_totals(results, out_dir, dt_min)
    plot_lai_sensitivity(results, out_dir, dt_min)
    print("Done.")


if __name__ == "__main__":
    main()
