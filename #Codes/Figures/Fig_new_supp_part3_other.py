"""
Draw supplementary Fig. 2a-style zone comparisons for HnD, MnD, HnS, and MnS.

Inputs are read from `##Data/Optimization/Fig_input_data`, together with the
electricity-price workbook in `##Data/Optimization/Electricity_price`.

The bars show the main case RPV35_FPV70. Horizontal intervals on bars show the
installability range generated from RPV25-45% and FPV55-75%. Ratio axes and ratio
points are intentionally omitted to match the Fig. 2a style.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MAIN_RPV = 0.35
MAIN_FPV = 0.70
MAIN_CASE_ID = "RPV35_FPV70"

BASELINE_RPV = 0.35
BASELINE_FPV = 0.90
RPV_RATIOS = (0.25, 0.30, 0.35, 0.40, 0.45)
FPV_RATIOS = (0.55, 0.60, 0.65, 0.70, 0.75)

ZONES = ("HnD", "MnD", "HnS", "MnS")

RATIO_PLOT = 3
FIGWIDTH = 8.5 * RATIO_PLOT
FS = 8 * RATIO_PLOT
LW_AXIS = 0.8

CITY_RENAME = {
    "Haerbin": "Harbin",
    "Huhehaote": "Hohhot",
    "Wulumuqi": "Urumqi",
    "Xian": "Xi'an",
}

LEVEL_LABELS = {
    3: "SLC",
    2: "VLC",
    1: "LC-I",
    0: "LC-II",
}

COLORS = {
    "rpv": "skyblue",
    "fpv": "mediumseagreen",
    "range": "#404040",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("FPV_DATA_ROOT", REPOSITORY_ROOT / "##Data")).expanduser()
FIG_INPUT_DIR = Path(
    os.environ.get(
        "FPV_FIG_INPUT_DIR", DATA_ROOT / "Optimization" / "Fig_input_data"
    )
).expanduser()
PRICE_FILE = Path(
    os.environ.get(
        "FPV_PRICE_FILE",
        DATA_ROOT / "Optimization" / "Electricity_price" / "Realtime Price.xlsx",
    )
).expanduser()


def repo_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "figures":
        return here.parents[1]
    return Path.cwd()


def setup_style(figures_dir: Path) -> None:
    font_path = figures_dir / "arial.ttf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        custom_font = fm.FontProperties(fname=str(font_path))
        family = custom_font.get_name()
    else:
        family = "Arial"

    rcParams.update(
        {
            "font.family": family,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def load_functions_module(figures_dir: Path):
    functions_path = SCRIPT_DIR.parent / "Optimization" / "Functions.py"
    spec = importlib.util.spec_from_file_location("figures_functions", functions_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {functions_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def carbon_factor_by_city(cities: list[str], figures_dir: Path) -> pd.DataFrame:
    funcs = load_functions_module(figures_dir)
    old_cwd = Path.cwd()
    rows = []
    try:
        os.chdir(figures_dir)
        for city in cities:
            rows.append({"city": city, "carbon_factor_Mt_per_TWh": float(funcs.TOU_period(city, PRICE_FILE)[-1])})
    finally:
        os.chdir(old_cwd)
    return pd.DataFrame(rows)


def city_level_table(fig_data_dir: Path) -> pd.DataFrame:
    city_info = pd.read_excel(fig_data_dir / "City_info.xlsx", usecols=list(range(8)))
    scale_col = city_info.columns[5]
    mapping = {
        "II\u578b\u5927\u57ce\u5e02": 0,
        "I\u578b\u5927\u57ce\u5e02": 1,
        "\u7279\u5927\u57ce\u5e02": 2,
        "\u8d85\u5927\u57ce\u5e02": 3,
    }
    out = city_info[["City", scale_col]].rename(
        columns={"City": "city", scale_col: "city_scale_label"}
    )
    out["city_level"] = out["city_scale_label"].map(mapping)
    return out


def read_hdf_metric(fig_data_dir: Path, filename: str, value_name: str) -> pd.DataFrame:
    path = fig_data_dir / filename
    try:
        raw = pd.read_hdf(path, key="df")
    except ImportError as exc:
        raise ImportError(
            "This script needs pandas HDF support (PyTables/tables) to read "
            f"{path}. Install it in the Python environment, for example: pip install tables"
        ) from exc

    df = raw.reset_index()
    if value_name in df.columns:
        value_col = value_name
    elif len(raw.columns) == 1:
        value_col = raw.columns[0]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        excluded = {
            "city_index",
            "grid_index",
            "grid_number",
            "grid_num",
            "index",
            "level_0",
            "level_1",
        }
        candidates = [c for c in numeric_cols if c not in excluded]
        if not candidates:
            raise ValueError(f"Cannot infer value column in {filename}.")
        value_col = candidates[-1]

    city_col = infer_column(df, ["city", "City", "level_0"])
    grid_col = infer_column(
        df,
        [
            "grid_index",
            "grid_number",
            "grid_num",
            "Grid_number",
            "Grid_Number",
            "grid",
            "Grid",
            "level_1",
            "index",
        ],
    )

    out = df[[city_col, grid_col, value_col]].copy()
    out.columns = ["city", "grid_index", value_name]
    out["grid_index"] = out["grid_index"].astype(int)
    return out


def infer_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Cannot infer column from candidates {candidates}; columns={list(df.columns)}")


def read_grid_labels(root: Path) -> pd.DataFrame:
    labels_path = FIG_INPUT_DIR / "nationwide_grid_zone_labels.csv"
    labels = pd.read_csv(labels_path)
    required = {"city", "grid_index", "zone"}
    missing = required.difference(labels.columns)
    if missing:
        raise KeyError(f"{labels_path} missing columns: {sorted(missing)}")
    labels = labels[["city", "grid_index", "zone", "eligible_for_facade_model"]].copy()
    labels["grid_index"] = labels["grid_index"].astype(int)
    return labels


def build_zone_baseline(root: Path) -> pd.DataFrame:
    fig_data_dir = FIG_INPUT_DIR
    figures_dir = SCRIPT_DIR

    labels = read_grid_labels(root)
    cap_roof = read_hdf_metric(fig_data_dir, "Cap_roof_ideal_all_df.h5", "rpv_capacity_baseline_GW")
    cap_fpv = read_hdf_metric(fig_data_dir, "Cap_facade_ideal_all_df.h5", "fpv_capacity_baseline_GW")
    pow_roof = read_hdf_metric(fig_data_dir, "Power_roof_ideal_1_sum_all_df.h5", "rpv_generation_baseline_MWh")
    pow_fpv = read_hdf_metric(fig_data_dir, "Power_facade_ideal_1_sum_all_df.h5", "fpv_generation_baseline_MWh")
    facade_area = read_hdf_metric(fig_data_dir, "Grid_type_all_df.h5", "facade_area")

    grid = labels.merge(cap_roof, on=["city", "grid_index"], how="left")
    grid = grid.merge(cap_fpv, on=["city", "grid_index"], how="left")
    grid = grid.merge(pow_roof, on=["city", "grid_index"], how="left")
    grid = grid.merge(pow_fpv, on=["city", "grid_index"], how="left")
    grid = grid.merge(facade_area, on=["city", "grid_index"], how="left")

    matched_counts = {
        "Cap_roof_ideal_all_df.h5": grid["rpv_capacity_baseline_GW"].notna().sum(),
        "Cap_facade_ideal_all_df.h5": grid["fpv_capacity_baseline_GW"].notna().sum(),
        "Power_roof_ideal_1_sum_all_df.h5": grid["rpv_generation_baseline_MWh"].notna().sum(),
        "Power_facade_ideal_1_sum_all_df.h5": grid["fpv_generation_baseline_MWh"].notna().sum(),
        "Grid_type_all_df.h5": grid["facade_area"].notna().sum(),
    }
    failed = [name for name, count in matched_counts.items() if count == 0]
    if failed:
        raise ValueError(
            "No rows matched between zone labels and these HDF inputs: "
            f"{failed}. Check whether HDF grid_number is zero-based and uses "
            "the same city/grid indexing as nationwide_grid_zone_labels.csv."
        )

    metric_cols = [
        "rpv_capacity_baseline_GW",
        "fpv_capacity_baseline_GW",
        "rpv_generation_baseline_MWh",
        "fpv_generation_baseline_MWh",
        "facade_area",
    ]
    grid[metric_cols] = grid[metric_cols].fillna(0.0)

    grouped = (
        grid.groupby(["city", "zone"], as_index=False)[metric_cols]
        .sum()
        .rename(
            columns={
                "rpv_generation_baseline_MWh": "rpv_generation_baseline_TWh",
                "fpv_generation_baseline_MWh": "fpv_generation_baseline_TWh",
                "facade_area": "fpv_aoi_area_m2",
            }
        )
    )
    grouped["rpv_generation_baseline_TWh"] = grouped["rpv_generation_baseline_TWh"] / 1e6
    grouped["fpv_generation_baseline_TWh"] = grouped["fpv_generation_baseline_TWh"] / 1e6
    grouped["fpv_aoi_area_km2"] = grouped["fpv_aoi_area_m2"] / 1e6

    cities = sorted(labels["city"].dropna().unique().tolist())
    grouped = grouped.merge(carbon_factor_by_city(cities, figures_dir), on="city", how="left")
    grouped = grouped.merge(city_level_table(fig_data_dir), on="city", how="left")
    return grouped


def build_zone_cases(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for rpv in RPV_RATIOS:
        for fpv in FPV_RATIOS:
            case = baseline.copy()
            r_scale = rpv / BASELINE_RPV
            f_scale = fpv / BASELINE_FPV
            case["case_id"] = f"RPV{int(round(rpv * 100)):02d}_FPV{int(round(fpv * 100)):02d}"
            case["rpv_installability"] = rpv
            case["fpv_installability_of_filtered_AOI"] = fpv
            case["fpv_installed_area_km2"] = case["fpv_aoi_area_km2"] * fpv
            case["rpv_capacity_GW"] = case["rpv_capacity_baseline_GW"] * r_scale
            case["fpv_capacity_GW"] = case["fpv_capacity_baseline_GW"] * f_scale
            case["rpv_generation_TWh"] = case["rpv_generation_baseline_TWh"] * r_scale
            case["fpv_generation_TWh"] = case["fpv_generation_baseline_TWh"] * f_scale
            case["rpv_carbon_Mt"] = case["rpv_generation_TWh"] * case["carbon_factor_Mt_per_TWh"]
            case["fpv_carbon_Mt"] = case["fpv_generation_TWh"] * case["carbon_factor_Mt_per_TWh"]
            case["fpv_to_rpv_carbon_ratio"] = safe_ratio(case["fpv_carbon_Mt"], case["rpv_carbon_Mt"])
            rows.append(case)

    cases = pd.concat(rows, ignore_index=True)
    main = cases.loc[cases["case_id"] == MAIN_CASE_ID].copy()
    ranges = (
        cases.groupby(["city", "zone"], as_index=False)
        .agg(
            rpv_carbon_Mt_min=("rpv_carbon_Mt", "min"),
            rpv_carbon_Mt_max=("rpv_carbon_Mt", "max"),
            fpv_carbon_Mt_min=("fpv_carbon_Mt", "min"),
            fpv_carbon_Mt_max=("fpv_carbon_Mt", "max"),
            rpv_capacity_GW_min=("rpv_capacity_GW", "min"),
            rpv_capacity_GW_max=("rpv_capacity_GW", "max"),
            fpv_capacity_GW_min=("fpv_capacity_GW", "min"),
            fpv_capacity_GW_max=("fpv_capacity_GW", "max"),
            rpv_generation_TWh_min=("rpv_generation_TWh", "min"),
            rpv_generation_TWh_max=("rpv_generation_TWh", "max"),
            fpv_generation_TWh_min=("fpv_generation_TWh", "min"),
            fpv_generation_TWh_max=("fpv_generation_TWh", "max"),
            fpv_installed_area_km2_min=("fpv_installed_area_km2", "min"),
            fpv_installed_area_km2_max=("fpv_installed_area_km2", "max"),
            fpv_to_rpv_carbon_ratio_min=("fpv_to_rpv_carbon_ratio", "min"),
            fpv_to_rpv_carbon_ratio_max=("fpv_to_rpv_carbon_ratio", "max"),
        )
    )
    return cases, main, ranges


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def asymmetric_xerr(values: pd.Series, lows: pd.Series, highs: pd.Series) -> np.ndarray:
    lower = np.maximum(values.to_numpy() - lows.to_numpy(), 0)
    upper = np.maximum(highs.to_numpy() - values.to_numpy(), 0)
    return np.vstack([lower, upper])


def prepare_zone_plot_data(main: pd.DataFrame, ranges: pd.DataFrame, zone: str) -> pd.DataFrame:
    df = main.loc[main["zone"] == zone].merge(ranges, on=["city", "zone"], how="left")
    df = df.loc[(df["rpv_carbon_Mt_max"] > 0) | (df["fpv_carbon_Mt_max"] > 0)].copy()
    df["City_plot"] = df["city"].replace(CITY_RENAME)
    df.sort_values(
        by=["city_level", "fpv_carbon_Mt"],
        ascending=[False, False],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def add_level_labels(ax: plt.Axes, panel_df: pd.DataFrame, x_pos: float) -> None:
    grouped = panel_df.groupby("city_level")["row_idx"]
    for lvl, rows in grouped:
        if pd.isna(lvl):
            continue
        min_idx = rows.min()
        max_idx = rows.max()
        mid_idx = 0.5 * (min_idx + max_idx)
        arrow_len = 0.5
        h_len = ax.get_xlim()[-1] * 0.04

        ax.annotate(
            "",
            xy=(x_pos, min_idx),
            xytext=(x_pos, min_idx - arrow_len),
            arrowprops=dict(
                arrowstyle="<-",
                lw=LW_AXIS,
                color="black",
                mutation_scale=20,
            ),
            annotation_clip=False,
            transform=ax.transData,
        )
        ax.plot(
            [x_pos - h_len, x_pos + h_len],
            [min_idx - 0.3, min_idx - 0.3],
            color="black",
            lw=LW_AXIS,
            transform=ax.transData,
            clip_on=False,
        )

        ax.plot(
            [x_pos, x_pos],
            [min_idx, max_idx],
            color="black",
            lw=LW_AXIS,
            transform=ax.transData,
            clip_on=False,
        )

        ax.annotate(
            "",
            xy=(x_pos, max_idx),
            xytext=(x_pos, max_idx + arrow_len),
            arrowprops=dict(
                arrowstyle="<-",
                lw=LW_AXIS,
                color="black",
                mutation_scale=20,
            ),
            annotation_clip=False,
            transform=ax.transData,
        )
        ax.plot(
            [x_pos - h_len, x_pos + h_len],
            [max_idx + 0.3, max_idx + 0.3],
            color="black",
            lw=LW_AXIS,
            transform=ax.transData,
            clip_on=False,
        )

        ax.text(
            x_pos,
            mid_idx,
            LEVEL_LABELS.get(int(lvl), f"Level {int(lvl)}"),
            ha="center",
            va="center",
            fontsize=FS - 9,
            transform=ax.transData,
            bbox=dict(facecolor="white", edgecolor="none"),
        )


def plot_panel(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    xlim_carbon: float,
    panel_label: str | None = None,
) -> None:
    temp_part = panel_df[
        [
            "City_plot",
            "city_level",
            "fpv_carbon_Mt",
            "rpv_carbon_Mt",
            "fpv_carbon_Mt_min",
            "fpv_carbon_Mt_max",
            "rpv_carbon_Mt_min",
            "rpv_carbon_Mt_max",
        ]
    ].copy()
    temp_part.set_index("City_plot", inplace=True)
    df_plot_part = temp_part[["fpv_carbon_Mt", "rpv_carbon_Mt"]]

    df_plot_part.plot(
        kind="barh",
        ax=ax,
        stacked=False,
        width=0.8,
        edgecolor="none",
        linewidth=LW_AXIS,
        alpha=0.8,
        color=[COLORS["fpv"], COLORS["rpv"]],
        legend=False,
    )

    temp_part.reset_index(inplace=True)
    temp_part["row_idx"] = temp_part.index

    y = temp_part["row_idx"].to_numpy(dtype=float)
    y_fpv = y - 0.2
    y_rpv = y + 0.2

    ax.errorbar(
        temp_part["fpv_carbon_Mt"],
        y_fpv,
        xerr=asymmetric_xerr(
            temp_part["fpv_carbon_Mt"],
            temp_part["fpv_carbon_Mt_min"],
            temp_part["fpv_carbon_Mt_max"],
        ),
        fmt="none",
        ecolor=COLORS["range"],
        elinewidth=LW_AXIS,
        capsize=2.5,
        capthick=LW_AXIS,
        zorder=4,
    )
    ax.errorbar(
        temp_part["rpv_carbon_Mt"],
        y_rpv,
        xerr=asymmetric_xerr(
            temp_part["rpv_carbon_Mt"],
            temp_part["rpv_carbon_Mt_min"],
            temp_part["rpv_carbon_Mt_max"],
        ),
        fmt="none",
        ecolor=COLORS["range"],
        elinewidth=LW_AXIS,
        capsize=2.5,
        capthick=LW_AXIS,
        zorder=4,
    )

    ax.invert_yaxis()
    ax.set_xlim(0, xlim_carbon)
    tick_step = 15 if xlim_carbon >= 30 else 5
    ax.set_xticks(np.arange(0, xlim_carbon + 0.1, tick_step))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Carbon mitigation potential (Million ton)", fontsize=FS - 9)
    ax.set_ylabel(None)

    add_level_labels(ax, temp_part, xlim_carbon * 0.83)

    if panel_label:
        ax.text(
            -0.17 * xlim_carbon,
            -3.0,
            panel_label,
            ha="center",
            va="center",
            fontsize=FS + 9,
            fontweight="bold",
        )


def draw_zone_figure(zone_df: pd.DataFrame, zone: str, output_dir: Path) -> None:
    if zone_df.empty:
        print(f"Skip {zone}: no non-zero RPV/FPV potential.")
        return

    n = len(zone_df)
    mid = n // 2
    parts = [zone_df.iloc[:mid].copy(), zone_df.iloc[mid:].copy()]

    xlim_carbon = max(
        5,
        np.ceil(zone_df[["rpv_carbon_Mt_max", "fpv_carbon_Mt_max"]].max().max() / 5) * 5 * 1.08,
    )

    plt.rc("font", size=FS - 12)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=np.array([FIGWIDTH, FIGWIDTH * 1.2]) / 2.54,
    )

    plot_panel(axes[0], parts[0], xlim_carbon, panel_label="a")
    plot_panel(axes[1], parts[1], xlim_carbon)

    legend_handles = [
        mpatches.Patch(fc=COLORS["fpv"], alpha=0.8, label="FPV carbon mitigation potential"),
        mpatches.Patch(fc=COLORS["rpv"], alpha=0.8, label="RPV carbon mitigation potential"),
        Line2D([0], [0], color=COLORS["range"], lw=LW_AXIS, label="Installability range"),
    ]
    lg = fig.legend(
        handles=legend_handles,
        handleheight=0.7,
        handlelength=1.0,
        loc="lower center",
        ncol=3,
        fontsize=FS - 9,
        bbox_to_anchor=(0.5, 0.00),
    )
    frame = lg.get_frame()
    frame.set_linewidth(0.6)
    frame.set_edgecolor("black")
    frame.set_facecolor("none")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.95, wspace=0.4, hspace=0)

    base = output_dir / f"Fig_supp_{zone}_RPV_FPV_potential"
    fig.savefig(base.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base.with_suffix('.pdf')}")


def main() -> None:
    root = repo_root()
    figures_dir = SCRIPT_DIR
    output_dir = figures_dir / "Figs_new_supp"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style(figures_dir)

    baseline = build_zone_baseline(root)
    cases, main_case, ranges = build_zone_cases(baseline)

    cases.to_csv(output_dir / "zone_installability_city_cases.csv", index=False)
    main_case.to_csv(output_dir / "zone_installability_main_case.csv", index=False)
    ranges.to_csv(output_dir / "zone_installability_ranges.csv", index=False)

    for zone in ZONES:
        zone_df = prepare_zone_plot_data(main_case, ranges, zone)
        draw_zone_figure(zone_df, zone, output_dir)


if __name__ == "__main__":
    main()
