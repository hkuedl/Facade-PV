"""
Redraw Fig. 2 with AOI installability-range results.

Workflow:
1. Place the prepared public CSV files in
   `##Data/Optimization/Fig_input_data`.
2. Run this script from the repository root or from the Figures folder.

Outputs:
- #Fig2a_installability: city-level FPV/RPV carbon mitigation bars.
- #Fig2b_installability: city-level FPV/RPV installable area scatter.
- #Fig2c_installability: city-level FPV/RPV annual generation scatter.

Panel a follows the original Fig_new_2.py bar style but removes the ratio axis
and ratio points. The bars are the main case RPV35_FPV70; the horizontal
intervals are the full installability range.
"""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MAIN_CASE_ID = "RPV35_FPV70"

RATIO_PLOT = 3
FIGWIDTH = 8.5 * RATIO_PLOT
FS = 8 * RATIO_PLOT
LW = 0.4 * RATIO_PLOT
LW2 = 0.75 * RATIO_PLOT
LW_AXIS = 0.8
GRID_ALPHA = 0.5

COLORS = ["skyblue", "mediumseagreen"]  # RPV, FPV
COLOR_RANGE = "#404040"

CITY_RENAME = {
    "Haerbin": "Harbin",
    "Huhehaote": "Hohhot",
    "Wulumuqi": "Urumqi",
    "Xian": "Xi'an",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("FPV_DATA_ROOT", REPOSITORY_ROOT / "##Data")).expanduser()
FIG_INPUT_DIR = Path(
    os.environ.get(
        "FPV_FIG_INPUT_DIR", DATA_ROOT / "Optimization" / "Fig_input_data"
    )
).expanduser()


LEVEL_LABELS = {
    3: "SLC",
    2: "VLC",
    1: "LC-I",
    0: "LC-II",
}


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


def read_installability_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = FIG_INPUT_DIR
    city_path = out_dir / "installability_city_cases.csv"
    range_path = out_dir / "installability_fig2a_ranges.csv"

    if not city_path.exists() or not range_path.exists():
        raise FileNotFoundError(
            "Download or regenerate the installability inputs. Missing "
            f"{city_path.name} or {range_path.name} in {out_dir}."
        )

    return pd.read_csv(city_path), pd.read_csv(range_path)


def read_population(fig_data_dir: Path) -> pd.DataFrame:
    info_path = fig_data_dir / "City_info.xlsx"
    info = pd.read_excel(info_path, usecols=list(range(8)))
    pop_col = info.columns[3]
    out = info[["City", pop_col]].copy()
    out.columns = ["City", "urban_population_10k"]
    out["urban_population_million"] = out["urban_population_10k"] / 100.0
    return out


def prepare_main_case(
    city_cases: pd.DataFrame,
    range_df: pd.DataFrame,
    fig_data_dir: Path,
) -> pd.DataFrame:
    main = city_cases.loc[city_cases["case_id"] == MAIN_CASE_ID].copy()
    if main.empty:
        raise ValueError(f"Main case {MAIN_CASE_ID} was not found.")

    ranges = range_df[
        [
            "City",
            "rpv_carbon_Mt_min",
            "rpv_carbon_Mt_max",
            "fpv_carbon_Mt_min",
            "fpv_carbon_Mt_max",
        ]
    ].copy()

    plot_df = main.merge(ranges, on="City", how="left")
    plot_df = plot_df.merge(read_population(fig_data_dir), on="City", how="left")
    plot_df["City_plot"] = plot_df["City"].replace(CITY_RENAME)

    plot_df.sort_values(
        by=["city_level", "fpv_carbon_Mt"],
        ascending=[False, False],
        inplace=True,
    )
    plot_df.reset_index(drop=True, inplace=True)
    return plot_df


def asymmetric_xerr(values: pd.Series, lows: pd.Series, highs: pd.Series) -> np.ndarray:
    lower = np.maximum(values.to_numpy() - lows.to_numpy(), 0)
    upper = np.maximum(highs.to_numpy() - values.to_numpy(), 0)
    return np.vstack([lower, upper])


def nice_limit(
    values: pd.Series | np.ndarray,
    base: float,
    minimum: float = 0,
    pad: float = 1.08,
) -> float:
    max_value = float(np.nanmax(values)) if len(values) else 0.0
    return max(minimum, float(np.ceil(max_value * pad / base) * base))


def fit_slope_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    denom = np.sum(x * x)
    if denom == 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def slope_p_label(x: np.ndarray, y: np.ndarray) -> str:
    try:
        import statsmodels.api as sm

        sm_x = sm.add_constant(np.asarray(x, dtype=float).reshape(-1, 1))
        model = sm.OLS(np.asarray(y, dtype=float).reshape(-1), sm_x).fit()
        p_value = float(model.pvalues[1])
    except Exception:
        return ""

    if p_value < 0.05:
        return "(p<0.05)"
    return f"(p={p_value:.2g})"


def add_level_labels(ax: plt.Axes, temp_part: pd.DataFrame, x_pos: float) -> None:
    grouped = temp_part.groupby("city_level")["row_idx"]
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


def plot_bar_panel(
    ax: plt.Axes,
    df_part: pd.DataFrame,
    xlim_carbon: float,
    panel_label: str | None = None,
) -> None:
    temp_part = df_part[
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
        color=COLORS[::-1],
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
        ecolor=COLOR_RANGE,
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
        ecolor=COLOR_RANGE,
        elinewidth=LW_AXIS,
        capsize=2.5,
        capthick=LW_AXIS,
        zorder=4,
    )

    ax.invert_yaxis()
    ax.set_xlim(0, xlim_carbon)
    ax.set_xticks(np.arange(0, xlim_carbon + 0.1, 15))
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Carbon mitigation potential (Million ton)", fontsize=FS - 9)
    ax.set_ylabel(None)

    add_level_labels(ax, temp_part, ax.get_xlim()[-1] * 0.83)

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


def draw_fig2a(plot_df: pd.DataFrame, output_dir: Path) -> None:
    plt.rc("font", size=FS - 12)

    n = len(plot_df)
    mid = n // 2
    parts = [plot_df.iloc[:mid].copy(), plot_df.iloc[mid:].copy()]

    xlim_carbon = nice_limit(
        plot_df[["rpv_carbon_Mt_max", "fpv_carbon_Mt_max"]].to_numpy().reshape(-1),
        base=15,
        minimum=90,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=np.array([FIGWIDTH, FIGWIDTH * 1.2]) / 2.54,
    )

    plot_bar_panel(axes[0], parts[0], xlim_carbon, panel_label="a")
    plot_bar_panel(axes[1], parts[1], xlim_carbon)

    legend_handles = [
        mpatches.Patch(fc=COLORS[1], alpha=0.8, label="FPV carbon mitigation potential"),
        mpatches.Patch(fc=COLORS[0], alpha=0.8, label="RPV carbon mitigation potential"),
        Line2D([0], [0], color=COLOR_RANGE, lw=LW_AXIS, label="Installability range"),
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

    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.1, top=0.95, wspace=0.4, hspace=0)
    save_figure(fig, output_dir / "Fig2a_installability")


def scatter_with_fit(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    panel_label: str,
    output_base: Path,
    fit_color: str,
    cmap: str,
    min_xlim: float,
    min_ylim: float,
    limit_base: float,
    label_xy: tuple[float, float],
    p_xy: tuple[float, float],
    y_equal_xy: tuple[float, float],
    xtick_step: float | None = None,
    ytick_step: float | None = None,
) -> None:
    plt.rc("font", size=FS - 3)

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    pop = data["urban_population_million"]
    norm = plt.Normalize(vmin=1, vmax=10)

    xlim = nice_limit(x, base=limit_base, minimum=min_xlim)
    ylim = nice_limit(y, base=limit_base, minimum=min_ylim)

    fig, ax = plt.subplots(
        figsize=np.array([FIGWIDTH * 2 / 3, FIGWIDTH * 2 / 3 * 0.7]) / 2.54
    )

    sc = ax.scatter(
        x,
        y,
        c=pop,
        cmap=cmap,
        norm=norm,
        alpha=0.8,
        s=70,
        edgecolors="black",
        linewidths=LW,
    )

    k = fit_slope_no_intercept(x, y)
    x_fit = np.linspace(0, xlim, 100)
    y_fit = k * x_fit
    ax.plot(x_fit, y_fit, "-", color=fit_color, linewidth=LW2)

    ax.text(
        label_xy[0],
        label_xy[1],
        f"$y={k:.2f}x$",
        fontsize=FS - 6,
        color="red",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    p_text = slope_p_label(x, y)
    if p_text:
        ax.text(
            p_xy[0],
            p_xy[1],
            p_text,
            fontsize=FS - 9,
            color="red",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("RPV", fontsize=FS)
    ax.set_ylabel("FPV", fontsize=FS)
    ax.set_title(title, fontsize=FS - 5, y=1, fontweight="bold")

    x_equal = np.linspace(0, xlim, 100)
    ax.plot(x_equal, x_equal, linestyle="-", color="gray", linewidth=LW)
    ax.text(
        y_equal_xy[0],
        y_equal_xy[1],
        "$y=x$",
        fontsize=FS - 6,
        color="gray",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    if xtick_step is not None:
        ax.set_xticks(np.arange(0, xlim + 0.1, xtick_step))
    if ytick_step is not None:
        ax.set_yticks(np.arange(0, ylim + 0.1, ytick_step))

    ax.grid(alpha=GRID_ALPHA)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Urban population (Million)", fontsize=FS - 5)
    cbar.set_ticks([1, 3, 5, 10])
    cbar.ax.tick_params(labelsize=FS - 5)

    ax.text(
        -0.25 * xlim,
        1.08 * ylim,
        panel_label,
        fontsize=FS + 15,
        fontweight="bold",
    )

    plt.subplots_adjust(left=0.19, right=0.96, bottom=0.18, top=0.96)
    save_figure(fig, output_base)


def draw_fig2b(plot_df: pd.DataFrame, output_dir: Path) -> None:
    scatter_with_fit(
        data=plot_df,
        x_col="rpv_installable_area_km2",
        y_col="fpv_installable_area_km2",
        title="Installable area (km$^2$)",
        panel_label="b",
        output_base=output_dir / "Fig2b_installability",
        fit_color="#831717",
        cmap="RdPu",
        min_xlim=0,
        min_ylim=0,
        limit_base=50,
        label_xy=(0.30, 0.64),
        p_xy=(0.27, 0.58),
        y_equal_xy=(0.94, 0.54),
        xtick_step=100,
        ytick_step=200,
    )


def draw_fig2c(plot_df: pd.DataFrame, output_dir: Path) -> None:
    scatter_with_fit(
        data=plot_df,
        x_col="rpv_generation_TWh",
        y_col="fpv_generation_TWh",
        title="Annual generation (TWh)",
        panel_label="c",
        output_base=output_dir / "Fig2c_installability",
        fit_color="#521783",
        cmap="OrRd",
        min_xlim=0,
        min_ylim=0,
        limit_base=5,
        label_xy=(0.92, 0.50),
        p_xy=(0.89, 0.45),
        y_equal_xy=(0.76, 0.90),
        xtick_step=10,
        ytick_step=10,
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = repo_root()
    figures_dir = root / "Figures"
    fig_data_dir = FIG_INPUT_DIR
    output_dir = Path(os.environ.get("FPV_FIG_OUTPUT_DIR", figures_dir / "Figs_new")).expanduser()

    setup_style(figures_dir)
    city_cases, range_df = read_installability_tables(root)
    plot_df = prepare_main_case(city_cases, range_df, fig_data_dir)

    draw_fig2a(plot_df, output_dir)
    draw_fig2b(plot_df, output_dir)
    draw_fig2c(plot_df, output_dir)

    print(f"Saved Fig. 2 panels to: {output_dir}")


if __name__ == "__main__":
    main()
