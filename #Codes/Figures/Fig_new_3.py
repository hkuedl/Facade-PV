"""Generate the Fig. 3 from compact public inputs or raw results."""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.io import loadmat
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("FPV_DATA_ROOT", REPOSITORY_ROOT / "##Data")).expanduser()
OPTIMIZATION_DATA_DIR = DATA_ROOT / "Optimization"
FIG_INPUT_DIR = Path(
    os.environ.get("FPV_FIG_INPUT_DIR", OPTIMIZATION_DATA_DIR / "Fig_input_data")
).expanduser()
ML_RESULTS_DIR = Path(
    os.environ.get("FPV_ML_RESULTS_DIR", DATA_ROOT / "Evaluation" / "All_output")
).expanduser()
CITY_FEATURE_DIR = Path(
    os.environ.get("FPV_CITY_FEATURE_DIR", DATA_ROOT / "Evaluation" / "All_input")
).expanduser()
PLANNING_RESULTS_DIR = Path(
    os.environ.get(
        "FPV_OPT_RESULTS_DIR", OPTIMIZATION_DATA_DIR / "Planning_results"
    )
).expanduser()
CITY_STATISTICS_FILE = Path(
    os.environ.get(
        "FPV_CITY_STATISTICS_FILE",
        OPTIMIZATION_DATA_DIR / "Other_parameters" / "City_statistic.xlsx",
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get("FPV_FIG_OUTPUT_DIR", SCRIPT_DIR / "Figs_new")
).expanduser()

font_path = SCRIPT_DIR / "arial.ttf"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
else:
    warnings.warn(f"Arial font file not found: {font_path}; using Matplotlib default font.")

ABNORMAL_VOLUME_CITY_INDICES = {2, 6, 8, 69, 92, 97}
YEARS = [2030, 2035, 2040, 2045, 2050]
Y, D = len(YEARS), 12
FPV_MAINTENANCE_RATE = 0.04
RPV_MAINTENANCE_RATE = 0.02
PV_LIFETIME = 25
DISCOUNT_RATE = 0.08


def read_city_scale_order(city_names):
    """Return city groups ordered as SLC, VLC, LC-I, and LC-II."""
    city_info = pd.read_excel(FIG_INPUT_DIR / "City_info.xlsx").set_index("City")
    labels = ["\u8d85\u5927\u57ce\u5e02", "\u7279\u5927\u57ce\u5e02", "I\u578b\u5927\u57ce\u5e02", "II\u578b\u5927\u57ce\u5e02"]
    groups = []
    for label in labels:
        names = city_info.index[city_info["\u89c4\u6a21\u7b49\u7ea7"] == label].tolist()
        groups.append([city_names.index(city) for city in names if city in city_names])
    grouped_names = {city_names[i] for group in groups for i in group}
    missing = set(city_names) - grouped_names
    if missing:
        raise ValueError(f"City scale grouping missed cities: {sorted(missing)}")
    return groups


def opt_result_path(city_index, city_name):
    suffix = "_hybrid_n.mat" if city_index in ABNORMAL_VOLUME_CITY_INDICES else "_hybrid.mat"
    return str(PLANNING_RESULTS_DIR / (city_name + suffix))


def pretty_city_name(city):
    replacements = {"Haerbin": "Harbin", "Huhehaote": "Hohhot", "Wulumuqi": "Urumqi", "Xian": "Xi'an"}
    return replacements.get(city, city)


def annual_generation_from_typical_days(power, representative_days):
    """Aggregate grid × year × month × hour power into annual generation for each year."""
    weights = np.asarray(representative_days, dtype=float)[None, None, :, None]
    return np.sum(power * weights, axis=(0, 2, 3))


def calculate_lcoe(capex, annual_generation, maintenance_rate):
    """LCOE using upfront CAPEX plus fixed annual O&M over a 25-year lifetime."""
    discount_factors = (1.0 + DISCOUNT_RATE) ** (-np.arange(1, PV_LIFETIME + 1))
    discounted_generation = annual_generation * np.sum(discount_factors)
    discounted_cost = capex + maintenance_rate * capex * np.sum(discount_factors)
    return np.divide(
        discounted_cost,
        discounted_generation,
        out=np.full_like(discounted_cost, np.nan, dtype=float),
        where=discounted_generation > 0,
    )


def compute_fig3_inputs():
    city_statistic = pd.read_excel(
        CITY_STATISTICS_FILE,
        sheet_name="Class_Volume",
        index_col=0,
    )
    city_names = city_statistic.index.tolist()
    city_groups = read_city_scale_order(city_names)
    clu_days = np.load(FIG_INPUT_DIR / "Clu_days.npy")

    n_city = len(city_names)
    fpv_capacity = np.zeros((n_city, Y))
    rpv_capacity = np.zeros((n_city, Y))
    fpv_lcoe = np.full((n_city, Y), np.nan)
    rpv_lcoe = np.full((n_city, Y), np.nan)

    wall_0 = [3.18, 2.98, 2.75, 2.70, 2.65, 2.60]
    win_0 = [4.80, 4.50, 4.15, 4.08, 4.00, 3.92]
    roof_price = np.asarray(wall_0) * 1e3
    window_price = np.asarray(win_0) * 1e3

    for cc, city_name in enumerate(city_names):
        print(city_name)
        data_path = opt_result_path(cc, city_name)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing optimization result: {data_path}")

        feature_path = str(CITY_FEATURE_DIR / f"{city_name}_ALL_Featuers.npy")
        features = np.load(feature_path)
        indices_non_zero = np.where(features[:, 11] != 0)[0]
        wwr = features[indices_non_zero, 13]

        data = loadmat(data_path)
        r_cap_f = np.asarray(data["R_Cap_f"], dtype=float)       # grid × 3 × year
        r_cap_r = np.asarray(data["R_Cap_r"], dtype=float)       # grid × 2 × year
        r_pow_f = np.asarray(data["R_Pow_f"], dtype=float)       # grid × year × month × hour
        r_pow_r = np.asarray(data["R_Pow_r"], dtype=float)       # grid × 2 × year × month × hour

        if r_cap_f.shape[0] != len(wwr) or r_cap_r.shape[0] != len(wwr):
            raise ValueError(
                f"{city_name}: inconsistent grid counts: WWR={len(wwr)}, "
                f"R_Cap_f={r_cap_f.shape[0]}, R_Cap_r={r_cap_r.shape[0]}"
            )

        # Panel a: optimized FPV capacity and generation in the RPV+FPV+storage case.
        fpv_capacity[cc, :] = np.sum(r_cap_f[:, 0, :], axis=0) / 1e6
        fpv_generation = annual_generation_from_typical_days(r_pow_f, clu_days[cc, :])

        # Panel b: RPV from type=1, i.e. the same RPV+FPV+storage optimization case as panel a.
        rpv_capacity[cc, :] = np.sum(r_cap_r[:, 1, :], axis=0) / 1e6
        rpv_generation = annual_generation_from_typical_days(r_pow_r[:, 1, :, :, :], clu_days[cc, :])

        fpv_capex = np.zeros(Y)
        rpv_capex = np.zeros(Y)
        for yy in range(Y):
            fpv_unit_price = roof_price[yy + 1] * (1.0 - wwr) + window_price[yy + 1] * wwr
            fpv_capex[yy] = np.sum(fpv_unit_price * r_cap_f[:, 0, yy])
            rpv_capex[yy] = roof_price[yy + 1] * np.sum(r_cap_r[:, 1, yy])

        fpv_lcoe[cc, :] = calculate_lcoe(fpv_capex, fpv_generation, FPV_MAINTENANCE_RATE)
        rpv_lcoe[cc, :] = calculate_lcoe(rpv_capex, rpv_generation, RPV_MAINTENANCE_RATE)

    # Preserve the original panel-a order: city scale first, then descending 2050 optimized FPV capacity.
    city_order = np.zeros((n_city, 3))
    city_order[:, 0] = np.arange(n_city)
    for group_rank, group in enumerate(city_groups):
        for city_index in group:
            city_order[city_index, 2] = 4 - group_rank
            city_order[city_index, 1] = fpv_capacity[city_index, -1]
    ordered_city_indices = city_order[np.lexsort((-city_order[:, 1], -city_order[:, 2])), 0].astype(int)

    fpv_capacity = fpv_capacity[ordered_city_indices, :]
    fpv_lcoe = fpv_lcoe[ordered_city_indices, :]
    rpv_capacity = rpv_capacity[ordered_city_indices, :]
    rpv_lcoe = rpv_lcoe[ordered_city_indices, :]

    input_dir = FIG_INPUT_DIR
    input_dir.mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(input_dir, "Fig3_list_city.npy"), ordered_city_indices)
    np.save(os.path.join(input_dir, "Fig3_FPV_capacity.npy"), fpv_capacity)
    np.save(os.path.join(input_dir, "Fig3_FPV_LCOE.npy"), fpv_lcoe)
    np.save(os.path.join(input_dir, "Fig3_RPV_capacity.npy"), rpv_capacity)
    np.save(os.path.join(input_dir, "Fig3_RPV_LCOE.npy"), rpv_lcoe)
    return city_names, ordered_city_indices, fpv_capacity, fpv_lcoe, rpv_capacity, rpv_lcoe


def rounded_upper(values, minimum, step):
    finite = np.asarray(values)[np.isfinite(values)]
    maximum = float(np.max(finite)) if finite.size else minimum
    return max(minimum, np.ceil(maximum / step) * step)


def add_city_group_annotation(ax, column_index, fontsize, line_width):
    if column_index == 0:
        rows = [(88, 100, "SLC"), (58, 86, "VLC"), (32, 56, "LC-I"), (0, 30, "LC-II")]
    else:
        rows = [(0, 100, "LC-II")]
    x_pos = ax.get_xlim()[1] * 0.83
    arrow_len = 0.5
    h_len = ax.get_xlim()[1] * 0.04
    for min_idx, max_idx, label in rows:
        mid_idx = 0.5 * (min_idx + max_idx)
        ax.annotate("", xy=(x_pos, min_idx), xytext=(x_pos, min_idx - arrow_len),
                    arrowprops=dict(arrowstyle="<-", lw=line_width, color="black", mutation_scale=20),
                    annotation_clip=False)
        ax.plot([x_pos - h_len, x_pos + h_len], [min_idx - 0.3, min_idx - 0.3],
                color="black", lw=line_width, clip_on=False)
        ax.plot([x_pos, x_pos], [min_idx, max_idx], color="black", lw=line_width, clip_on=False)
        ax.annotate("", xy=(x_pos, max_idx), xytext=(x_pos, max_idx + arrow_len),
                    arrowprops=dict(arrowstyle="<-", lw=line_width, color="black", mutation_scale=20),
                    annotation_clip=False)
        ax.plot([x_pos - h_len, x_pos + h_len], [max_idx + 0.3, max_idx + 0.3],
                color="black", lw=line_width, clip_on=False)
        ax.text(x_pos, mid_idx, label, ha="center", va="center", fontsize=fontsize,
                bbox=dict(facecolor="white", edgecolor="none"))


def draw_half_panel(ax, city_labels, capacity, lcoe, capacity_colors, lcoe_colors,
                    capacity_xlim, lcoe_xlim, column_index, fontsize, line_width):
    n = len(city_labels)
    ind = np.arange(n)
    y_pos = (n - ind - 1) * 2
    bottom = np.zeros(n)
    for yy in range(Y):
        ax.barh(y_pos, capacity[:, yy] - bottom, 1.5, left=bottom, color=capacity_colors[yy])
        bottom = capacity[:, yy]

    ax.set_xlim(0, capacity_xlim)
    ax.set_ylim(-1, n * 2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(city_labels, fontsize=fontsize - 9)
    ax.tick_params(axis="x", labelsize=fontsize - 6)
    ax.tick_params(axis="y", labelsize=fontsize - 8)
    ax.xaxis.tick_top()
    ax.set_xlabel("GW", fontsize=fontsize - 5, labelpad=5)
    add_city_group_annotation(ax, column_index, fontsize - 9, line_width)

    ax_lcoe = ax.twiny()
    for yy in range(Y):
        ax_lcoe.scatter(lcoe[:, yy], y_pos, color=lcoe_colors[yy], zorder=5, s=100)
    ax_lcoe.set_xlim(0, lcoe_xlim)
    ax_lcoe.set_ylim(ax.get_ylim())
    ax_lcoe.tick_params(axis="x", labelsize=fontsize - 6)
    ax_lcoe.set_xlabel("CNY/kWh", fontsize=fontsize - 5, labelpad=5)


def plot_fig3(city_names, ordered_city_indices, fpv_capacity, fpv_lcoe, rpv_capacity, rpv_lcoe):
    cities = [pretty_city_name(city_names[i]) for i in ordered_city_indices]
    capacity_cmap = mcolors.LinearSegmentedColormap.from_list("green_gradient", ["#E0F2E9", "#007E2E"])
    lcoe_cmap = mcolors.LinearSegmentedColormap.from_list("blue_gradient", ["#E6F0FF", "#003366"])
    capacity_colors = [capacity_cmap(i / 4) for i in range(Y)]
    lcoe_colors = [lcoe_cmap(i / 4) for i in range(Y)]

    ratio_plot = 3
    figwidth = 8.5 * ratio_plot
    fs = 8 * ratio_plot
    lw_axis = 0.8
    plt.rc("font", size=fs - 12)

    fig, axes = plt.subplots(2, 2, figsize=np.array([figwidth * 1.2, figwidth * 2.55]) / 2.54)
    city_halves = [cities[:51], cities[51:]]
    fpv_cap_halves = [fpv_capacity[:51, :], fpv_capacity[51:, :]]
    fpv_lcoe_halves = [fpv_lcoe[:51, :], fpv_lcoe[51:, :]]
    rpv_cap_halves = [rpv_capacity[:51, :], rpv_capacity[51:, :]]
    rpv_lcoe_halves = [rpv_lcoe[:51, :], rpv_lcoe[51:, :]]

    fpv_cap_xlim = rounded_upper(fpv_capacity, minimum=30, step=5)
    rpv_cap_xlim = rounded_upper(rpv_capacity, minimum=30, step=5)
    fpv_lcoe_xlim = rounded_upper(fpv_lcoe, minimum=0.8, step=0.1)
    rpv_lcoe_xlim = rounded_upper(rpv_lcoe, minimum=0.8, step=0.1)

    for col in range(2):
        draw_half_panel(axes[0, col], city_halves[col], fpv_cap_halves[col], fpv_lcoe_halves[col],
                        capacity_colors, lcoe_colors, fpv_cap_xlim, fpv_lcoe_xlim, col, fs, lw_axis)
        draw_half_panel(axes[1, col], city_halves[col], rpv_cap_halves[col], rpv_lcoe_halves[col],
                        capacity_colors, lcoe_colors, rpv_cap_xlim, rpv_lcoe_xlim, col, fs, lw_axis)

    panel_label_fs = fs + 2
    panel_title_fs = fs - 1

    fig.text(0.015, 0.986, "a", fontsize=panel_label_fs,
            fontweight="bold", va="top")
    fig.text(0.50, 0.986, "Facade photovoltaics (FPV)",
            fontsize=panel_title_fs, fontweight="bold",
            ha="center", va="top")

    fig.text(0.015, 0.507, "b", fontsize=panel_label_fs,
            fontweight="bold", va="top")
    fig.text(0.50, 0.507, "Rooftop photovoltaics (RPV)",
            fontsize=panel_title_fs, fontweight="bold",
            ha="center", va="top")
    cax = fig.add_axes([0.35, 0.018, 0.40, 0.010])
    norm = mcolors.BoundaryNorm(np.linspace(0, 1, 6), capacity_cmap.N)
    colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=capacity_cmap, norm=norm), cax=cax,
                            orientation="horizontal", ticks=np.linspace(0.1, 0.9, 5))
    colorbar.set_ticklabels([str(year) for year in YEARS])
    colorbar.ax.tick_params(labelsize=fs - 5, length=0, pad=10)
    colorbar.outline.set_visible(False)

    x_centers = np.linspace(0.39, 0.71, 5)
    for x, color in zip(x_centers, lcoe_colors):
        marker = Line2D(
            [x], [0.042],
            marker="o",
            markersize=30,
            markerfacecolor=color,
            markeredgecolor="none",
            linestyle="none",
            transform=fig.transFigure,
            zorder=10,
        )
        fig.add_artist(marker)
    fig.text(0.28, 0.042, "LCOE", ha="center", va="center", fontsize=fs - 5, fontweight="bold")
    fig.text(0.28, 0.023, "Capacity", ha="center", va="center", fontsize=fs - 5, fontweight="bold")

    plt.subplots_adjust(left=0.10, right=0.96, top=0.955, bottom=0.075, wspace=0.50, hspace=0.18)
    output_path = str(OUTPUT_DIR / "Fig3_new.pdf")
    fig.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def load_public_inputs():
    city_statistic = pd.read_excel(
        CITY_STATISTICS_FILE, sheet_name="Class_Volume", index_col=0
    )
    city_names = city_statistic.index.tolist()
    ordered_city_indices = np.load(FIG_INPUT_DIR / "Fig3_list_city.npy").astype(int)
    fpv_capacity = np.load(FIG_INPUT_DIR / "Fig3_FPV_capacity.npy")
    fpv_lcoe = np.load(FIG_INPUT_DIR / "Fig3_FPV_LCOE.npy")
    rpv_capacity = np.load(FIG_INPUT_DIR / "Fig3_RPV_capacity.npy")
    rpv_lcoe = np.load(FIG_INPUT_DIR / "Fig3_RPV_LCOE.npy")
    return (
        city_names,
        ordered_city_indices,
        fpv_capacity,
        fpv_lcoe,
        rpv_capacity,
        rpv_lcoe,
    )


def main():
    parser = argparse.ArgumentParser(description="Draw Fig. 3.")
    parser.add_argument(
        "--rebuild-inputs",
        action="store_true",
        help="Recompute compact arrays from all 102 optimization MAT files.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = (
        compute_fig3_inputs()
        if args.rebuild_inputs
        else load_public_inputs()
    )
    plot_fig3(*results)


if __name__ == "__main__":
    main()



