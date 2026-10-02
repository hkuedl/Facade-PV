import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from matplotlib import rcParams
from shapely.geometry import MultiPolygon, Polygon, shape
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("FPV_DATA_ROOT", REPOSITORY_ROOT / "##Data")).expanduser()
FIG_INPUT_DIR = Path(
    os.environ.get(
        "FPV_FIG_INPUT_DIR", DATA_ROOT / "Optimization" / "Fig_input_data"
    )
).expanduser()
ML_RESULTS_DIR = Path(
    os.environ.get("FPV_ML_RESULTS_DIR", DATA_ROOT / "Evaluation" / "All_output")
).expanduser()
CITY_FEATURE_DIR = Path(
    os.environ.get("FPV_CITY_FEATURE_DIR", DATA_ROOT / "Evaluation" / "All_input")
).expanduser()
PLANNING_RESULTS_DIR = Path(
    os.environ.get(
        "FPV_OPT_RESULTS_DIR", DATA_ROOT / "Optimization" / "Planning_results"
    )
).expanduser()
OLD_OPT_DIR = Path(
    os.environ.get("FPV_BOUNDARY_DIR", DATA_ROOT / "Optimization" / "Planning_results")
).expanduser()
CITY_STATISTICS_FILE = Path(
    os.environ.get(
        "FPV_CITY_STATISTICS_FILE",
        DATA_ROOT / "Optimization" / "Other_parameters" / "City_statistic.xlsx",
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get("FPV_FIG_OUTPUT_DIR", SCRIPT_DIR / "Figs_new")
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABNORMAL_VOLUME_CITY_INDICES = {2, 6, 8, 69, 92, 97}
CITY_INDICES = [3, 74, 59]  # Beijing, Wuhan, Shenzhen
FORMS = ["HnD", "HnS", "MnD", "MnS"]
TYPE_CMAPS = ["Blues", "Oranges", "Greens", "Purples"]
YEARS_FOR_MAP = [2030, 2040, 2050]
FPV_VMAX_MW = 60


def setup_font():
    font_path = os.path.join(SCRIPT_DIR, "arial.ttf")
    if os.path.exists(font_path):
        custom_font = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
        rcParams["font.family"] = custom_font.get_name()
    else:
        rcParams["font.family"] = "DejaVu Sans"


def planning_mat_path(city_index, city_name):
    suffix = "_hybrid_n.mat" if city_index in ABNORMAL_VOLUME_CITY_INDICES else "_hybrid.mat"
    return os.path.join(PLANNING_RESULTS_DIR, city_name + suffix)


def city_boundary_path(city_name):
    planning_path = os.path.join(PLANNING_RESULTS_DIR, city_name + ".json")
    old_path = os.path.join(OLD_OPT_DIR, city_name + ".json")
    if os.path.exists(planning_path):
        return planning_path
    if os.path.exists(old_path):
        return old_path
    raise FileNotFoundError(f"Missing city boundary json in both {planning_path} and {old_path}")


def city_circle_config(city_index):
    configs = {
        3: (116.4, 39.92, [0.18, 0.18 + 0.15, 0.18 + 0.15 + 0.13], "Beijing (SLC)"),
        74: (114.3, 30.55, [0.16, 0.16 + 0.1, 0.16 + 0.1 + 0.09], "Wuhan (VLC)"),
        59: (114.11, 22.58, [0.15, 0.15 + 0.15, 0.15 + 0.15], "Shenzhen (SLC)"),
    }
    return configs.get(city_index)


def panel_label(city_index, pv_kind):
    fpv_labels = {3: "a", 74: "b", 59: "c"}
    rpv_labels = {3: "d", 74: "e", 59: "f"}
    return fpv_labels[city_index] if pv_kind.upper() == "FPV" else rpv_labels[city_index]


def create_square(lat, lon, size_km=2.5):
    half_size_km = size_km / 2
    bottom_left = geodesic(kilometers=half_size_km).destination((lat, lon), 225)
    bottom_right = geodesic(kilometers=half_size_km).destination((lat, lon), 315)
    top_right = geodesic(kilometers=half_size_km).destination((lat, lon), 45)
    top_left = geodesic(kilometers=half_size_km).destination((lat, lon), 135)
    return Polygon([
        (bottom_left.longitude, bottom_left.latitude),
        (bottom_right.longitude, bottom_right.latitude),
        (top_right.longitude, top_right.latitude),
        (top_left.longitude, top_left.latitude),
    ])


def load_city_grid(city_index, city_name, pv_kind="FPV"):
    mat_path = planning_mat_path(city_index, city_name)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Missing optimization result: {mat_path}")

    grid_type = np.load(os.path.join(ML_RESULTS_DIR, "Grid_type", f"Grid_type_{city_name}.npy"))
    data = loadmat(mat_path)
    r_cap_f = data["R_Cap_f"]
    r_cap_r = data["R_Cap_r"]

    n_gg = np.where(grid_type[:, 0] != 888)[0]
    th_hh = 18
    th_std = [0, 0]
    th_aa = [200, 200]
    grid_valid = grid_type[n_gg, :]
    form_indices = [
        list(np.where((grid_valid[:, 1] >= th_hh) & (grid_valid[:, 2] >= th_std[1]) & (grid_valid[:, 3] >= th_aa[1]))[0]),
        list(np.where((grid_valid[:, 1] >= th_hh) & (grid_valid[:, 2] >= th_std[1]) & (grid_valid[:, 3] < th_aa[0]))[0]),
        list(np.where((grid_valid[:, 1] < th_hh) & (grid_valid[:, 2] >= th_std[0]) & (grid_valid[:, 3] >= th_aa[1]))[0]),
        list(np.where((grid_valid[:, 1] < th_hh) & (grid_valid[:, 2] >= th_std[0]) & (grid_valid[:, 3] < th_aa[0]))[0]),
    ]

    grid = pd.DataFrame(np.zeros((len(n_gg), 6)), columns=["Long.", "Lat.", "Type", "Capacity-2030(KW)", "Capacity-2040(KW)", "Capacity-2050(KW)"])

    if pv_kind.upper() == "FPV":
        cap_valid = r_cap_f[n_gg, 0, :]  # RS+F unified case, optimized FPV capacity
    elif pv_kind.upper() == "RPV":
        cap_valid = r_cap_r[n_gg, 1, :]  # RS+F unified case, rooftop capacity in the joint system
    else:
        raise ValueError(f"Unsupported pv_kind={pv_kind}")

    for form_index, form_name in enumerate(FORMS):
        rows = form_indices[form_index]
        grid.iloc[rows, 0:2] = grid_valid[rows, 6:8]
        grid.iloc[rows, 2] = form_name
        grid.iloc[rows, 3] = cap_valid[rows, 0]
        grid.iloc[rows, 4] = cap_valid[rows, 2]
        grid.iloc[rows, 5] = cap_valid[rows, 4]

    grid["geometry"] = grid.apply(lambda row: create_square(float(row["Lat."]), float(row["Long."])), axis=1)
    return grid


def draw_boundary(ax, boundary_geojson):
    for feature in boundary_geojson["features"]:
        geom = shape(feature["geometry"])
        polygons = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in polygons:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="black", linewidth=1.2)


def add_city_circles(ax, city_index):
    config = city_circle_config(city_index)
    if config is None or city_index == 59:
        return
    lon, lat, radii, _ = config
    linestyles = ["-", "--", "-."]
    for radius, linestyle in zip(radii, linestyles):
        ax.add_patch(patches.Circle((lon, lat), radius=radius, edgecolor="grey", facecolor="none", linewidth=3, linestyle=linestyle))


def capacity_ratios(grid):
    ratios = np.zeros((len(FORMS), len(YEARS_FOR_MAP)))
    for year_i, year in enumerate(YEARS_FOR_MAP):
        col = f"Capacity-{year}(KW)"
        total = grid[col].sum()
        if total <= 0:
            continue
        for form_i, form_name in enumerate(FORMS):
            ratios[form_i, year_i] = round(100 * grid.loc[grid["Type"] == form_name, col].sum() / total, 1)
        if not np.isclose(ratios[:, year_i].sum(), 100):
            ratios[0, year_i] = round(100 - ratios[1:, year_i].sum(), 1)
    return ratios


def style_map_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def compute_unified_rpv_vmax(city_names):
    max_mw = 0.0
    for cc in CITY_INDICES:
        grid = load_city_grid(cc, city_names[cc], pv_kind="RPV")
        city_max = np.nanmax(grid[[f"Capacity-{year}(KW)" for year in YEARS_FOR_MAP]].values) / 1000.0
        max_mw = max(max_mw, city_max)
    return int(max(1, np.ceil(max_mw / 10.0) * 10.0))


def render_city(city_index, city_name, pv_kind, rpv_vmax_mw):
    print(f"{pv_kind} | {city_name}")
    grid = load_city_grid(city_index, city_name, pv_kind=pv_kind)
    with open(city_boundary_path(city_name), "r", encoding="utf-8") as f:
        boundary_geojson = json.load(f)

    s_font_title = 60
    s_font_legend = 60
    s_font_label = 60
    s_font_label_title = 60

    vmax = FPV_VMAX_MW if pv_kind.upper() == "FPV" else rpv_vmax_mw
    norm = plt.Normalize(0, vmax)
    ratios = capacity_ratios(grid)

    fig, axes = plt.subplots(1, 3, figsize=(45, 15), gridspec_kw={"wspace": 0.1})

    for ax in axes:
        draw_boundary(ax, boundary_geojson)

    for form_name, cmap_name in zip(FORMS, TYPE_CMAPS):
        cmap = plt.get_cmap(cmap_name)
        subset = grid[grid["Type"] == form_name]
        for _, row in subset.iterrows():
            x, y = row["geometry"].exterior.xy
            for year_i, ax in enumerate(axes):
                col = f"Capacity-{YEARS_FOR_MAP[year_i]}(KW)"
                ax.fill(x, y, color=cmap(norm(row[col] / 1000.0)), alpha=0.9, linewidth=0)

    for year_i, ax in enumerate(axes):
        ax.set_title(str(YEARS_FOR_MAP[year_i]), fontsize=s_font_title, fontweight="bold", y=0.97)
        style_map_axis(ax)
        add_city_circles(ax, city_index)

    config = city_circle_config(city_index)
    if config is not None:
        _, _, _, city_label = config
        panel = panel_label(city_index, pv_kind)
        title_y = 1.16 if city_index == 59 else 1.13
        axes[1].text(0.5, title_y, city_label, fontsize=s_font_title, fontweight="bold", ha="center", va="center", transform=axes[1].transAxes)
        # \u4ec5\u5317\u4eac\u7684 a、d \u663e\u793a FPV/RPV；\u5176\u4ed6\u5b50\u56fe\u53ea\u663e\u793a\u57ce\u5e02\u540d\u79f0
        if city_index == 3:
            axes[1].text(
                0.5,
                title_y + 0.09,
                pv_kind,
                fontsize=s_font_title - 4,
                fontweight="bold",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
        #axes[1].text(0.5, title_y + 0.09, pv_kind, fontsize=s_font_title - 4, fontweight="bold", ha="center", va="center", transform=axes[1].transAxes)
        axes[0].text(0.0, 1.1, panel, fontweight="bold", fontsize=s_font_label + 40, ha="center", va="center", transform=axes[0].transAxes)

    if city_index == 3:
        center_line = mlines.Line2D([], [], color="gray", linestyle="-", linewidth=8, label="Center")
        expansion_line = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=8, label="Expansion")
        suburb_line = mlines.Line2D([], [], color="gray", linestyle="-.", linewidth=8, label="Suburb")
        axes[1].legend(handles=[center_line, expansion_line, suburb_line], loc="upper left", bbox_to_anchor=(-0.2, 1), frameon=True, fontsize=s_font_legend - 20)

    fig.subplots_adjust(bottom=0.3)
    # Shenzhen（c、f）：\u53f3\u4fa7\u8272\u6807\u548c\u6570\u5b57\u6574\u4f53\u4e0a\u79fb
    cbar_y = 0.29 if city_index == 59 else 0.25
    # cbar_axes = [
    #     fig.add_axes([0.14 + 0.14, 0.25, 0.13, 0.02]),
    #     fig.add_axes([0.14 + 0.34 - 0.04, 0.25, 0.13, 0.02]),
    #     fig.add_axes([0.14 + 0.54 - 0.08, 0.25, 0.13, 0.02]),
    #     fig.add_axes([0.14 + 0.74 - 0.12, 0.25, 0.13, 0.02]),
    # ]
    cbar_axes = [
        fig.add_axes([0.14 + 0.14, cbar_y, 0.13, 0.02]),
        fig.add_axes([0.14 + 0.34 - 0.04, cbar_y, 0.13, 0.02]),
        fig.add_axes([0.14 + 0.54 - 0.08, cbar_y, 0.13, 0.02]),
        fig.add_axes([0.14 + 0.74 - 0.12, cbar_y, 0.13, 0.02]),
    ]

    for form_i, (cbar_ax, form_name, cmap_name) in enumerate(zip(cbar_axes, FORMS, TYPE_CMAPS)):
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=s_font_label - 5)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.set_ticks([0, vmax])
        cbar.set_ticklabels(["0", str(int(vmax))])
        ratio_text = "->".join(str(ratios[form_i, year_i]) for year_i in range(len(YEARS_FOR_MAP)))
        cbar.set_label(f"{form_name}\n{ratio_text}", fontsize=s_font_label_title - 5, labelpad=35)

    #axes[0].text(0.25, -0.07, "Capacity (MW)", fontsize=s_font_label, ha="center", va="center", transform=axes[0].transAxes)
    #axes[0].text(0.25, -0.23, "Capacity ratio (%)", fontsize=s_font_label, ha="center", va="center", transform=axes[0].transAxes)

    # Shenzhen（c、f）：\u5de6\u4fa7\u8bf4\u660e\u6587\u5b57\u9002\u5f53\u4e0b\u79fb
    capacity_text_y = -0.19 if city_index == 59 else -0.07
    ratio_text_y = -0.38 if city_index == 59 else -0.23

    axes[0].text(
        0.25,
        capacity_text_y,
        "Capacity (MW)",
        fontsize=s_font_label,
        ha="center",
        va="center",
        transform=axes[0].transAxes,
    )
    axes[0].text(
        0.25,
        ratio_text_y,
        "Capacity ratio (%)",
        fontsize=s_font_label,
        ha="center",
        va="center",
        transform=axes[0].transAxes,
    )

    out_stub = f"Fig5_{pv_kind}_{city_name}"
    pdf_path = os.path.join(OUTPUT_DIR, f"{out_stub}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{out_stub}.png")
    fig.savefig(pdf_path, format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")

setup_font()
city_statistic = pd.read_excel(CITY_STATISTICS_FILE, sheet_name="Class_Volume", index_col=0)
city_names = city_statistic.index.tolist()
rpv_vmax_mw = 90
#rpv_vmax_mw = compute_unified_rpv_vmax(city_names)
print(f"Unified RPV vmax = {rpv_vmax_mw} MW")

for pv_kind in ["FPV", "RPV"]:
    for cc in CITY_INDICES:
        render_city(cc, city_names[cc], pv_kind=pv_kind, rpv_vmax_mw=rpv_vmax_mw)



