import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
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
CITY_STATISTICS_FILE = Path(
    os.environ.get(
        "FPV_CITY_STATISTICS_FILE",
        DATA_ROOT / "Optimization" / "Other_parameters" / "City_statistic.xlsx",
    )
).expanduser()
FIG_OUTPUT_DIR = Path(
    os.environ.get("FPV_FIG_OUTPUT_DIR", SCRIPT_DIR / "Figs_new")
).expanduser()
FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABNORMAL_VOLUME_CITY_INDICES = {2, 6, 8, 69, 92, 97}
CITY_INDICES = [3, 74, 59]
MONTH_ORDER = [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2]


def setup_font():
    font_path = os.path.join(SCRIPT_DIR, "arial.ttf")
    if os.path.exists(font_path):
        custom_font = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
        rcParams["font.family"] = custom_font.get_name()


def opt_result_path(city_index, city_name):
    suffix = "_hybrid_n.mat" if city_index in ABNORMAL_VOLUME_CITY_INDICES else "_hybrid.mat"
    return os.path.join(PLANNING_RESULTS_DIR, city_name + suffix)


def save_panel_pdf(fig, panel_name, city_name, city_index):
    if city_index == 3:
        filename = f"Fig6{panel_name}{city_name}.pdf"
    else:
        filename = f"sFig6{panel_name}{city_name}.pdf"
    out_path = os.path.join(FIG_OUTPUT_DIR, filename)
    fig.savefig(out_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def get_hnd_indices(city_name, grid_type):
    cached_path = str(FIG_INPUT_DIR / f"list_form_{city_name}.npy")
    if os.path.exists(cached_path):
        return np.load(cached_path).astype(int)

    valid = np.where(grid_type[:, 0] != 888)[0]
    grid_valid = grid_type[valid, :]
    relative_hnd = np.where(
        (grid_valid[:, 1] >= 18)
        & (grid_valid[:, 2] >= 0)
        & (grid_valid[:, 3] >= 200)
    )[0]
    return valid[relative_hnd]


def load_city_data(city_index, city_name):
    result_path = opt_result_path(city_index, city_name)
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Missing optimization result: {result_path}")

    grid_type = np.load(os.path.join(ML_RESULTS_DIR, "Grid_type", f"Grid_type_{city_name}.npy"))
    feature_path = os.path.join(CITY_FEATURE_DIR, f"{city_name}_ALL_Featuers.npy")
    if os.path.exists(feature_path):
        _ = np.load(feature_path)[:, [i for i in range(14)] + [15, 16]]

    data = loadmat(result_path)
    required = [
        "R_Cap_r",
        "R_Cap_s",
        "R_Cap_f",
        "R_Pow",
        "R_Pow_f",
        "R_Pow_ch",
        "R_Pow_dis",
        "R_Pow_G",
        "R_Pow_r",
        "R_Pow_Buy",
        "R_Pow_AB",
        "R_Car_t",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{result_path} misses variables: {missing}")

    return grid_type, {key: data[key] for key in required}


def plot_load_balance(city_index, city_name, list_form, arrays, case_index, s_font=30):
    r_pow = arrays["R_Pow"]
    r_pow_f = arrays["R_Pow_f"]
    r_pow_ch = arrays["R_Pow_ch"]
    r_pow_dis = arrays["R_Pow_dis"]
    r_pow_g = arrays["R_Pow_G"]
    r_pow_r = arrays["R_Pow_r"]
    r_pow_buy = arrays["R_Pow_Buy"]
    r_pow_ab = arrays["R_Pow_AB"]

    pu_ele = np.max(np.sum(r_pow[list_form, :, :, :], axis=0))
    load_curve = (np.sum(r_pow[list_form, :, :, :], axis=(0, 1)) / 5 / pu_ele)[MONTH_ORDER].ravel()
    rpv_output = (np.sum(r_pow_r[list_form, case_index, :, :, :], axis=(0, 1)) / 5 / pu_ele)[MONTH_ORDER].ravel()
    fpv_output = (case_index * np.sum(r_pow_f[list_form, :, :, :], axis=(0, 1)) / 5 / pu_ele)[MONTH_ORDER].ravel()
    storage_charge = (np.sum(r_pow_ch[list_form, case_index, :, :, :], axis=(0, 1)) / 5 / pu_ele)[MONTH_ORDER].ravel()
    storage_discharge = (np.sum(r_pow_dis[list_form, case_index, :, :, :], axis=(0, 1)) / 5 / pu_ele)[MONTH_ORDER].ravel()
    grid_purchase = (np.sum(r_pow_buy[list_form, case_index, :, :, :], axis=(0, 1)) / 5 / pu_ele)[MONTH_ORDER].ravel()
    grid_discard = (np.sum(r_pow_ab[list_form, case_index, :, :, :], axis=(0, 1)) / 5 / pu_ele)[MONTH_ORDER].ravel()
    grid_sell = (np.sum(r_pow_g[list_form, case_index, :, :, :], axis=(0, 1)) / 5 / pu_ele)[MONTH_ORDER].ravel()

    hours = np.arange(0, 24 * 12)
    fig, ax = plt.subplots(figsize=(18, 6))
    background = [0, 24 * 3, 24 * 6, 24 * 9, 24 * 12]
    backgroundcolor = ["whitesmoke", "lightgrey", "whitesmoke", "lightgrey"]
    for i in range(4):
        ax.axvspan(background[i], background[i + 1], facecolor=backgroundcolor[i], alpha=0.5)

    colors = ["skyblue", "mediumseagreen", "moccasin", "tomato", "darksalmon", "darkorange", "orange"]
    ax.plot(hours, load_curve, label="Load", color="black", linestyle="--", linewidth=2)
    ax.stackplot(
        hours,
        rpv_output,
        fpv_output,
        grid_purchase,
        storage_discharge,
        storage_charge,
        grid_discard,
        grid_sell,
        labels=["RPV", "FPV", "Purchase", "Discharge", "Charge", "Discard", "Sell"],
        colors=colors,
    )
    ax.set_ylabel("p.u.", fontsize=s_font + 5)
    if case_index == 0:
        ax.set_title("Load balance under RS", fontsize=s_font, fontweight="bold", y=0.9)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.3),
            ncol=4,
            frameon=False,
            fontsize=s_font - 5,
            labelspacing=0.8,
        )
        panel_name = "a"
    else:
        ax.set_title("Load balance under RS+F", fontsize=s_font, fontweight="bold", y=0.9)
        panel_name = "b"

    ax.set_xticks([12 * 3, 36 * 3, 60 * 3, 84 * 3])
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=s_font - 5)
    ax.set_ylim(0, 1)
    ax.set_xlim(-2, 24 * 12 + 2)
    save_panel_pdf(fig, panel_name, city_name, city_index)


def plot_carbon_emission(city_index, city_name, list_form, arrays, s_font=30):
    r_car_t = arrays["R_Car_t"]
    pu_car = np.max(np.sum(r_car_t[list_form, 0, :, :, :], axis=0))
    values_rs = (np.sum(r_car_t[list_form, 0, :, :, :], axis=(0, 1)) / 5 / pu_car)[MONTH_ORDER].ravel()
    values_rsf = (np.sum(r_car_t[list_form, 1, :, :, :], axis=(0, 1)) / 5 / pu_car)[MONTH_ORDER].ravel()

    x = np.arange(0, 24 * 12, 1)
    fig, ax = plt.subplots(figsize=(18, 6))
    background = [0, 24 * 3, 24 * 6, 24 * 9, 24 * 12]
    backgroundcolor = ["whitesmoke", "lightgrey", "whitesmoke", "lightgrey"]
    for i in range(4):
        ax.axvspan(background[i], background[i + 1], facecolor=backgroundcolor[i], alpha=0.5)

    ax.plot(x, values_rs, label="RS", color="deepskyblue", linestyle="--", linewidth=2.5, alpha=1)
    ax.plot(x, values_rsf, label="RS+F", color="seagreen", linestyle="-", linewidth=2.5, alpha=1)
    ax.set_title("Carbon emission", fontsize=s_font + 2, fontweight="bold", y=0.9)
    ax.set_ylabel("p.u.", fontsize=s_font + 7)
    ax.set_xlim(-2, 24 * 12 + 2)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xticks([12 * i for i in range(1, 24, 2)])
    ax.set_xticklabels(["Day1", "Day2", "Day3"] * 4, fontsize=s_font)
    ax.tick_params(axis="y", labelsize=s_font - 3)

    season_x = [12 * 3, 36 * 3, 60 * 3, 84 * 3]
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    for x_pos, season in zip(season_x, seasons):
        ax.text(x_pos, -0.12, season, fontsize=s_font + 2, color="k", ha="center", va="center")

    ax.legend(loc="upper left", fontsize=s_font)
    plt.tight_layout()
    save_panel_pdf(fig, "c", city_name, city_index)


def main():
    setup_font()
    city_statistic = pd.read_excel(
        CITY_STATISTICS_FILE,
        sheet_name="Class_Volume",
        index_col=0,
    )

    for city_index in CITY_INDICES:
        city_name = city_statistic.index[city_index]
        print(city_name)
        grid_type, arrays = load_city_data(city_index, city_name)
        list_form = get_hnd_indices(city_name, grid_type)
        print(f"HnD grids used for Fig. 6: {len(list_form)}")

        plot_load_balance(city_index, city_name, list_form, arrays, case_index=0)
        plot_load_balance(city_index, city_name, list_form, arrays, case_index=1)
        plot_carbon_emission(city_index, city_name, list_form, arrays)


if __name__ == "__main__":
    main()

