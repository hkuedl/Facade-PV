"""Generate the grid-specific annual load profiles used in the study.

This version replaces the former uniform 80/20 residential/commercial split
with land-use-informed residential, office, and commercial-service shares.
Set FPV_DATA_ROOT to the downloaded ##Data directory when it is not stored at
the repository root.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(
    os.environ.get("FPV_DATA_ROOT", REPOSITORY_ROOT / "##Data")
).expanduser()
LOAD_INPUT_DIR = Path(
    os.environ.get("FPV_LOAD_INPUT_DIR", DATA_ROOT / "Optimization" / "Loads")
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get(
        "FPV_LOAD_OUTPUT_DIR",
        DATA_ROOT / "Optimization" / "Loads" / "Landuse_Loads",
    )
).expanduser()
ML_RESULTS_DIR = Path(
    os.environ.get(
        "FPV_ML_RESULTS_DIR", DATA_ROOT / "Evaluation" / "All_output"
    )
).expanduser()
CITY_FEATURE_DIR = Path(
    os.environ.get(
        "FPV_CITY_FEATURE_DIR", DATA_ROOT / "Evaluation" / "All_input"
    )
).expanduser()
CITY_STATISTICS_FILE = Path(
    os.environ.get(
        "FPV_CITY_STATISTICS_FILE",
        DATA_ROOT / "Optimization" / "Other_parameters" / "City_statistic.xlsx",
    )
).expanduser()

LANDUSE_CANDIDATES = [
    Path(os.environ["FPV_LANDUSE_FILE"]).expanduser()
    if "FPV_LANDUSE_FILE" in os.environ
    else DATA_ROOT / "Optimization" / "Other_parameters" / "Landuse.xlsx",
    LOAD_INPUT_DIR / "Landuse.xlsx",
    SCRIPT_DIR / "Landuse.xlsx",
]

CITY_COUNT = 102
GRID_SIZE_M = 2000.0
FALLBACK_K_NEAREST = 16
RANDOM_SEED = 20

LAND_USE_CODES = ["101", "201", "202"]
LOAD_NAMES = ["residential", "office", "commercial_service"]
SUPPLEMENTARY_INFO_CITY_INDICES = {0, 1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 93, 94}
ABNORMAL_VOLUME_CITY_INDICES = {2, 6, 8, 69, 92, 97}

EER_C = 3.3
EER_H = 3.0
HEATING_ELECTRIFICATION_NORTH = 0.4
HEATING_ELECTRIFICATION_SOUTH = 0.6


LAND_USE_MAPPING = {
    "101": "Residential",
    "201": "Commercial office",
    "202": "Commercial service",
    "301": "Industrial",
    "401": "Road",
    "402": "Transport station",
    "403": "Airport",
    "501": "Government",
    "502": "Education/research",
    "503": "Medical",
    "504": "Sport/culture",
    "505": "Park/green space",
}

LAND_USE_TO_BUILDING_TYPES = {
    "Residential": [
        "Terraced house",
        "Low-rise",
        "High-rise (slab-type)",
        "High-rise (tower-type)",
    ],
    "Commercial office": ["Commercial office A", "Commercial office B"],
    "Commercial service": ["Small hotel", "Large hotel", "Shopping mall"],
}

BUILDING_TYPES_TO_ABBREVIATIONS = {
    "Terraced house": "Th",
    "Low-rise": "Low",
    "High-rise (slab-type)": "HighS",
    "High-rise (tower-type)": "HighT",
    "Commercial office A": "CoA",
    "Commercial office B": "CoB",
    "Small hotel": "SH",
    "Large hotel": "LH",
    "Shopping mall": "Mall",
}

ABBREVIATION_TO_COEFFICIENTS = {
    "Th": 0.1,
    "Low": 0.3,
    "HighS": 0.3,
    "HighT": 0.3,
    "CoA": 0.8,
    "CoB": 0.2,
    "SH": 0.1,
    "LH": 0.2,
    "Mall": 0.7,
}

ABBREVIATION_TO_STORY_HEIGHTS = {
    "Th": 3.0,
    "Low": 3.0,
    "HighS": 3.0,
    "HighT": 3.0,
    "CoA": 3.0,
    "CoB": 3.0,
    "SH": 3.0,
    "LH": 3.0,
    "Mall": 3.0,
}


def find_first_existing(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of these files exist: " + ", ".join(map(str, candidates)))


def read_landuse():
    landuse_path = find_first_existing(LANDUSE_CANDIDATES)
    landuse = pd.read_excel(landuse_path)
    landuse = landuse.rename(
        columns={
            landuse.columns[0]: "lon",
            landuse.columns[1]: "lat",
            landuse.columns[2]: "level1",
            landuse.columns[3]: "level2",
        }
    )
    landuse = landuse[["lon", "lat", "level1", "level2"]].dropna()
    landuse["level1"] = landuse["level1"].astype(int)
    landuse["level2"] = landuse["level2"].astype(int)
    return landuse


def maybe_projected_to_lonlat(xy):
    xy = np.asarray(xy, dtype=float)
    if np.nanmax(np.abs(xy[:, 0])) <= 180 and np.nanmax(np.abs(xy[:, 1])) <= 90:
        return xy.copy()

    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lon, lat])

def landuse_row_weights(level1, level2):
    """Return moderated weights for 101 residential, 201 office, and 202 service loads.

    This version is between the original aggressive mapping and the overly
    conservative mapping: it expands office/service shares for plausible
    non-residential land uses, but keeps a residential base to avoid strongly
    inflating total load magnitude.
    """
    code = str(int(level2))
    lvl1 = int(level1)

    # Residential land remains residential.
    if code == "101" or lvl1 == 1:
        return np.array([1.0, 0.0, 0.0])

    # Clear commercial/office land uses: stronger than conservative version,
    # but still not pure office/service.
    if code == "201":
        return np.array([0.45, 0.55, 0.0])
    if code == "202":
        return np.array([0.45, 0.0, 0.55])

    # Industrial: some service/office load, but still mostly residential-like
    # to avoid excessive load inflation.
    if code == "301":
        return np.array([0.75, 0.10, 0.15])

    # Road: mostly residential-like, because road pixels should not strongly
    # add building load.
    if code == "401":
        return np.array([0.90, 0.05, 0.05])

    # Transport station / airport: allow more commercial-service behavior.
    if code in {"402", "403"}:
        return np.array([0.70, 0.10, 0.20])

    # Government and education: office-like schedules, but moderated.
    if code == "501":
        return np.array([0.65, 0.35, 0.0])
    if code == "502":
        return np.array([0.70, 0.30, 0.0])

    # Medical: mixed office/service behavior.
    if code == "503":
        return np.array([0.65, 0.20, 0.15])

    # Sport/culture: some service-like load.
    if code == "504":
        return np.array([0.65, 0.15, 0.20])

    # Park/green space: keep residential-like.
    if code == "505":
        return np.array([1.0, 0.0, 0.0])

    # Fallbacks by first-level land-use class.
    if lvl1 == 2:
        return np.array([0.55, 0.225, 0.225])
    if lvl1 == 3:
        return np.array([0.75, 0.10, 0.15])
    if lvl1 == 4:
        return np.array([0.85, 0.05, 0.10])
    if lvl1 == 5:
        return np.array([0.75, 0.20, 0.05])

    return np.array([0.8, 0.1, 0.1])


def subset_landuse_for_city(landuse, centers_lonlat):
    pad = 0.25
    lon_min, lat_min = centers_lonlat.min(axis=0) - pad
    lon_max, lat_max = centers_lonlat.max(axis=0) + pad
    city_landuse = landuse[
        (landuse["lon"] >= lon_min)
        & (landuse["lon"] <= lon_max)
        & (landuse["lat"] >= lat_min)
        & (landuse["lat"] <= lat_max)
    ].copy()
    if len(city_landuse) == 0:
        city_landuse = landuse.copy()
    return city_landuse


def compute_landuse_load_weights(centers_lonlat, city_landuse):
    from scipy.spatial import cKDTree

    points = city_landuse[["lon", "lat"]].to_numpy(dtype=float)
    level1 = city_landuse["level1"].to_numpy(dtype=int)
    level2 = city_landuse["level2"].to_numpy(dtype=int)
    order = np.argsort(points[:, 0])
    lon_sorted = points[order, 0]
    lat_sorted = points[order, 1]
    tree = cKDTree(points)

    weights = np.zeros((centers_lonlat.shape[0], 3), dtype=float)
    matched_points = np.zeros(centers_lonlat.shape[0], dtype=int)
    used_fallback = np.zeros(centers_lonlat.shape[0], dtype=bool)
    dominant_level2 = np.zeros(centers_lonlat.shape[0], dtype=int)

    half_lat = (GRID_SIZE_M / 2.0) / 111_320.0

    for i, (lon0, lat0) in enumerate(centers_lonlat):
        cos_lat = max(np.cos(np.deg2rad(lat0)), 0.2)
        half_lon = (GRID_SIZE_M / 2.0) / (111_320.0 * cos_lat)
        left = np.searchsorted(lon_sorted, lon0 - half_lon, side="left")
        right = np.searchsorted(lon_sorted, lon0 + half_lon, side="right")
        candidate_order = order[left:right]
        candidate_order = candidate_order[
            np.abs(points[candidate_order, 1] - lat0) <= half_lat
        ]

        if len(candidate_order) == 0:
            k = min(FALLBACK_K_NEAREST, len(points))
            _, nearest = tree.query([lon0, lat0], k=k)
            candidate_order = np.atleast_1d(nearest)
            used_fallback[i] = True

        matched_points[i] = len(candidate_order)
        row_weights = np.vstack(
            [landuse_row_weights(level1[j], level2[j]) for j in candidate_order]
        )
        weights[i] = row_weights.mean(axis=0)

        values, counts = np.unique(level2[candidate_order], return_counts=True)
        dominant_level2[i] = int(values[np.argmax(counts)])

    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights, matched_points, used_fallback, dominant_level2


def city_heating_rate(city_name, city_north):
    return HEATING_ELECTRIFICATION_SOUTH if city_north.loc[city_name, "North"] == 0 else HEATING_ELECTRIFICATION_NORTH


def read_city_info(city_name, city_index):
    feature_path = CITY_FEATURE_DIR / f"{city_name}_ALL_Featuers.npy"
    all_features = np.load(feature_path)
    static = all_features[:, [i for i in range(14)] + [15, 16]]
    non_zero = np.where(static[:, 11] != 0)[0]

    if city_index in SUPPLEMENTARY_INFO_CITY_INDICES:
        info = np.load(CITY_FEATURE_DIR / f"{city_name}_ALL_Featuers_supplementary.npy")
    else:
        info = all_features[:, range(17, 29)]

    grid_type = np.load(ML_RESULTS_DIR / "Grid_type" / f"Grid_type_{city_name}.npy")
    if grid_type.shape[0] != len(non_zero):
        raise ValueError(
            f"{city_name}: Grid_type rows ({grid_type.shape[0]}) do not match "
            f"non-zero feature grids ({len(non_zero)})."
        )
    return all_features, static, info, non_zero, grid_type


def make_load_candidates(
    city_name,
    city_index,
    grid_type,
    info,
    non_zero,
    loads_per_area,
):
    load_candidates = [
        np.zeros((grid_type.shape[0], 3, 8760), dtype=np.float32) for _ in LAND_USE_CODES
    ]
    volume_col = 8 if city_index in ABNORMAL_VOLUME_CITY_INDICES else 6

    for grid_idx in range(grid_type.shape[0]):
        if grid_idx % 1000 == 0:
            print(f"  grid {grid_idx}/{grid_type.shape[0]}")

        building_volume_land_use = info[non_zero, :][grid_idx, volume_col]

        for load_type_idx, land_use_code in enumerate(LAND_USE_CODES):
            land_use_type = LAND_USE_MAPPING[land_use_code]
            loads_city_ele = []
            loads_city_heat = []
            loads_city_cool = []

            for building_type in LAND_USE_TO_BUILDING_TYPES[land_use_type]:
                abbr = BUILDING_TYPES_TO_ABBREVIATIONS[building_type]
                volume_coefficient = ABBREVIATION_TO_COEFFICIENTS[abbr]
                story_height = ABBREVIATION_TO_STORY_HEIGHTS[abbr]
                building_area = building_volume_land_use * volume_coefficient / story_height

                per_area = loads_per_area.loc[city_name][land_use_code + abbr]
                real_ele = per_area["electricity"]
                if land_use_code == "101" and real_ele.sum() < 0.8 * per_area["heat"].sum():
                    real_ele = per_area["electricity"]

                loads_city_ele.append(real_ele.to_numpy() * building_area)
                loads_city_heat.append(per_area["heat"].to_numpy() * building_area)
                loads_city_cool.append(per_area["cool"].to_numpy() * building_area)

            load_candidates[load_type_idx][grid_idx, 0, :] = np.sum(loads_city_ele, axis=0)
            load_candidates[load_type_idx][grid_idx, 1, :] = np.sum(loads_city_heat, axis=0)
            load_candidates[load_type_idx][grid_idx, 2, :] = np.sum(loads_city_cool, axis=0)

    return load_candidates


def mix_load_candidates(load_candidates, weights):
    mixed = np.zeros_like(load_candidates[0], dtype=np.float32)
    for i in range(len(load_candidates)):
        mixed += weights[:, i, None, None].astype(np.float32) * load_candidates[i]
    return mixed


def normalized_daily_profiles(mixed_load, heating_rate):
    total_load = (
        mixed_load[:, 0, :]
        + mixed_load[:, 2, :] / EER_C
        + heating_rate * mixed_load[:, 1, :] / EER_H
    )
    daily_profile = total_load.reshape(total_load.shape[0], 365, 24).mean(axis=1)
    max_load = np.nanmax(daily_profile, axis=1, keepdims=True)
    max_load[max_load <= 0] = 1.0
    return daily_profile / max_load


def select_profile_sample(residential_share, profiles, target, n=30):
    idx = np.argsort(np.abs(residential_share - target))[: min(n, len(residential_share))]
    sample = profiles[idx]
    median_profile = np.median(sample, axis=0)
    typical_idx = idx[np.argmin(np.sum((sample - median_profile) ** 2, axis=1))]
    return idx, typical_idx


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    cities_hku_dest = pd.read_hdf(
        LOAD_INPUT_DIR / "cities_hku_dest.hdf", key="cities_hku_dest"
    )
    loads_per_area = pd.read_hdf(
        LOAD_INPUT_DIR / "loads_diff_city_diff_building_type_per_area.hdf",
        key="loads_diff_city_diff_building_type_per_area",
    )
    building_volume_df = pd.read_excel(
        CITY_STATISTICS_FILE, index_col=0
    ).drop(columns="Total(km3)")
    building_volume_df.columns = building_volume_df.columns.astype(str)
    city_north = pd.read_excel(LOAD_INPUT_DIR / "City_north_south.xlsx", index_col=0)
    landuse = read_landuse()
    
    summary_frames = []
    profile_blocks = []
    
    for city_index in range(CITY_COUNT):
        city_name = building_volume_df.index[city_index]
        city_dest = cities_hku_dest[
            cities_hku_dest["city_HKU_en"] == city_name
        ].iloc[0]["city_DeST"]
        print(f"============================== #{city_index}, {city_name}, {city_dest} ==============================")
    
        _, static, info, non_zero, grid_type = read_city_info(city_name, city_index)
        centers_lonlat = maybe_projected_to_lonlat(static[non_zero, :2])
        city_landuse = subset_landuse_for_city(landuse, centers_lonlat)
        weights, matched_points, used_fallback, dominant_level2 = compute_landuse_load_weights(
            centers_lonlat, city_landuse
        )
    
        load_candidates = make_load_candidates(
            city_name, city_index, grid_type, info, non_zero, loads_per_area
        )
        mixed_load = mix_load_candidates(load_candidates, weights)
    
        np.save(OUTPUT_DIR / f"Load_{city_name}_hybrid.npy", mixed_load)
        print(
            f"  saved {OUTPUT_DIR / ('Load_' + city_name + '_hybrid.npy')} | "
            f"annual load={mixed_load.sum() / 1e9:.3f} TWh-components"
        )
    
        city_summary = pd.DataFrame(
            {
                "city": city_name,
                "grid_index": np.arange(grid_type.shape[0]),
                "source_feature_index": non_zero,
                "original_level2": grid_type[:, 0].astype(int),
                "dominant_landuse_level2": dominant_level2,
                "matched_landuse_points": matched_points,
                "used_nearest_fallback": used_fallback,
                "residential_share": weights[:, 0],
                "office_share": weights[:, 1],
                "commercial_service_share": weights[:, 2],
                "commercial_office_share": weights[:, 1] + weights[:, 2],
                "grid_lon": centers_lonlat[:, 0],
                "grid_lat": centers_lonlat[:, 1],
            }
        )
        city_summary.to_csv(OUTPUT_DIR / f"Load_split_{city_name}.csv", index=False)
        summary_frames.append(city_summary)
    
        heating_rate = city_heating_rate(city_name, city_north)
        profile_blocks.append(normalized_daily_profiles(mixed_load, heating_rate).astype(np.float32))
    
    summary_df = pd.concat(summary_frames, ignore_index=True)
    profiles = np.vstack(profile_blocks)
    summary_df.to_csv(OUTPUT_DIR / "load_split_summary.csv", index=False)


if __name__ == "__main__":
    main()
