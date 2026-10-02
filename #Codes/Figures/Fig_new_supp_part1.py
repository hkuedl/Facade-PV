#%% python 3.11
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
from matplotlib import rcParams
import os
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("FPV_DATA_ROOT", REPOSITORY_ROOT / "##Data")).expanduser()
OPTIMIZATION_DATA_DIR = DATA_ROOT / "Optimization"
FIG_INPUT_DIR = Path(
    os.environ.get("FPV_FIG_INPUT_DIR", OPTIMIZATION_DATA_DIR / "Fig_input_data")
).expanduser()
MAIN_CASE_DIR = str(FIG_INPUT_DIR)
os.chdir(OPTIMIZATION_DATA_DIR)

font_path = 'arial.ttf'
custom_font = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
rcParams['font.family'] = custom_font.get_name()
## ------------------------ Plotting parameter setup ------------------------
ratio_plot = 3  # Scaling factor for figure-related sizes. Using a larger figure ensures clarity even after conversion to bitmap.

figwidth =  8.5 * ratio_plot  # In centimeters, but the default unit is inches (1 inch = 2.54 cm)
fs = 8 * ratio_plot  # Minimum font size, used for axis labels (6~14pt)
lw = 0.4 * ratio_plot  # Axis line width (0.25~1.5pt)
lw2 = 0.75 * ratio_plot  # Curve line width
lw3 = 1 * ratio_plot  # Bold curve line width

grid_alpha = 0.5
# bar_width = 0.3

import numpy as np

MAIN_CASE_PATH = os.path.join(MAIN_CASE_DIR, 'main_case_city_totals.csv')


def assign_city_level(population_table):
    population_table = population_table.copy()
    city_level_map = {
        'II\u578b\u5927\u57ce\u5e02': 0,
        'I\u578b\u5927\u57ce\u5e02': 1,
        '\u7279\u5927\u57ce\u5e02': 2,
        '\u8d85\u5927\u57ce\u5e02': 3
    }
    population_table['city_level'] = population_table['\u89c4\u6a21\u7b49\u7ea7'].map(city_level_map)
    population_table[['\u57ce\u533a\u4eba\u53e3', '\u5e38\u4f4f\u4eba\u53e3', '\u57ce\u9547\u5316\u7387']] = (
        population_table[['\u57ce\u533a\u4eba\u53e3', '\u5e38\u4f4f\u4eba\u53e3', '\u57ce\u9547\u5316\u7387']].astype(float)
    )

    missing_level = population_table['city_level'].isna()
    if missing_level.any():
        urban_pop = population_table.loc[missing_level, '\u57ce\u533a\u4eba\u53e3']
        population_table.loc[missing_level, 'city_level'] = np.select(
            [
                urban_pop >= 1000,
                urban_pop >= 500,
                urban_pop >= 300,
                urban_pop >= 99,
            ],
            [3, 2, 1, 0],
            default=np.nan,
        )

    if population_table['city_level'].isna().any():
        missing_cities = population_table.index[population_table['city_level'].isna()].tolist()
        raise ValueError(f'Cannot assign city_level for: {missing_cities}')

    population_table['city_level'] = population_table['city_level'].astype(int)
    return population_table


area_df = pd.read_excel('Fig_input_data/City_statistic.xlsx', sheet_name='Key_information',
        usecols=[0, 3, 4, 5], index_col=[0])

# Capacity data contains data at different stages. 'Roof' is rooftop capacity, 'Ideal' is ideal facade capacity,
# and 'Facade' is cost-effective facade capacity. Unit: GW
# The numbers at the end represent different development stages, as PV cost and efficiency evolve,
# so 'Ideal' values differ by stage
cap_df = pd.read_excel('Fig_input_data/City_Cap.xlsx', index_col=[0])

# Convert DataFrame to numpy arrays
area_data = area_df.to_numpy()
cap_data = cap_df.to_numpy()

# ------------------- Population of selected cities -------------------
population_df = pd.read_excel('Fig_input_data/City_info.xlsx', sheet_name=0,
        usecols=list(np.arange(8)))  # The last column 'City' is the index of the cities in the study

population_df.set_index('City', inplace=True)

population_df = assign_city_level(population_df)

# ------------------- Average building height data -------------------
# File path
file_path = 'Fig_input_data/City_statistic.xlsx'  # Path to the Excel file
# Read total building volume
building_volume_df = pd.read_excel(file_path, index_col=0)
total_building_volume = building_volume_df.iloc[:, 0].copy()
# Read total rooftop area of buildings
key_information = pd.read_excel(file_path, sheet_name=2, index_col=0)
total_building_area = key_information['Area_roof-0(km2)']
# Calculate average building height (m)
average_building_height = total_building_volume / total_building_area * 1000
average_building_height.name = 'average_building_height'

# ------------------- Read power curve data -------------------
# Read .npy files
file_path = 'Fig_input_data/'
ideal_facade_data_1_path=file_path+'Power_facade_ideal_1.npy'
ideal_facade_data_1 = np.sum((np.load(ideal_facade_data_1_path)),axis=1)/1e6

real_facade_data_1_path=file_path+'Power_facade_1.npy'
real_facade_data_1 = np.sum((np.load(real_facade_data_1_path)),axis=1)/1e6

roof_data_1_path = file_path+'Power_roof_1.npy'
roof_data_1 = np.sum(np.load(roof_data_1_path),axis=1)/1e6

# ------------------- City geographic and other information -------------------
cities_hku_dest = pd.read_hdf('Fig_input_data/cities_hku_dest.hdf', key='cities_hku_dest')
## City Mapping


os.makedirs("Figs_new_supp", exist_ok=True)

main_case = pd.read_csv(MAIN_CASE_PATH).set_index('City')
main_case = main_case.reindex(area_df.index)
main_case_required_cols = [
    'rpv_installable_area_km2',
    'fpv_installable_area_km2',
    'rpv_generation_TWh',
    'fpv_generation_TWh',
]
if main_case[main_case_required_cols].isna().any().any():
    missing_cities = main_case.index[
        main_case[main_case_required_cols].isna().any(axis=1)
    ].tolist()
    raise ValueError(f'Missing cities in main case: {missing_cities}')

# Specify the file path
file_path = "Fig_input_data/City_adcode.xlsx"

# Ensure the file exists
if os.path.exists(file_path):
    # Read the Excel data
    city_adcode_df = pd.read_excel(file_path, engine='openpyxl', index_col=0)
    city_adcode_df = city_adcode_df[~city_adcode_df.index.duplicated(keep='first')].copy()
    print("Data loaded successfully!")
    print(city_adcode_df.head())  # Display the first few rows of the data
else:
    print(f"The file {file_path} does not exist. Please check if the path is correct.")

# Specify the path to store the result HDF file
result_folder = r'Fig_input_data'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_roof_ideal_1_sum_all_df.h5')

# If the HDF file already exists, read the data directly; otherwise, perform data processing and save the result
Power_roof_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_facade_ideal_1_sum_all_df.h5')

Power_facade_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_feas_wea_all_df.h5')

Grid_feas_wea_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

solar_df = Grid_feas_wea_all_df.groupby(level=0).mean()/1000  # kWh/m²

#%% Fig. 2. Facade Area and Power Generation
## Data Preparation
df_plot = area_df.iloc[:, :2].copy()  # km^2

df_plot['Area_roof-0(km2)'] = main_case['rpv_installable_area_km2'].values
df_plot['Area_facade-0(km2)'] = main_case['fpv_installable_area_km2'].values
roof_power = main_case['rpv_generation_TWh'].values  # TWh
facade_power = main_case['fpv_generation_TWh'].values  # TWh
df_plot['roof_power'] = roof_power
df_plot['facade_power'] = facade_power

# Merge average building height data
df_plot = df_plot.merge(average_building_height, left_index=True, right_index=True)

# Calculate the ratio of facade to roof in terms of area and power generation
df_plot['facade_roof_ratio_area'] = df_plot['Area_facade-0(km2)'] / df_plot['Area_roof-0(km2)']
df_plot['facade_roof_ratio_generation'] = df_plot['facade_power'] / df_plot['roof_power']

# Add 'city_adcode' column
df_plot = df_plot.merge(city_adcode_df, left_index=True, right_index=True, how='left')

# Convert to int
df_plot['city_adcode'] = df_plot['city_adcode'].astype(int)

import geopandas as gpd
import pyproj
file_path_taiwan = 'Fig_input_data/\u53f0\u6e7e\u77e2\u91cf\u5730\u56feshp\u6570\u636e/'
taiwan = gpd.read_file(file_path_taiwan + '\u53f0\u6e7e\u7701-\u5e02\u77e2\u91cfshp.shp').to_crs(epsg=4326)

# Project the GeoDataFrame from WGS84 (EPSG:4326) to Azimuthal Equidistant projection,
# specifying the central longitude and latitude (105°E, 35°N)
proj_aeqd = (
    "+proj=aeqd "
    "+lat_0=35 "   # Central latitude
    "+lon_0=105 "  # Central longitude
    "+datum=WGS84 "
    "+units=m "
    "+no_defs "
)
aeqd_crs = pyproj.CRS.from_proj4(proj_aeqd)

# Convert to the Azimuthal Equidistant projection
taiwan_aeqd = taiwan.to_crs(aeqd_crs)


#% Subfigure (b)
## Facade generation
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import LAND
from scipy.ndimage import gaussian_filter
from matplotlib.cm import ScalarMappable

import frykit.plot as fplt
import frykit.shp as fshp


def normalize_city_table(city_table):
    city_table = city_table.copy()
    if 'city_adcode' in city_table.columns:
        return city_table

    if 'ct_adcode' in city_table.columns:
        city_table['city_adcode'] = city_table['ct_adcode']
        return city_table

    if city_table.index.name is not None:
        idx_name = str(city_table.index.name)
        normalized_idx_name = idx_name.lower().replace('_', '').replace('-', '').replace(' ', '')
        if normalized_idx_name in {'cityadcode', 'adcode', 'code'} or 'adcode' in normalized_idx_name:
            city_table = city_table.reset_index()
            if idx_name != 'city_adcode':
                city_table = city_table.rename(columns={idx_name: 'city_adcode'})
            return city_table

    normalized_columns = {
        str(col).lower().replace('_', '').replace('-', '').replace(' ', ''): col
        for col in city_table.columns
    }
    for key in ['cityadcode', 'adcode', 'code', 'citycode']:
        if key in normalized_columns:
            return city_table.rename(columns={normalized_columns[key]: 'city_adcode'})

    raise KeyError(
        "Cannot find city_adcode in frykit city table. "
        f"Available columns: {list(city_table.columns)}; index name: {city_table.index.name}"
    )


def get_cn_city_table_compatible():
    try:
        city_table = fshp.get_cn_city_table(data_source='tianditu')
    except TypeError:
        city_table = fshp.get_cn_city_table()
    return normalize_city_table(city_table)


def _as_geometry_list(geoms):
    if geoms is None:
        return []
    if isinstance(geoms, (list, tuple)):
        return list(geoms)
    if hasattr(geoms, 'geometry'):
        return list(geoms.geometry)
    if hasattr(geoms, 'geom_type'):
        return [geoms]
    try:
        return list(geoms)
    except TypeError:
        return [geoms]


def get_cn_city_compatible(city_adcode):
    adcodes = [int(code) for code in np.asarray(city_adcode).ravel().tolist()]

    try:
        all_cities = _as_geometry_list(fshp.get_cn_city(data_source='tianditu'))
    except TypeError:
        try:
            all_cities = _as_geometry_list(fshp.get_cn_city())
        except TypeError:
            all_cities = []

    if all_cities:
        city_table = get_cn_city_table_compatible()
        if len(city_table) == len(all_cities) and 'city_adcode' in city_table.columns:
            geom_by_adcode = {}
            for code, geom in zip(city_table['city_adcode'].astype(int), all_cities):
                geom_by_adcode.setdefault(int(code), geom)
            if all(code in geom_by_adcode for code in adcodes):
                return [geom_by_adcode[code] for code in adcodes]

    for codes_arg in [adcodes, tuple(adcodes)]:
        try:
            return _as_geometry_list(fshp.get_cn_city(codes_arg, data_source='tianditu'))
        except TypeError:
            pass
        try:
            return _as_geometry_list(fshp.get_cn_city(codes_arg))
        except TypeError:
            pass

    cities = []
    for code in adcodes:
        try:
            geoms = fshp.get_cn_city(code, data_source='tianditu')
        except TypeError:
            try:
                geoms = fshp.get_cn_city(code)
            except TypeError:
                geoms = fshp.get_cn_city([code])
        cities.extend(_as_geometry_list(geoms))
    return cities


def prepare_city_map_data(plot_df, value_col):
    map_df = plot_df.copy()
    if 'city_adcode' not in map_df.columns:
        raise KeyError("plot_df must contain a 'city_adcode' column before mapping.")
    if value_col not in map_df.columns:
        raise KeyError(f"plot_df must contain '{value_col}' before mapping.")

    map_df = map_df.dropna(subset=['city_adcode']).copy()
    map_df['city_adcode'] = map_df['city_adcode'].astype(int)
    if map_df['city_adcode'].duplicated().any():
        map_df = map_df[~map_df['city_adcode'].duplicated(keep='first')].copy()

    value_by_adcode = pd.to_numeric(map_df[value_col], errors='coerce')
    value_by_adcode.index = map_df['city_adcode'].astype(int)

    # Draw every city in the frykit table. Cities outside the study sample keep
    # NaN values and are rendered with cmap.set_bad(...), i.e. the N/A grey.
    city_table = get_cn_city_table_compatible()
    city_table = city_table.dropna(subset=['city_adcode']).copy()
    city_table['city_adcode'] = city_table['city_adcode'].astype(int)
    city_table = city_table[~city_table['city_adcode'].duplicated(keep='first')].copy()

    city_adcode = city_table['city_adcode'].astype(int).tolist()
    cities = get_cn_city_compatible(city_adcode)
    if len(cities) != len(city_adcode):
        raise ValueError(
            f'frykit returned {len(cities)} city geometries for {len(city_adcode)} adcodes; '
            'cannot safely align map data.'
        )
    data_series = city_table['city_adcode'].map(value_by_adcode)
    if 'pr_adcode' in city_table.columns:
        pr_values = city_table['pr_adcode'].astype(int).map(value_by_adcode)
        data_series = data_series.fillna(pr_values)
    data = data_series.to_numpy(dtype=float)
    return city_adcode, cities, data


def add_cn_line_compatible(ax, **kwargs):
    if hasattr(fplt, 'add_cn_line'):
        return fplt.add_cn_line(ax, **kwargs)
    return None


def add_geometries_compatible(ax, geoms, array=None, cmap=None, norm=None, **kwargs):
    if hasattr(fplt, 'add_geometries'):
        return fplt.add_geometries(ax, geoms, array=array, cmap=cmap, norm=norm, **kwargs)

    edgecolor = kwargs.pop('ec', kwargs.pop('edgecolor', 'black'))
    linewidth = kwargs.pop('lw', kwargs.pop('linewidth', 0.5))
    zorder = kwargs.pop('zorder', None)
    values = np.asarray(array) if array is not None else np.full(len(geoms), np.nan)

    for geom, value in zip(geoms, values):
        if cmap is not None and norm is not None and np.isfinite(value):
            facecolor = cmap(norm(value))
        else:
            facecolor = 'none'
        ax.add_geometries(
            [geom],
            crs=data_crs,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
            **kwargs,
        )
    return None


# Parameter settings
col_plot = "facade_power"

cmap = plt.get_cmap('YlGn')
bad_color = '#E1E1E1'
cmap.set_bad(bad_color)

lw_cn_map = lw * 0.1
lw_axis = lw * 0.5

# Data preparation
city_adcode, cities, data = prepare_city_map_data(df_plot, col_plot)

vmin, vmax = np.floor(np.nanmin(data) * 1) / 1, np.ceil(np.nanmax(data) * 1) / 1
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Set map projection
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = fplt.PLATE_CARREE

# Set tick marks
xticks = np.arange(-180, 181, 10)
yticks = np.arange(-90, 91, 10)

# Prepare main map
font_options = {'size': fs}
plt.rc('font', **font_options)

fig = plt.figure(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)
main_ax = fig.add_subplot(projection=map_crs)

fplt.set_map_ticks(main_ax, (74, 136, 17, 55), xticks, yticks)
# main_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')
main_ax.set_adjustable('datalim')  # Adjust xlim and ylim to fit data
main_ax.set_xticks([])
main_ax.set_yticks([])
main_ax.axis('off')

# Prepare inset map
mini_ax = fplt.add_mini_axes(main_ax, shrink=0.35)
mini_ax.spines[:].set_linewidth(lw_axis)
mini_ax.set_extent((105, 122, 2, 25), data_crs)
# mini_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')

# Add map features
for ax in [main_ax, mini_ax]:
    fplt.add_cn_border(ax, lw=lw_cn_map, fc='none', zorder=-10)
    add_cn_line_compatible(ax, lw=lw_cn_map)

# Draw filled polygons
for ax in [main_ax, mini_ax]:
    add_geometries_compatible(
        ax, cities, array=data,
        cmap=cmap, norm=norm,
        ec='black', lw=lw_cn_map,
    )
    taiwan_aeqd.plot(
        ax=ax, color='none', edgecolor='black',
        linewidth=lw_cn_map,
        )

# Set title
main_ax.set_title(
    'Annual electricity generation of FPV',
    y=0.92,
    fontsize=fs-3,
    weight='normal',
)

# --------------------- Create colorbar ---------------------
# Main colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cax = fig.add_axes([0.08 + 0.06, 0.12, 0.25, 0.03])  # Add space on the left for NaN block
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_ticks(np.linspace(vmin, vmax, 2))
cbar.ax.tick_params(labelsize=fs - 7)
cbar.outline.set_visible(True)
cbar.outline.set_linewidth(lw_axis)

# Align tick labels: left-aligned for min, right-aligned for max
tick_labels = cbar.ax.get_xticklabels()
tick_labels[0].set_horizontalalignment('left')
tick_labels[-1].set_horizontalalignment('right')

# Add NaN color block (gray)
na_cax = fig.add_axes([0.08, 0.12, 0.045, 0.03])  # Position of mini colorbar
na_cax.set_facecolor(bad_color)
na_cax.set_xticks([])
na_cax.set_yticks([])
na_cax.spines[:].set_linewidth(lw_axis)
na_cax.text(0.5, -1.5, 'N/A', ha='center', va='center', fontsize=fs - 7, transform=na_cax.transAxes)

# Add unit label
main_ax.text(
    0.08, 0.12 + 0.03 + 0.01,  # Position: just above the NaN block
    '(TWh)',                   # Annotation text
    transform=fig.transFigure,
    ha='left', va='bottom', fontsize=fs - 6
)

# Adjust layout
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)

fig.savefig('Figs_new_supp/sFig2_generation_comp_1.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig2_generation_comp_1.png", dpi=600, bbox_inches='tight')

plt.show()


#%%  Roof generation potential
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import LAND
from scipy.ndimage import gaussian_filter
from matplotlib.cm import ScalarMappable

import frykit.plot as fplt
import frykit.shp as fshp

# Parameter settings
#col_plot = "facade_roof_ratio_generation"
col_plot = "roof_power"

cmap = plt.get_cmap('YlGnBu')
# cmap = plt.get_cmap('seismic')
# cmap = mcolors.LinearSegmentedColormap.from_list(
#     "truncated", cmap(np.linspace(1/4, 1, 256))  # Use only the latter portion of the colormap
# )
bad_color = '#E1E1E1'
cmap.set_bad(bad_color)

lw_cn_map = lw * 0.1
lw_axis = lw * 0.5

# Data preparation
city_adcode, cities, data = prepare_city_map_data(df_plot, col_plot)

# vmin, vmax = data.min(), data.max()
# norm = plt.Normalize(vmin=0, vmax=3)

vmin, vmax = np.floor(np.nanmin(data) * 1) / 1, np.ceil(np.nanmax(data) * 1) / 1
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Set projection
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = fplt.PLATE_CARREE

# Set tick marks
xticks = np.arange(-180, 181, 10)
yticks = np.arange(-90, 91, 10)

# Prepare main map
font_options = {'size': fs}
plt.rc('font', **font_options)

fig = plt.figure(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)
main_ax = fig.add_subplot(projection=map_crs)

fplt.set_map_ticks(main_ax, (74, 136, 17, 55), xticks, yticks)
# main_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')
main_ax.set_adjustable('datalim')  # Adjust xlim and ylim to fit data
main_ax.set_xticks([])
main_ax.set_yticks([])
main_ax.axis('off')

# Prepare inset map
mini_ax = fplt.add_mini_axes(main_ax, shrink=0.35)
mini_ax.spines[:].set_linewidth(lw_axis)
mini_ax.set_extent((105, 122, 2, 25), data_crs)
# mini_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')

# Add features
for ax in [main_ax, mini_ax]:
    fplt.add_cn_border(ax, lw=lw_cn_map, fc='none', zorder=-10)
    add_cn_line_compatible(ax, lw=lw_cn_map)

# Draw filled polygons
for ax in [main_ax, mini_ax]:
    add_geometries_compatible(
        ax, cities, array=data,
        cmap=cmap, norm=norm,
        ec='black', lw=lw_cn_map,
    )
    taiwan_aeqd.plot(
        ax=ax, color='none', edgecolor='black',
        linewidth=lw_cn_map,
        )

# Set title
main_ax.set_title(
    'Annual electricity generation of RPV',
    y=0.92,
    fontsize=fs-3,
    weight='normal',
)

# --------------------- Create colorbar ---------------------
# Main colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cax = fig.add_axes([0.08 + 0.06, 0.12, 0.25, 0.03])  # Shift to the right to leave space for NaN block
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
#cbar.set_ticks(np.arange(0, 4, 1))  # Set 5 evenly spaced ticks (adjust if needed)
cbar.set_ticks(np.linspace(vmin, vmax, 2))
cbar.ax.tick_params(labelsize=fs - 7)
cbar.outline.set_visible(True)
cbar.outline.set_linewidth(lw_axis)

# Align tick labels: left-aligned for min, right-aligned for max
tick_labels = cbar.ax.get_xticklabels()
tick_labels[0].set_horizontalalignment('left')
tick_labels[-1].set_horizontalalignment('right')

# Add NaN gray block
na_cax = fig.add_axes([0.08, 0.12, 0.045, 0.03])  # Position of the NaN color block
na_cax.set_facecolor(bad_color)
na_cax.set_xticks([])
na_cax.set_yticks([])
na_cax.spines[:].set_visible(False)
na_cax.text(0.5, -1.5, 'N/A', ha='center', va='center', fontsize=fs - 7, transform=na_cax.transAxes)

# Add unit label
main_ax.text(
    0.08, 0.12 + 0.03 + 0.01,  # Position: just above the NaN block
    '(TWh)',                # Annotation text (dash or no unit)
    transform=fig.transFigure,
    ha='left', va='bottom', fontsize=fs - 6
)

# Adjust subplot layout
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)

# Save figure
fig.savefig('Figs_new_supp/sFig2_generation_comp_2.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig2_generation_comp_2.png", dpi=600, bbox_inches='tight')
# Show figure
plt.show()


#%% Facade generation vs Roof generation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Parameter settings
cmap = 'RdPu'
vmin, vmax = df_plot['average_building_height'].min(), df_plot['average_building_height'].max()  # (4.99..., 13.52...)
norm = plt.Normalize(vmin=5, vmax=14)

color_fit = '#d62728'

font_options = {'size': fs-3}
plt.rc('font', **font_options)

fig, ax = plt.subplots(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)
# ax.spines[:].set_linewidth(lw)
# ax.tick_params(width=lw)  # Default value: axes.linewidth and xtick.major.width = 0.8

# Scatter plot with third variable (average building height) controlling color
sc = ax.scatter(df_plot['roof_power'], df_plot['facade_power'],
                c=df_plot['average_building_height'], cmap=cmap, norm=norm,
                alpha=0.8, s=70, edgecolors='black', linewidths=lw)

# Plot fitted line
x = df_plot['roof_power'].values.reshape(-1, 1)
y = df_plot['facade_power'].values

model = LinearRegression(fit_intercept=False)
model.fit(x, y)
k = model.coef_[0]
R2 = model.score(x, y)

x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = k * x_fit
ax.plot(x_fit, y_fit, '-', color=color_fit, linewidth=lw2, 
        label=f'$y={k:.2f}x$')  #  Optionally also show R²
ax.text(
    0.95, 0.35,           # Near the end of the fitted line
    f'$R^2={R2:.2f}$\n(p<0.05)',    # Annotation text
    fontsize=fs - 6,
    color=color_fit,
    transform=ax.transAxes,
    ha='right', va='bottom',
)

# Set axis labels
ax.set_xlim(0, 90)
ax.set_ylim(0, 90)
ax.set_xlabel('RPV', fontsize=fs-4)
ax.set_ylabel('FPV', fontsize=fs-4)

# Add y = x reference line
x_limits = ax.get_xlim()
x_line = np.linspace(x_limits[0], x_limits[1], 100)
ax.plot(x_line, x_line, linestyle='-', color='gray', linewidth=lw, label='$y=x$')

# Set grid and legend
ax.grid(alpha=0.5)
ax.set_title('Electricity generation comparison (TWh)', fontsize=fs - 5)
lg = ax.legend(loc='best', fontsize=fs - 6)
frame = lg.get_frame()
frame.set_linewidth(0.6)
frame.set_edgecolor('black')
frame.set_facecolor('none')

# Add colorbar showing third variable (average building height)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Average building height (m)', fontsize=fs - 3)
cbar.set_ticks(np.arange(5, 15, 3))  # Set 5 evenly spaced ticks (adjust as needed)
cbar.ax.tick_params(labelsize=fs - 3)

# Adjust subplot layout
plt.subplots_adjust(left=0.19, right=0.96, bottom=0.18, top=0.96)

# Save figure
fig.savefig('Figs_new_supp/sFig2_generation_comp_3.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig2_generation_comp_3.png", dpi=600, bbox_inches='tight')
# Show figure
plt.show()


#%% Data Preparation
# Columns represent roof area, maximum facade area, and filtered (cost-effective) facade area
df_plot = area_df.iloc[:, :2].copy()

# Merge population data and solar radiation data
df_plot = df_plot.merge(population_df, left_index=True, right_index=True)
df_plot = df_plot.merge(solar_df, left_index=True, right_index=True)

df_plot = assign_city_level(df_plot)

# Sort by urban population in ascending order within each city level
df_plot.sort_values(['city_level', '\u57ce\u533a\u4eba\u53e3'], ascending=True, inplace=True)

# Add 'city_adcode' column
df_plot = df_plot.merge(city_adcode_df, left_index=True, right_index=True, how='left')
df_plot['city_adcode'] = df_plot['city_adcode'].astype(int)

# --------------------- Load and unify to geographic coordinate system (EPSG:4326) ---------------------
import geopandas as gpd
import pyproj
file_path_taiwan = 'Fig_input_data/\u53f0\u6e7e\u77e2\u91cf\u5730\u56feshp\u6570\u636e/'
taiwan = gpd.read_file(file_path_taiwan + '\u53f0\u6e7e\u7701-\u5e02\u77e2\u91cfshp.shp').to_crs(epsg=4326)

# Convert the GeoDataFrame from WGS84 (EPSG:4326) to Azimuthal Equidistant projection
# Set the central coordinates to 105°E, 35°N
proj_aeqd = (
    "+proj=aeqd "
    "+lat_0=35 "   # Central latitude
    "+lon_0=105 "  # Central longitude
    "+datum=WGS84 "
    "+units=m "
    "+no_defs "
)
# Alternatively, use pyproj.CRS.from_proj4() to create a CRS object
aeqd_crs = pyproj.CRS.from_proj4(proj_aeqd)
# Convert to the Azimuthal Equidistant projection
taiwan_aeqd = taiwan.to_crs(aeqd_crs)
## Subfigure (a) City Tier Classification
## Facade area (City level map of selected cities)
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import LAND
import matplotlib.patches as mpatches

import frykit.plot as fplt
import frykit.shp as fshp

# Parameter settings
col_plot = "city_level"

colors_level = ['#FFFFBF', '#FAFA64', '#A0C29B', '#2892C7']
cmap = ListedColormap(colors_level)
bad_color = '#E1E1E1'
cmap.set_bad(bad_color)
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

#labels_level = ['LC-II', 'LC-I', 'VLC', 'SLC']
legend_items = [
    (3, 'Super large-sized city'),
    (2, 'Very large-sized city'),
    (1, 'Large-sized city I'),
    (0, 'Large-sized city II'),
]

lw_cn_map = lw * 0.1
lw_axis = lw * 0.5

# Data preparation
city_adcode, cities, data = prepare_city_map_data(df_plot, col_plot)

# Set projection
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = fplt.PLATE_CARREE

# Set tick marks
xticks = np.arange(-180, 181, 10)
yticks = np.arange(-90, 91, 10)

# Prepare main map
font_options = {'size': fs}
plt.rc('font', **font_options)

fig = plt.figure(figsize=np.array([figwidth * 2 / 3, figwidth * 2 / 3 * 0.7]) / 2.54)
main_ax = fig.add_subplot(projection=map_crs)

fplt.set_map_ticks(main_ax, (74, 136, 17, 55), xticks, yticks)
# main_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')
main_ax.set_adjustable('datalim')  # Allow Axes to adjust xlim and ylim based on data
main_ax.set_xticks([])
main_ax.set_yticks([])
main_ax.axis('off')

# Prepare inset map
mini_ax = fplt.add_mini_axes(main_ax, shrink=0.35)
mini_ax.spines[:].set_linewidth(lw_axis)
mini_ax.set_extent((105, 122, 2, 25), data_crs)
# mini_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')

# Add map features
for ax in [main_ax, mini_ax]:
    fplt.add_cn_border(ax, lw=lw_cn_map, fc='none', zorder=-10)
    add_cn_line_compatible(ax, lw=lw_cn_map)

# Plot filled polygons
for ax in [main_ax, mini_ax]:
    add_geometries_compatible(
        ax, cities, array=data,
        cmap=cmap, norm=norm,
        ec='black', lw=lw_cn_map,
    )
    taiwan_aeqd.plot(
        ax=ax, color='none', edgecolor='black',
        linewidth=lw_cn_map,
    )

# Set title
main_ax.set_title(
    'Selected cities for study',
    y=0.92,
    fontsize=fs - 3,
    weight='normal',
)

# Add legend
patches = []
for level, label in legend_items:
    patch = mpatches.Patch(fc=colors_level[level], ec='k',
            lw=lw_axis, label=label)
    patches.append(patch)

main_ax.legend(
    handles=patches,
    loc=(-0.02, -0.02),
    frameon=False,
    handleheight=0.7,      # Default is 0.7
    handlelength=1.5,      # Default is 2
    fontsize=fs - 12,
    # title='data (units)',
    labelspacing=0.3,      # Vertical spacing between legend entries
    handletextpad=0.5,     # Space between handle and text
)

# # Add subplot label
# main_ax.text(0, 1, f'a', transform=main_ax.transAxes,
#              ha='left', va='top', fontsize=fs + 9, fontweight='bold')

# Adjust layout
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)

# Save figure
fig.savefig('Figs_new_supp/sFig2_select_city_select.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig2_select_city_select.png", dpi=600, bbox_inches='tight')
# Show figure
plt.show()



#%% Subfigure (b) Solar Energy Resources
## Facade area
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import LAND
from scipy.ndimage import gaussian_filter
from matplotlib.cm import ScalarMappable

import frykit.plot as fplt
import frykit.shp as fshp

# Parameter settings
col_plot = "Global"

cmap = plt.get_cmap('YlOrRd')
bad_color = '#E1E1E1'
cmap.set_bad(bad_color)

lw_cn_map = lw * 0.1
lw_axis = lw * 0.5

# Data preparation
city_adcode, cities, data = prepare_city_map_data(df_plot, col_plot)

vmin, vmax = np.floor(np.nanmin(data) * 1) / 1, np.ceil(np.nanmax(data) * 1) / 1
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Set projection
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = fplt.PLATE_CARREE

# Set ticks
xticks = np.arange(-180, 181, 10)
yticks = np.arange(-90, 91, 10)

# Prepare main map
font_options = {'size': fs}
plt.rc('font', **font_options)

fig = plt.figure(figsize=np.array([figwidth * 2 / 3, figwidth * 2 / 3 * 0.7]) / 2.54)
main_ax = fig.add_subplot(projection=map_crs)

fplt.set_map_ticks(main_ax, (74, 136, 17, 55), xticks, yticks)
# main_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')
main_ax.set_adjustable('datalim')  # Allow Axes to adjust xlim and ylim based on data
main_ax.set_xticks([])
main_ax.set_yticks([])
main_ax.axis('off')

# Prepare inset map
mini_ax = fplt.add_mini_axes(main_ax, shrink=0.35)
mini_ax.spines[:].set_linewidth(lw_axis)
mini_ax.set_extent((105, 122, 2, 25), data_crs)
# mini_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')

# Add map features
for ax in [main_ax, mini_ax]:
    fplt.add_cn_border(ax, lw=lw_cn_map, fc='none', zorder=-10)
    add_cn_line_compatible(ax, lw=lw_cn_map)

# Draw filled polygons
for ax in [main_ax, mini_ax]:
    add_geometries_compatible(
        ax, cities, array=data,
        cmap=cmap, norm=norm,
        ec='black', lw=lw_cn_map,
    )
    taiwan_aeqd.plot(
        ax=ax, color='none', edgecolor='black',
        linewidth=lw_cn_map,
    )

# Set title
main_ax.set_title(
    'Annual gross radiation',
    y=0.92,
    fontsize=fs-3,
    weight='normal',
)

# --------------------- Create colorbar ---------------------
# Main colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cax = fig.add_axes([0.08 + 0.06, 0.12, 0.25, 0.03])  # Move right to leave space for NaN color block
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_ticks(np.linspace(vmin, vmax, 2))
cbar.ax.tick_params(labelsize=fs - 7,  # length=0
        )  # Set font size as fs (the size you defined)
cbar.outline.set_visible(True)  # Remove outline
cbar.outline.set_linewidth(lw_axis)

# Align tick label: left-align for min value, right-align for max value
tick_labels = cbar.ax.get_xticklabels()
tick_labels[0].set_horizontalalignment('left')
tick_labels[-1].set_horizontalalignment('right')

# Add NaN gray block
na_cax = fig.add_axes([0.08, 0.12, 0.045, 0.03])  # Color block position (small colorbar)
na_cax.set_facecolor(bad_color)  # Set to gray
na_cax.set_xticks([])  # Remove ticks
na_cax.set_yticks([])
na_cax.spines[:].set_linewidth(lw_axis)  # Remove borders
na_cax.text(0.5, -1.5, 'N/A', ha='center', va='center', fontsize=fs - 7, transform=na_cax.transAxes)

# Add annotation
main_ax.text(
    0.08, 0.12 + 0.03 + 0.01,  # Position: a little above the NaN color block center
    '(kWh/m$^2$)',                  # Annotation text
    transform=fig.transFigure,            # Use figure coordinates for positioning
    ha='left', va='bottom', fontsize=fs - 6
)

# # Add subplot label
# main_ax.text(0, 1, f'a', transform=main_ax.transAxes,
#              ha='left', va='top', fontsize=fs + 9, fontweight='bold')

# Adjust subplot layout
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96,
		# wspace=0.25, hspace=0.21
        )

# Save the figure
fig.savefig('Figs_new_supp/sFig2_select_city_solar.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig2_select_city_solar.png", dpi=600, bbox_inches='tight')

# Show the figure
plt.show()



#%% Supplementary of Fig. 2. Detailed Data of Different Cities
## Data Preparation
# import numpy as np
# import pandas as pd

# # Merge area data and annual generation data
# df_plot = area_df.iloc[:, :2].copy()  # km^2

# df_plot['Area_roof-0(km2)'] = main_case['rpv_installable_area_km2'].values
# df_plot['Area_facade-0(km2)'] = main_case['fpv_installable_area_km2'].values
# roof_power = main_case['rpv_generation_TWh'].values  # TWh
# facade_power = main_case['fpv_generation_TWh'].values  # TWh
# df_plot['roof_power'] = roof_power
# df_plot['facade_power'] = facade_power

# # Merge average building height data
# df_plot = df_plot.merge(average_building_height, left_index=True, right_index=True)

# # Calculate the ratio of facade to roof area and generation
# df_plot['facade_roof_ratio_area'] = df_plot['Area_facade-0(km2)'] / df_plot['Area_roof-0(km2)']
# df_plot['facade_roof_ratio_generation'] = df_plot['facade_power'] / df_plot['roof_power']

# # Merge population data
# df_plot = df_plot.merge(population_df, left_index=True, right_index=True)

# ## Subfigure (a) Area
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# colors = ['#87CEEB', '#2E8B57']  # Blue and green, corresponding to roof and facade
# color_ratio = '#003366'
# lw_axis = 0.8

# # ----------------- Data Preparation -----------------
# # df_plot index is city names, columns include:
# #   city_level, Area_facade-0(km2), Area_roof-0(km2)
# df_vis = df_plot.copy()
# df_vis = df_vis.rename(index={
#     'Haerbin': 'Harbin',
#     'Huhehaote': 'Hohhot',
#     'Wulumuqi': 'Urumqi',
#     'Xian': "Xi'an"
# })

# # 1) Sort by city_level (descending) and Area_facade-0(km2) (descending)
# df_vis.sort_values(
#     by=['city_level', 'Area_facade-0(km2)'],
#     ascending=[False, False],
#     inplace=True
# )

# # 2) Split the cities into two parts
# n = len(df_vis)
# mid = n // 2
# df_vis_part1 = df_vis.iloc[:mid]
# df_vis_part2 = df_vis.iloc[mid:]

# # Define a mapping dictionary for city levels (example, can be modified/expanded)
# level_labels = {
#     3: 'SLC',
#     2: 'VLC',
#     1: 'LC-I',
#     0: 'LC-II'
# }

# # ----------------- Create canvas and subplots -----------------
# font_options = {'size': fs-12}
# plt.rc('font', **font_options)

# fig, axes = plt.subplots(
#     1, 2,
#     figsize=np.array([figwidth, figwidth * 1.2]) / 2.54,
# )

# parts = [df_vis_part1, df_vis_part2]

# for i_ax, df_part in enumerate(parts):
#     ax = axes[i_ax]

#     # Construct a temporary table temp_part to retain city_level for later labeling
#     temp_part = df_part[['city_level', 'Area_facade-0(km2)', 'Area_roof-0(km2)', 'facade_roof_ratio_area']].copy()
#     # Only two columns for actual plotting
#     df_plot_part = temp_part[['Area_facade-0(km2)', 'Area_roof-0(km2)']]

#     # Plot horizontal bar chart
#     df_plot_part.plot(
#         kind='barh',
#         ax=ax,
#         stacked=False,
#         width=0.8,
#         edgecolor='none',
#         linewidth=lw_axis,
#         alpha=0.8,
#         color=colors[::-1],  # Facade, Roof
#         legend=False,
#     )

#     # Reverse the y-axis order
#     ax.invert_yaxis()

#     ax.set_xlim(0, 3000)
#     ax.yaxis.set_tick_params(length=0)
#     ax.set_xlabel("Area (km$^2$)", fontsize=fs-9)
#     ax.set_ylabel(None)

#     # -------------- Labeling city levels --------------
#     # Reset index to match row_idx with barh y-axis order
#     temp_part.reset_index(inplace=True)       # Turn city names into a column
#     temp_part['row_idx'] = temp_part.index    # Row index for y-axis in barh

#     # Group by city_level, find the min and max row indices for each group on the y-axis
#     grouped = temp_part.groupby('city_level')['row_idx']
#     for lvl, rows in grouped:
#         min_idx = rows.min()
#         max_idx = rows.max()
#         mid_idx = 0.5 * (min_idx + max_idx)  # Vertical middle

#         # x_pos: offset from the far-right of the largest bar, used for drawing arrows and text
#         x_pos = ax.get_xlim()[-1] * 0.83
#         # Arrow length
#         arrow_len = 0.5
#         h_len = ax.get_xlim()[-1] * 0.04  # Half-length of horizontal line (example value)

#         # (1) Top arrow
#         ax.annotate(
#             '',
#             xy=(x_pos, min_idx),
#             xytext=(x_pos, min_idx - arrow_len),
#             arrowprops=dict(arrowstyle='<-', lw=lw_axis, color='black', mutation_scale=20),  # Larger value makes the arrow bigger
#             annotation_clip=False,
#             transform=ax.transData,
#         )
#         ax.plot(
#             [x_pos - h_len, x_pos + h_len],  # Left and right endpoints
#             [min_idx-0.3, min_idx-0.3],              # Same y-value
#             color='black',
#             lw=lw_axis,
#             transform=ax.transData,
#             clip_on=False
#         )

#         # (2) Vertical line
#         ax.plot(
#             [x_pos, x_pos], 
#             [min_idx, max_idx],
#             color='black', lw=lw_axis,
#             transform=ax.transData,
#             clip_on=False
#         )

#         # (3) Bottom arrow
#         ax.annotate(
#             '',
#             xy=(x_pos, max_idx),
#             xytext=(x_pos, max_idx + arrow_len),
#             arrowprops=dict(arrowstyle='<-', lw=lw_axis, color='black', mutation_scale=20),
#             annotation_clip=False,
#             transform=ax.transData
#         )
#         ax.plot(
#             [x_pos - h_len, x_pos + h_len],
#             [max_idx+0.3, max_idx+0.3],
#             color='black',
#             lw=lw_axis,
#             transform=ax.transData,
#             clip_on=False
#         )

#         # (4) Add text at the midpoint of the vertical line
#         text_label = level_labels.get(lvl, f'Level {lvl}')
#         ax.text(
#             x_pos, mid_idx,
#             text_label,
#             ha='center', va='center',
#             fontsize=fs-9,
#             transform=ax.transData,
#             bbox=dict(
#                 facecolor='white',         # Background color
#                 edgecolor='none'           # Remove border
#             )
#         )

#     # -------------- Plot ratio scatter on the same subplot with ax.twiny() --------------
#     # Create a new x-axis with shared y-axis
#     ax2 = ax.twiny()
    
#     # y-values correspond to barh, need to use row_idx
#     y_vals = temp_part['row_idx']
#     # x-values are the ratios
#     x_vals = temp_part['facade_roof_ratio_area']
    
#     # Plot scatter plot
#     ax2.scatter(x_vals, y_vals, color=color_ratio, s=50,
#             alpha=0.8, marker='o')
    
#     # Set x-axis labels, limits, ticks, etc.
#     ax2.set_xlim(0, 3)
#     ax2.set_xlabel("Ratio of facade to roof (\u2013)", fontsize=fs-9)
    
#     # Make sure y-axis range matches ax
#     ax2.set_ylim(ax.get_ylim())

# # ----------------- Add unified legend -----------------
# lg = fig.legend(
#     handles=[
#         mpatches.Patch(fc=colors[1], label='Facade area'),  # Green
#         mpatches.Patch(fc=colors[0], label='Roof area'),    # Blue
#         plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color_ratio,
#                    markersize=7, label='Area ratio')  # Dark blue circle
#     ],
#     handleheight=0.7,  # Default value 0.7
#     handlelength=2,  # Default value 2
#     loc='lower center',
#     ncol=3,
#     fontsize=fs - 9,
#     bbox_to_anchor=(0.5, 0.00)  # Legend placed below the entire figure
# )
# frame = lg.get_frame()
# frame.set_linewidth(0.6)
# frame.set_edgecolor('black')
# frame.set_facecolor('none')

# # ----------------- Adjust layout and save -----------------
# # Adjust subplot layout
# plt.subplots_adjust(left=0.10, right=0.98, bottom=0.1, top=0.95,
# 		wspace=0.45, hspace=0
#         )

# fig.savefig('Figs_new_supp/sFig2_area_comp_1.pdf', format='pdf', dpi=600, bbox_inches='tight')
# fig.savefig("Figs_new_supp/sFig2_area_comp_1.png", dpi=600, bbox_inches='tight')

# plt.show()

# #%% Subfigure (c) Area Boxplot
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# #colors = ['#87CEEB', '#2E8B57']  # Blue and green, corresponding to roof and facade
# colors = ['skyblue', 'mediumseagreen']
# color_ratio = "#00336699"
# lw_axis = 0.8
# lw_box = 1.0

# # ---------------- Custom city level labels ----------------
# level_labels = {
#     3: 'SLC',
#     2: 'VLC',
#     1: 'LC-I',
#     0: 'LC-II'
# }

# # Map city level to category names and sort in descending order
# df_plot['city_level_label'] = df_plot['city_level'].map(level_labels)
# ordered_labels = [level_labels[i] for i in sorted(level_labels.keys(), reverse=True)]

# # ----------------- Organize Data -----------------
# # df_plot contains:
# #   - city_level_label (grouping basis)
# #   - 'Area_facade-0(km2)' (Facade)
# #   - 'Area_roof-0(km2)'   (Roof)
# # Extract Facade/Roof for each city_level_label

# # Loop through ordered_labels for plotting
# facade_data = []
# roof_data = []
# for label in ordered_labels:
#     df_sub = df_plot[df_plot['city_level_label'] == label]
#     # Facade and Roof values
#     facade_vals = df_sub['Area_facade-0(km2)'].dropna().values
#     roof_vals = df_sub['Area_roof-0(km2)'].dropna().values
#     facade_data.append(facade_vals)
#     roof_data.append(roof_vals)

# # ----------------- Create Plot -----------------
# font_options = {'size': fs - 12}
# plt.rc('font', **font_options)

# fig, ax = plt.subplots(figsize=np.array([figwidth * 0.5, figwidth * 0.5 * 0.9]) / 2.54,)  # Figure size in inches

# # Center positions for each city level on the x-axis
# x_positions = np.arange(len(ordered_labels))

# # Offsets for Facade and Roof within the same group
# offset_facade = -0.2
# offset_roof   = +0.2

# # Common parameters for the boxplot
# common_props = dict(
#     showfliers=False,  # Do not show outliers
#     patch_artist=True, # Allow custom filling color for the box
#     showmeans=True,    # Show mean point
#     widths=0.3,        # Box width
#     medianprops=dict(color='orange', linewidth=lw_box),  # Median line
#     # boxprops=dict(edgecolor='black', linewidth=lw_box),  # Box edge
#     # whiskerprops=dict(color='black', linewidth=lw_box),  # Whisker line
#     # capprops=dict(color='black', linewidth=lw_box),  # Cap line
#     meanprops=dict(marker='x', markerfacecolor='black',
#         markeredgecolor='black', markersize=7, linewidth=lw_axis),
# )

# # Colors
# color_facade = colors[-1]
# color_roof   = colors[0]

# # --------------- Draw Facade and Roof Boxplots ---------------
# # 1) Facade
# for i, data_vals in enumerate(facade_data):
#     # Draw boxplot at x_positions[i] + offset_facade
#     bp = ax.boxplot(
#         [data_vals],            # Boxplot requires an iterable, here only one group of data
#         positions=[x_positions[i] + offset_facade],
#         **common_props
#     )
#     # Fill the box with color
#     for box in bp['boxes']:
#         box.set_facecolor(color_facade)

# # 2) Roof
# for i, data_vals in enumerate(roof_data):
#     bp = ax.boxplot(
#         [data_vals],
#         positions=[x_positions[i] + offset_roof],
#         **common_props
#     )
#     for box in bp['boxes']:
#         box.set_facecolor(color_roof)

# # ----------------- Draw Vertical Guide Lines -----------------
# # Draw vertical lines between each group to help differentiate city types
# # Since x-axis ticks are at 0, 1, 2,..., we can draw guide lines at x = i - 0.5 (i=1,2,...)
# for i in range(1, len(ordered_labels)):
#     ax.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=lw_axis)

# # ----------------- Axis and Legend Settings -----------------
# # Place x-axis ticks at the center of each group
# ax.set_xticks(np.arange(len(ordered_labels)))
# ax.set_xticklabels(ordered_labels,fontsize = fs-9)

# # Set y-axis limits and ticks (based on your needs)
# ax.set_xlim(-0.5, len(ordered_labels)-0.5)
# ax.set_ylim(0, 1600)
# ax.set_yticks(np.arange(0, 2000, 400))
# ax.set_ylabel("Area (km$^2$)", fontsize=fs - 6)
# #ax.set_xlabel("City types", fontsize=fs - 6)

# # Set axis appearance
# ax.tick_params(axis='both', which='major', length=5, width=lw_axis, color='black')
# for spine in ax.spines.values():
#     spine.set_linewidth(lw_axis)
#     spine.set_color('black')

# # Manually create legend
# patch_facade = mpatches.Patch(facecolor=color_facade, edgecolor='black', label='Facade area')
# patch_roof   = mpatches.Patch(facecolor=color_roof, edgecolor='black', label='Roof area')
# lg = ax.legend(handles=[patch_facade, patch_roof],
#                fontsize=fs - 9, loc='upper right')
# lg.get_frame().set_linewidth(0.6)
# lg.get_frame().set_edgecolor('black')
# lg.get_frame().set_facecolor('none')

# # ----------------- Adjust Layout and Save -----------------
# # Adjust subplot layout
# plt.subplots_adjust(left=0.17, right=0.98, bottom=0.12, top=0.96,
# 		wspace=0.45, hspace=0
#         )

# fig.savefig('Figs_new_supp/sFig2_area_comp_2.pdf', format='pdf', dpi=600, bbox_inches='tight')
# fig.savefig("Figs_new_supp/sFig2_area_comp_2.png", dpi=600, bbox_inches='tight')

# plt.show()


# lw_axis = 0.8
# lw_box = 1.0
# df_plot['city_level_label'] = df_plot['city_level'].map(level_labels)
# ordered_labels = [level_labels[i] for i in sorted(level_labels.keys(), reverse=True)]

# # ----------------- Organize Data -----------------
# # Extract facade_roof_ratio_area values for each city_level_label
# ratio_data = []
# for label in ordered_labels:
#     df_sub = df_plot[df_plot['city_level_label'] == label]
#     ratio_vals = df_sub['facade_roof_ratio_area'].dropna().values
#     ratio_data.append(ratio_vals)

# # ----------------- Create Plot -----------------
# font_options = {'size': fs - 12}
# plt.rc('font', **font_options)

# fig, ax = plt.subplots(figsize=np.array([figwidth * 0.5, figwidth * 0.5 * 0.9]) / 2.54)  # Figure size in inches

# # Center positions for each city level on the x-axis
# x_positions = np.arange(len(ordered_labels))

# # ----------------- Draw Boxplot -----------------
# common_props = dict(
#     showfliers=False,             # Do not show outliers
#     patch_artist=True,            # Allow box filling
#     showmeans=True,
#     widths=0.3,                   # Box width
#     medianprops=dict(color='orange', linewidth=lw_box),  # Median line style
#     meanprops=dict(marker='x', markerfacecolor='black',
#             markeredgecolor='black', markersize=7, linewidth=lw_axis),
# )

# bp = ax.boxplot(
#     ratio_data,
#     positions=x_positions,
#     **common_props
# )
# # Fill the boxes with color
# for box in bp['boxes']:
#     box.set_facecolor(color_ratio)
#     # box.set_alpha(0.5)

# # ----------------- Axis and Legend Settings -----------------
# ax.set_xticks(x_positions)
# ax.set_xticklabels(ordered_labels, fontsize=fs - 9)
# # Adjust y-axis limits based on actual data; here is an example
# ax.set_xlim(-0.5, len(ordered_labels)-0.5)
# ax.set_ylim(1, 3)
# ax.set_yticks(np.arange(1, 3.1, 0.5))
# ax.set_ylabel("Ratio of facade to roof (\u2013)", fontsize=fs - 6)
# #ax.set_xlabel("City types", fontsize=fs - 9)

# ax.tick_params(axis='both', which='major', length=5, width=lw_axis, color='black')
# for spine in ax.spines.values():
#     spine.set_linewidth(lw_axis)
#     spine.set_color('black')

# # Manually create legend
# patch_ratio = mpatches.Patch(facecolor=color_ratio, edgecolor='black', label='Area ratio')
# lg = ax.legend(handles=[patch_ratio], fontsize=fs - 9, loc='upper right')
# lg.get_frame().set_linewidth(0.6)
# lg.get_frame().set_edgecolor('black')
# lg.get_frame().set_facecolor('none')

# # ----------------- Adjust Layout and Save -----------------
# plt.subplots_adjust(left=0.17, right=0.98, bottom=0.12, top=0.96,
# 		# wspace=0.45, hspace=0
#         )

# fig.savefig('Figs_new_supp/sFig2_area_comp_3.pdf', format='pdf', dpi=600, bbox_inches='tight')
# fig.savefig("Figs_new_supp/sFig2_area_comp_3.png", dpi=600, bbox_inches='tight')

# plt.show()








#%%  #Begin to draw time-seris data
# python 3.11
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

## ------------------------ Plotting Parameters Setup ------------------------
ratio_plot = 3  # Scaling factor for plot dimensions. Larger plots ensure clarity even when converted to bitmap, hence enlarged.

figwidth =  8.5 * ratio_plot  # In centimeters, but the default unit is inches. 1 inch = 2.54 cm.
fs = 8 * ratio_plot  # Minimum font size for axis labels (6~14 pt).
lw = 0.4 * ratio_plot  # Line width for axes (0.25~1.5 pt).
lw2 = 0.75 * ratio_plot  # Line width for curves.
lw3 = 1 * ratio_plot  # Line width for bold curves.

grid_alpha = 0.5
# bar_width = 0.3  # Bar width can be adjusted if needed.

import os
import numpy as np

# Specify the HDF file path for storing results
result_folder = r'Fig_input_data/' 
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Cap_facade_ideal_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save it
Cap_facade_ideal_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the existing HDF file")

Cap_facade_ideal_all_df
import os
import numpy as np
import pandas as pd

# Specify the HDF file path for storing results

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Cap_roof_ideal_all_df.h5')

# If the HDF file already exists, read the data
Cap_roof_ideal_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

# Specify the HDF file path for storing results
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_facade_ideal_1_sum_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Power_facade_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_roof_ideal_1_sum_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Power_roof_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_type_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Grid_type_all_df = pd.read_hdf(hdf_file, key='df')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_feas_static_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Grid_feas_static_all_df = pd.read_hdf(hdf_file, key='df')

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_feas_wea_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Grid_feas_wea_all_df = pd.read_hdf(hdf_file, key='df')

# skew: skewness index - the skewness of the mean compared to the median;
# WWR: Window-to-Wall Ratio, consistent within the grid
# can_avg: other - mean height of vegetation canopy above ground;
# can_std: other - standard deviation of vegetation canopy height above ground
Grid_feature_label = pd.concat([Grid_feas_static_all_df.iloc[:, :-4],  # Excluding the last 4 columns: ['skew','WWR','can_avg','can_std']
        Grid_feas_wea_all_df['Global'],
        Cap_facade_ideal_all_df, Power_facade_ideal_1_sum_all_df],
        axis=1)

Grid_feature_label['util_hour'] = Grid_feature_label['Power_facade_ideal_1_sum'] / Grid_feature_label['Cap_facade_ideal'] / 1e3  # MWh/GW, multiplied by 1e3 for conversion

Grid_feature_label['generation_per_area'] = \
    Grid_feature_label['Power_facade_ideal_1_sum'] / Grid_type_all_df['facade_area'] * 1e3  # MWh/m2, multiplied by 1e3 to convert to kWh/m2

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = Grid_feature_label.copy()

all_cols = df.columns.tolist()
label_col = all_cols[-3]    # The last column is the label: total annual generation. The second last column is installed capacity
feature_cols = all_cols[:-4]  # All columns except the last 2 are features

X = df[feature_cols]
y = df[label_col] / 1000  # Convert to GWh

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=0.2, 
            random_state=42)

# ------------------------- Train the XGBoost model -------------------------
model = xgb.XGBRegressor(
    n_estimators=100,  # 'n_estimators': [50, 100, 200],  # Default value: 100 (number of weak learners)
    max_depth=6,  # 'max_depth': [3, 6, 9],  # Default value: 6 (maximum depth of the tree)
    learning_rate=0.2,  # 'learning_rate': [0.01, 0.1, 0.2],  # Default value: 0.3 (learning rate)
    random_state=42)
model.fit(X_train, y_train)

# ------------------------- Evaluate prediction performance -------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Take the square root of mean squared error
y_mean = np.mean(y_test)  # Calculate the mean of y_test
cv_rmse = (rmse / y_mean)

print(f"Test R^2: {r2:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test CV-RMSE: {cv_rmse*100:.2f}%")  # Display as percentage
# import shap
# import numpy as np

# # ----------------- Define SHAP result storage path -----------------

# shap_hdf_path = "annual_generation_per_grid_shap.hdf"  # HDF5 storage path

# # ----------------- Check if SHAP results already exist, if so, load them -----------------
# if os.path.exists(shap_hdf_path):
#     print("Detected SHAP result file, loading...")
#     shap_df_grid = pd.read_hdf(shap_hdf_path, key='shap_values')
#     print("SHAP results loaded.")
# else:
#     print("No SHAP result file detected, calculating SHAP values...")
    
#     # Build Explainer and calculate SHAP values
#     explainer = shap.TreeExplainer(model, X_train)  # Inform SHAP about the training data
#     shap_values_array = explainer.shap_values(X_test)  # Calculate SHAP values
    
#     # **Convert SHAP results into DataFrame**
#     shap_df_grid = pd.DataFrame(shap_values_array, columns=X_test.columns)  # Columns corresponding to features
    
#     # Store SHAP results to HDF5
#     os.makedirs(os.path.dirname(shap_hdf_path), exist_ok=True)
#     shap_df_grid.to_hdf(shap_hdf_path, key='shap_values', mode='w')
    
#     print(f"SHAP results have been calculated and stored in {shap_hdf_path}")

# shap_values = shap_df_grid.values

# X_test.columns
# # Step 1: Compute SHAP importance and convert to Series (including feature names)
# shap_importance = np.abs(shap_values).mean(axis=0)  # Calculate the mean absolute SHAP values for each feature
# feature_names = X_test.columns  # Get the feature names from the test set

# # Convert the SHAP importance values into a pandas Series with feature names as index
# shap_series = pd.Series(shap_importance, index=feature_names)

# # Step 2: Replace feature names with full names
# name_map = {
#     'lon': 'Longitude',
#     'lat': 'Latitude',
#     'density': 'Building density',
#     'hei_avg': 'Mean building height',
#     'hei_std': 'Std. of building height',
#     'area_std': 'Std. of building footprint area',
#     'complex': 'Complexity',
#     'compact': 'Compactness',
#     'volume': 'Number of building volumes',
#     'outdoor': 'Mean outdoor distance',
#     '12ratio': '12-ratio',
#     'Global': 'Annual gross radiation'
# }

# # Map the feature names to their full descriptions
# shap_series.index = shap_series.index.map(name_map)

# # Step 3: Drop the '12-ratio' feature, as it's not needed
# shap_series.drop('12-ratio', inplace=True)

# # Step 4: Calculate the relative importance (normalize by the sum of all SHAP values)
# shap_series /= shap_series.sum()

# shap_series

#%% Fig. 3. Temporal Characteristics of FPV Power Generation
## Data Preparation and Normalization
import numpy as np
from scipy.interpolate import interp1d

# ------------------- Read Area Data -------------------
# Roof area, maximum facade area, and filtered facade area
area_df = pd.read_excel('Fig_input_data/City_statistic.xlsx', sheet_name='Key_information',
        usecols=[0, 3, 4, 5], index_col=[0])

# The data includes different stage data. "Roof" is roof capacity, "Ideal" is the ideal facade capacity, and "Facade" is the economically viable facade capacity. Unit: GW.
# The numbers represent different stages, as both solar panel price and efficiency change over time, so the Ideal value is also split by stages.
cap_df = pd.read_excel('Fig_input_data/City_Cap.xlsx', index_col=[0])

# ------------------- Read Whether the City is in the North -------------------
# 1 represents northern cities, 0 represents southern cities
city_north = pd.read_excel('Fig_input_data/City_north_south.xlsx', index_col=0)

# ------------------- Read Power Generation Time Series Data to Improve Resolution -------------------
file_path = 'Fig_input_data/'
ideal_facade_data_1_path = file_path + 'Power_facade_ideal_1.npy'
roof_data_1_path = file_path + 'Power_roof_1.npy'

# Hourly power generation curve for each city
ideal_facade_curve = np.load(ideal_facade_data_1_path)
roof_curve = np.load(roof_data_1_path)  # Rows represent cities, columns represent hour_of_year

# Organizing the data into DataFrame
ideal_facade_curve_df = pd.DataFrame(ideal_facade_curve, index=area_df.index)
roof_curve_df = pd.DataFrame(roof_curve, index=area_df.index)  # Rows represent cities, columns represent hour_of_year

# Standardization by dividing by the mean
row_means = np.mean(ideal_facade_curve, axis=1, keepdims=True)
ideal_facade_curve_norm_annual = ideal_facade_curve / row_means
row_means = np.mean(roof_curve, axis=1, keepdims=True)
roof_curve_norm_annual = roof_curve / row_means

# ---------- Standardization to daily data ----------
# Assuming 365 days in a year, and each day has xx data points
n_cities, n_points = ideal_facade_curve.shape  # n_points should be 35040 = 365*96
n_points_day = int(n_points / 365)

# Daily standardization of the ideal_facade_curve
ideal_facade_reshaped = ideal_facade_curve.reshape(n_cities, 365, n_points_day)
daily_means_facade = np.mean(ideal_facade_reshaped, axis=2, keepdims=True)
ideal_facade_curve_norm = (ideal_facade_reshaped / daily_means_facade).reshape(n_cities, n_points)

# Daily standardization of the roof_curve
roof_reshaped = roof_curve.reshape(n_cities, 365, n_points_day)
daily_means_roof = np.mean(roof_reshaped, axis=2, keepdims=True)
roof_curve_norm = (roof_reshaped / daily_means_roof).reshape(n_cities, n_points)
# ------------------- Organize daily power generation curves for summer and winter -------------------  
summer_curve_data_norm = np.zeros((102, 2, 92, n_points_day))  # Store daily curves for different seasons. The 4 dimensions are: cities, seasons, days, number of time slots (with resolution increased to 15 minutes)
winter_curve_data_norm = np.zeros((102, 2, 90, n_points_day)) 

for i in range(92):
    for j in range(n_points_day):
        for k in range(102):
            summer_curve_data_norm[k, 0, i, j] = ideal_facade_curve_norm[k, 151 * n_points_day + i * n_points_day + j]
            summer_curve_data_norm[k, 1, i, j] = roof_curve_norm[k, 151 * n_points_day + i * n_points_day + j]

for i in range(90):
    for j in range(n_points_day):
        for k in range(102):
            if 334 * n_points_day + i * n_points_day + j < 365 * n_points_day: 
                winter_curve_data_norm[k, 0, i, j] = ideal_facade_curve_norm[k, 334 * n_points_day + i * n_points_day + j]
                winter_curve_data_norm[k, 1, i, j] = roof_curve_norm[k, 334 * n_points_day + i * n_points_day + j]
            else:
                winter_curve_data_norm[k, 0, i, j] = ideal_facade_curve_norm[k, 334 * n_points_day + i * n_points_day + j - 365 * n_points_day]
                winter_curve_data_norm[k, 1, i, j] = roof_curve_norm[k, 334 * n_points_day + i * n_points_day + j - 365 * n_points_day]

# ------------------- Organize average daily power generation curves for summer and winter -------------------  
avg_summer_facade_norm = np.zeros((102, n_points_day))  # Store typical daily curves for different seasons. The 2 dimensions are: cities, hours
avg_summer_roof_norm = np.zeros((102, n_points_day))
avg_winter_facade_norm = np.zeros((102, n_points_day))
avg_winter_roof_norm = np.zeros((102, n_points_day))

for i in range(102):
    for j in range(n_points_day):
        avg_summer_facade_norm[i, j] = np.mean(summer_curve_data_norm[i, 0, :, j])
        avg_summer_roof_norm[i, j] = np.mean(summer_curve_data_norm[i, 1, :, j])
        avg_winter_facade_norm[i, j] = np.mean(winter_curve_data_norm[i, 0, :, j])
        avg_winter_roof_norm[i, j] = np.mean(winter_curve_data_norm[i, 1, :, j])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

colors = ['skyblue', 'mediumseagreen']  # Blue, Green, corresponding to Roof and Facade
color_fill = '#87CEEB'
color_season = 'black'
alpha_plot = 1
alpha_fill = 0.4
lw_plot = 1.5

# Select cities
cities = {'Changchun'}

# Get the row index for the cities in area_df
city_indices = [area_df.index.get_loc(city) for city in cities if city in area_df.index]
cities_actual = area_df.index[city_indices]

# Calculate combinations of different a values: a * roof + (1 - a) * facade
a_values = np.arange(0, 1.02, 0.02)  # Iterate over a with a step of 0.02

norm = mcolors.Normalize(vmin=a_values.min(), vmax=a_values.max())

# Set up the figure
font_options = {'size': fs-3}  # Adjust the font size based on actual needs
plt.rc('font', **font_options)

# Loop through each city to plot the graphs
for i_city, (city, city_idx) in enumerate(zip(cities_actual, city_indices)):

    # Get the summer and winter curves for the current city
    roof_summer = avg_summer_roof_norm[city_idx]
    facade_summer = avg_summer_facade_norm[city_idx]
    roof_winter = avg_winter_roof_norm[city_idx]
    facade_winter = avg_winter_facade_norm[city_idx]

    n_points_day = len(facade_summer)

    # Create a new figure
    fig, axes = plt.subplots(2, 1, figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54,
            sharex=True, sharey=True)
    
    x = np.arange(n_points_day) / (n_points_day / 24)
    
    for i_ax, ax in enumerate(axes):
        ax = axes[i_ax]

        if i_ax == 0:  # Plot Summer curves
            facade_curve, roof_curve = facade_summer, roof_summer
        else:
            facade_curve, roof_curve = facade_winter, roof_winter
        ax.plot(x, facade_curve, linewidth=lw_plot, marker='o', markersize=5,  # Default size 6
                color=colors[-1], linestyle='-', label=f'FPV curve',
                alpha=alpha_plot, zorder=5)
        ax.plot(x, roof_curve, linewidth=lw_plot, marker='o', markersize=5,  # Default size 6
                color=colors[0], linestyle='-', label=f'RPV curve',
                alpha=alpha_plot, zorder=6)
        ax.fill_between(
                x, facade_curve, roof_curve,
                fc=color_fill,
                alpha=alpha_fill, zorder=-4,  # label='Difference'
            )
        plt.rcParams['hatch.linewidth'] = 2.0  # Default is 1.0, can be set to a thicker value like 2.0
    
    # Set titles, labels, and grid lines
    for i_ax, ax in enumerate(axes):
        ax.set_xlim(5, 20)
        ax.set_xticks(np.arange(5, 21, 3))
        ax.set_ylim(-0.2, 4.2)
        ax.set_yticks(np.arange(0, 5, 2))

        ax.grid(alpha=0.5)

        if i_ax == 0:
            ax.text(
                0.05, 0.96, "Summer", color=color_season,
                transform=ax.transAxes,         # Using Axes coordinates (0~1)
                ha='left', va='top',            # Left-align + Top-align
                fontsize=fs - 6,                # Font size can be adjusted
                weight='normal'                   # Font weight (optional)
            )
        if i_ax == 1:
            ax.text(
                0.05, 0.96, "Winter", color=color_season,
                transform=ax.transAxes,         # Using Axes coordinates (0~1)
                ha='left', va='top',            # Left-align + Top-align
                fontsize=fs - 6,                # Font size can be adjusted
                weight='normal'                   # Font weight (optional)
            )

    fig.add_subplot(1, 1, 1, frameon=False)  # Add a hidden global coordinate axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Hour of day',  # labelpad=xlabel_pad,
        fontsize=fs-3, loc='center')
    plt.ylabel('Normalized power (\u2013)',  labelpad=0,
        fontsize=fs-3, loc='center')

    # Add legend
    lg = axes[0].legend(fontsize=fs - 9, loc='upper right', ncol=3)
    lg.get_frame().set(linewidth=0.6, edgecolor='black', facecolor='none')

    # Adjust subplot layout
    plt.subplots_adjust(left=0.13, right=0.96, bottom=0.15, top=0.93,
            wspace=0.25, hspace=0.05
            )

    # Save the figure
    axes[0].set_title(f'Northern city', fontsize=fs-3)
    fig.savefig('Figs_new_supp/sFig2_time_daily_north.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig("Figs_new_supp/sFig2_time_daily_north.png", dpi=600, bbox_inches='tight')

    # Display the figure
    # plt.tight_layout()
    plt.show()

#%% Typical Southern City
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

color_fill = '#87CEEB'
color_season = 'black'
alpha_plot = 1
alpha_fill = 0.4
lw_plot = 1.5

# Select cities
cities = {'Hangzhou'}

# Get the row index for the cities in area_df
city_indices = [area_df.index.get_loc(city) for city in cities if city in area_df.index]
cities_actual = area_df.index[city_indices]

# Calculate combinations of different a values: a * roof + (1 - a) * facade
a_values = np.arange(0, 1.02, 0.02)  # Iterate over a with a step of 0.02

norm = mcolors.Normalize(vmin=a_values.min(), vmax=a_values.max())

# Set up the figure
font_options = {'size': fs-3}  # Adjust the font size based on actual needs
plt.rc('font', **font_options)

# Loop through each city to plot the graphs
for i_city, (city, city_idx) in enumerate(zip(cities_actual, city_indices)):

    # Get the summer and winter curves for the current city
    roof_summer = avg_summer_roof_norm[city_idx]
    facade_summer = avg_summer_facade_norm[city_idx]
    roof_winter = avg_winter_roof_norm[city_idx]
    facade_winter = avg_winter_facade_norm[city_idx]

    n_points_day = len(facade_summer)

    # Create a new figure
    fig, axes = plt.subplots(2, 1, figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54,
            sharex=True, sharey=True)
    
    x = np.arange(n_points_day) / (n_points_day / 24)
    
    for i_ax, ax in enumerate(axes):
        ax = axes[i_ax]

        if i_ax == 0:  # Plot Summer curves
            facade_curve, roof_curve = facade_summer, roof_summer
        else:
            facade_curve, roof_curve = facade_winter, roof_winter
        ax.plot(x, facade_curve, linewidth=lw_plot, marker='o', markersize=5,  # Default size 6
                color=colors[-1], linestyle='-', label=f'FPV curve',
                alpha=alpha_plot, zorder=5)
        ax.plot(x, roof_curve, linewidth=lw_plot, marker='o', markersize=5,  # Default size 6
                color=colors[0], linestyle='-', label=f'RPV curve',
                alpha=alpha_plot, zorder=6)
        ax.fill_between(
                x, facade_curve, roof_curve,
                fc=color_fill,
                alpha=alpha_fill, zorder=-4,  # label='Difference'
            )
        plt.rcParams['hatch.linewidth'] = 2.0  # Default is 1.0, can be set to a thicker value like 2.0
    
    # Set titles, labels, and grid lines
    for i_ax, ax in enumerate(axes):
        ax.set_xlim(5, 20)
        ax.set_xticks(np.arange(5, 21, 3))
        ax.set_ylim(-0.2, 4.2)
        ax.set_yticks(np.arange(0, 5, 2))

        ax.grid(alpha=0.5)

        if i_ax == 0:
            ax.text(
                0.05, 0.96, "Summer", color=color_season,
                transform=ax.transAxes,         # Using Axes coordinates (0~1)
                ha='left', va='top',            # Left-align + Top-align
                fontsize=fs - 6,                # Font size can be adjusted
                weight='normal'                   # Font weight (optional)
            )
        if i_ax == 1:
            ax.text(
                0.05, 0.96, "Winter", color=color_season,
                transform=ax.transAxes,         # Using Axes coordinates (0~1)
                ha='left', va='top',            # Left-align + Top-align
                fontsize=fs - 6,                # Font size can be adjusted
                weight='normal'                   # Font weight (optional)
            )

    fig.add_subplot(1, 1, 1, frameon=False)  # Add a hidden global coordinate axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Hour of day',  # labelpad=xlabel_pad,
        fontsize=fs-3, loc='center')
    plt.ylabel('Normalized power (\u2013)',  labelpad=0,
        fontsize=fs-3, loc='center')

    # Add legend
    lg = axes[0].legend(fontsize=fs - 9, loc='upper right', ncol=3)
    lg.get_frame().set(linewidth=0.6, edgecolor='black', facecolor='none')

    # Adjust subplot layout
    plt.subplots_adjust(left=0.13, right=0.96, bottom=0.15, top=0.93,
            wspace=0.25, hspace=0.05
            )

    # Save the figure
    axes[0].set_title(f'Southern city', fontsize=fs-3)
    fig.savefig('Figs_new_supp/sFig2_time_daily_south.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig("Figs_new_supp/sFig2_time_daily_south.png", dpi=600, bbox_inches='tight')
    plt.show()

#%% Subfigure (b)：Seasonal Curve
### Typical Northern City
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

color_roof = colors[0]
color_facade = colors[-1]
color_fill = '#87CEEB'
lw_axis = 0.8
lw_plot = 1.5
alpha_fill = 0.4

# Example: selecting one city for demonstration
cities = ['Yinchuan']  # list(city_north.index)[:1]
city_indices = [area_df.index.get_loc(city) for city in cities if city in area_df.index]
cities_actual = area_df.index[city_indices]

# The following example uses summer data; winter can be plotted similarly.
for i_city, (city, city_idx) in enumerate(zip(cities_actual, city_indices)):
    # Get the normalized hourly curve for the current city over the year
    facade = ideal_facade_curve_norm_annual[city_idx]
    roof = roof_curve_norm_annual[city_idx]

    # Assume m is the number of data points in a year (e.g., m = 8760 or m = 8760*4), here m=8760
    m = len(roof)
    # Number of data points per day (assuming m=8760, so 1 data point per day; if m=8760*4, points_per_day=4)
    points_per_day = m // 365

    # Calculate the daily average: reshape annual data to (365, points_per_day) and calculate mean
    facade_daily = np.mean(facade.reshape(365, points_per_day), axis=1)
    roof_daily = np.mean(roof.reshape(365, points_per_day), axis=1)

    # Divide the 365 days into 12 months (approximated: Jan 31, Feb 28, Mar 31, Apr 30, etc.)
    month_boundaries = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    facade_monthly = []
    roof_monthly = []
    for i in range(12):
        start = month_boundaries[i]
        end = month_boundaries[i+1]
        facade_monthly.append(facade_daily[start:end])
        roof_monthly.append(roof_daily[start:end])
    
    # Calculate the monthly average
    facade_monthly_mean = [np.mean(arr) for arr in facade_monthly]
    roof_monthly_mean = [np.mean(arr) for arr in roof_monthly]
    
    # ---------- Plot the curves ----------
    # Create figure and axes (for one city)
    font_options = {'size': fs-3}
    plt.rc('font', **font_options)
    fig, ax = plt.subplots(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)
       
    # x-axis positions for each month, using 1-12 to represent months
    x_positions = np.arange(1, 13)
    
    # Plot the average facade (FPV) curve
    ax.plot(x_positions, facade_monthly_mean, marker='o', linestyle='-', color=color_facade, 
            linewidth=lw_plot, label='FPV')
    # Plot the average roof (RPV) curve
    ax.plot(x_positions, roof_monthly_mean, marker='o', linestyle='-', color=color_roof, 
            linewidth=lw_plot, label='RPV')
    ax.fill_between(
            x_positions, facade_monthly_mean, roof_monthly_mean,
            fc=color_fill,
            alpha=alpha_fill, zorder=-4,  # label='Difference'
        )
    
    # Set axis labels, title, and grid lines
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks(np.arange(0.5, 1.6, 0.2))

    ax.set_xlabel("Month", fontsize=fs-3)
    ax.set_ylabel("Normalized monthly generation", fontsize=fs-5)
    ax.set_xticks(x_positions)
    ax.grid(alpha=0.5)
    
    # Add legend
    lg = ax.legend(fontsize=fs-9, loc='upper right', ncol=1)
    lg.get_frame().set(linewidth=0.6, edgecolor='black', facecolor='none')
    
    # Adjust subplot layout
    plt.subplots_adjust(left=0.13, right=0.96, bottom=0.15, top=0.93,
            wspace=0.25, hspace=0.05
            )
    
    # Save the figure
    if city_north.loc[city].iloc[0] == 1:
        plt.title(f'Northern city', fontsize=fs-3)
        fig.savefig('Figs_new_supp/sFig2_time_monthly_north.pdf', format='pdf', dpi=600, bbox_inches='tight')
        fig.savefig("Figs_new_supp/sFig2_time_monthly_north.png", dpi=600, bbox_inches='tight')
    plt.show()

#%% Typical Southern City
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

color_roof = colors[0]
color_facade = colors[-1]
color_fill = '#87CEEB'
lw_axis = 0.8
lw_plot = 1.5
alpha_fill = 0.4

# Example: selecting one city for demonstration
cities = ['Guangzhou']  # list(city_north.index)[:1]
city_indices = [area_df.index.get_loc(city) for city in cities if city in area_df.index]
cities_actual = area_df.index[city_indices]

# The following example uses summer data; winter can be plotted similarly.
for i_city, (city, city_idx) in enumerate(zip(cities_actual, city_indices)):
    # Get the normalized hourly curve for the current city over the year
    facade = ideal_facade_curve_norm_annual[city_idx]
    roof = roof_curve_norm_annual[city_idx]

    # Assume m is the number of data points in a year (e.g., m = 8760 or m = 8760*4), here m=8760
    m = len(roof)
    # Number of data points per day (assuming m=8760, so 1 data point per day; if m=8760*4, points_per_day=4)
    points_per_day = m // 365

    # Calculate the daily average: reshape annual data to (365, points_per_day) and calculate mean
    facade_daily = np.mean(facade.reshape(365, points_per_day), axis=1)
    roof_daily = np.mean(roof.reshape(365, points_per_day), axis=1)

    # Divide the 365 days into 12 months (approximated: Jan 31, Feb 28, Mar 31, Apr 30, etc.)
    month_boundaries = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    facade_monthly = []
    roof_monthly = []
    for i in range(12):
        start = month_boundaries[i]
        end = month_boundaries[i+1]
        facade_monthly.append(facade_daily[start:end])
        roof_monthly.append(roof_daily[start:end])
    
    # Calculate the monthly average
    facade_monthly_mean = [np.mean(arr) for arr in facade_monthly]
    roof_monthly_mean = [np.mean(arr) for arr in roof_monthly]
    
    # ---------- Plot the curves ----------
    # Create figure and axes (for one city)
    font_options = {'size': fs-3}
    plt.rc('font', **font_options)
    fig, ax = plt.subplots(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)
       
    # x-axis positions for each month, using 1-12 to represent months
    x_positions = np.arange(1, 13)
    
    # Plot the average facade (FPV) curve
    ax.plot(x_positions, facade_monthly_mean, marker='o', linestyle='-', color=color_facade, 
            linewidth=lw_plot, label='FPV')
    # Plot the average roof (RPV) curve
    ax.plot(x_positions, roof_monthly_mean, marker='o', linestyle='-', color=color_roof, 
            linewidth=lw_plot, label='RPV')
    ax.fill_between(
            x_positions, facade_monthly_mean, roof_monthly_mean,
            fc=color_fill,
            alpha=alpha_fill, zorder=-4,  # label='Difference'
        )
    
    # Set axis labels, title, and grid lines
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks(np.arange(0.5, 1.6, 0.2))

    ax.set_xlabel("Month", fontsize=fs-3)
    ax.set_ylabel("Normalized monthly generation", fontsize=fs-5)
    ax.set_xticks(x_positions)
    ax.grid(alpha=0.5)
    
    # Add legend
    lg = ax.legend(fontsize=fs-9, loc='upper right', ncol=1)
    lg.get_frame().set(linewidth=0.6, edgecolor='black', facecolor='none')
    
    # Adjust subplot layout
    plt.subplots_adjust(left=0.13, right=0.96, bottom=0.15, top=0.93,
            wspace=0.25, hspace=0.05
            )
    

    plt.title(f'Southern city', fontsize=fs-3)
    fig.savefig('Figs_new_supp/sFig2_time_monthly_south.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig("Figs_new_supp/sFig2_time_monthly_south.png", dpi=600, bbox_inches='tight')
    
    plt.show()

#%% Supplementary of Fig. 3. Metrics of Temporal Characteristics
## Data Preparation
# ------------------- Read Area Data -------------------  
# These correspond to roof area, maximum facade area, and filtered facade area
area_df = pd.read_excel('Fig_input_data/City_statistic.xlsx', sheet_name='Key_information',
        usecols=[0, 3, 4, 5], index_col=[0])
# Contains data from different stages. Roof is the roof capacity, Ideal is the ideal facade capacity, and Facade is the economically viable facade capacity. Units: GW
# The subsequent numbers represent different development stages, where photovoltaic prices and efficiency change, so Ideal is also split into different stages
cap_df = pd.read_excel('Fig_input_data/City_Cap.xlsx', index_col=[0])

# Convert the DataFrame to numpy arrays
area_data = area_df.to_numpy()
cap_data = cap_df.to_numpy()

# ------------------- Selected City Population and Urbanization Rate ------------------- 
population_df = pd.read_excel('Fig_input_data/City_info.xlsx', sheet_name=0,
        usecols=list(np.arange(8)))  # The last column 'City' is the city index for the entire study

population_df.set_index('City', inplace=True)

population_df = assign_city_level(population_df)

# ------------------- Building Average Height Data ------------------- 
# File path
file_path = 'Fig_input_data/City_statistic.xlsx'  # Path to the Excel file
# Read building total volume
building_volume_df = pd.read_excel(file_path, index_col=0)  
total_building_volume = building_volume_df.iloc[:, 0].copy()
# Read total roof area of buildings
key_information = pd.read_excel(file_path, sheet_name=2, index_col=0)
total_building_area = key_information['Area_roof-0(km2)']
# Calculate average building height in meters
average_building_height = total_building_volume / total_building_area * 1000
average_building_height.name = 'average_building_height'
import numpy as np
import pandas as pd

# ------------------- Retrieve Ideal Facade Generation Curves -------------------  
file_path = 'Fig_input_data/'
ideal_facade_data_1_path = file_path + 'Power_facade_ideal_1.npy'
roof_data_1_path = file_path + 'Power_roof_1.npy'

ideal_facade_curve = np.load(ideal_facade_data_1_path)
roof_curve = np.load(roof_data_1_path)
summer_curve_data = np.zeros((102, 2, 92, 24)) 
winter_curve_data = np.zeros((102, 2, 90, 24)) 
roof_daily = np.zeros((102, 365))
facade_daily = np.zeros((102, 365))

# Process summer curves
for i in range(92):
    for j in range(24):
        for k in range(102):
            summer_curve_data[k, 0, i, j] = ideal_facade_curve[k, 151 * 24 + i * 24 + j] / 1e3
            summer_curve_data[k, 1, i, j] = roof_curve[k, 151 * 24 + i * 24 + j] / 1e3

# Process winter curves
for i in range(90):
    for j in range(24):
        for k in range(102):
            if 334 * 24 + i * 24 + j < 365 * 24: 
                winter_curve_data[k, 0, i, j] = ideal_facade_curve[k, 334 * 24 + i * 24 + j] / 1e3
                winter_curve_data[k, 1, i, j] = roof_curve[k, 334 * 24 + i * 24 + j] / 1e3
            else:
                winter_curve_data[k, 0, i, j] = ideal_facade_curve[k, 334 * 24 + i * 24 + j - 365 * 24] / 1e3
                winter_curve_data[k, 1, i, j] = roof_curve[k, 334 * 24 + i * 24 + j - 365 * 24] / 1e3

# Calculate various features (average, median, peak times)
avg_summer_facade = np.zeros((102, 24))
avg_summer_roof = np.zeros((102, 24))
avg_winter_facade = np.zeros((102, 24))
avg_winter_roof = np.zeros((102, 24))
median_summer_facade = np.zeros((102, 24))
median_summer_roof = np.zeros((102, 24))
median_winter_facade = np.zeros((102, 24))
median_winter_roof = np.zeros((102, 24))
peak_time_summer_facade = np.zeros((102, 92))
peak_time_summer_roof = np.zeros((102, 92))
peak_time_winter_facade = np.zeros((102, 90))
peak_time_winter_roof = np.zeros((102, 90))
avg_peak_time_summer_facade = np.zeros(102)
avg_peak_time_summer_roof = np.zeros(102)
avg_peak_time_winter_facade = np.zeros(102)
avg_peak_time_winter_roof = np.zeros(102)
ratio_summer_winter_facade_peak = np.zeros(102)
ratio_summer_winter_roof_peak = np.zeros(102)

# Compute average, median, and peak times
for i in range(102):
    for j in range(24):
        avg_summer_facade[i, j] = np.mean(summer_curve_data[i, 0, :, j])
        avg_summer_roof[i, j] = np.mean(summer_curve_data[i, 1, :, j])
        avg_winter_facade[i, j] = np.mean(winter_curve_data[i, 0, :, j])
        avg_winter_roof[i, j] = np.mean(winter_curve_data[i, 1, :, j])
        median_summer_facade[i, j] = np.median(summer_curve_data[i, 0, :, j])
        median_summer_roof[i, j] = np.median(summer_curve_data[i, 1, :, j])
        median_winter_facade[i, j] = np.median(winter_curve_data[i, 0, :, j])
        median_winter_roof[i, j] = np.median(winter_curve_data[i, 1, :, j])

# Find peak times
for i in range(102):
    for j in range(92):
        peak_time_summer_facade[i, j] = np.argmax(summer_curve_data[i, 0, j, :])
        peak_time_summer_roof[i, j] = np.argmax(summer_curve_data[i, 1, j, :])
    for j in range(90):
        peak_time_winter_facade[i, j] = np.argmax(winter_curve_data[i, 0, j, :])
        peak_time_winter_roof[i, j] = np.argmax(winter_curve_data[i, 1, j, :])

# Calculate average peak times
for i in range(102):
    avg_peak_time_summer_facade[i] = np.mean(peak_time_summer_facade[i])
    avg_peak_time_summer_roof[i] = np.mean(peak_time_summer_roof[i])
    avg_peak_time_winter_facade[i] = np.mean(peak_time_winter_facade[i])
    avg_peak_time_winter_roof[i] = np.mean(peak_time_winter_roof[i])

# Compute standard deviation, coefficient of variation (CV), and peak values
std_summer_facade = np.zeros(102)
std_summer_roof = np.zeros(102)
std_winter_facade = np.zeros(102)
std_winter_roof = np.zeros(102)
cv_summer_facade = np.zeros(102)
cv_summer_roof = np.zeros(102)
cv_winter_facade = np.zeros(102)
cv_winter_roof = np.zeros(102)
peak_summer_facade = np.zeros(102)
peak_summer_roof = np.zeros(102)
peak_winter_facade = np.zeros(102)
peak_winter_roof = np.zeros(102)

# Calculate standard deviation and CV for each city
for i in range(102):
    std_summer_facade[i] = np.std(avg_summer_facade[i])
    std_summer_roof[i] = np.std(avg_summer_roof[i])
    std_winter_facade[i] = np.std(avg_winter_facade[i])
    std_winter_roof[i] = np.std(avg_winter_roof[i])
    cv_summer_facade[i] = std_summer_facade[i] / np.mean(avg_summer_facade[i])
    cv_summer_roof[i] = std_summer_roof[i] / np.mean(avg_summer_roof[i])
    cv_winter_facade[i] = std_winter_facade[i] / np.mean(avg_winter_facade[i])
    cv_winter_roof[i] = std_winter_roof[i] / np.mean(avg_winter_roof[i])
    peak_summer_facade[i] = np.max(avg_summer_facade[i])
    peak_summer_roof[i] = np.max(avg_summer_roof[i])
    peak_winter_facade[i] = np.max(avg_winter_facade[i])
    peak_winter_roof[i] = np.max(avg_winter_roof[i])

# Calculate the ratio of summer peak to winter peak
for i in range(102):
    ratio_summer_winter_facade_peak[i] = peak_summer_facade[i] / peak_winter_facade[i] - 1
    ratio_summer_winter_roof_peak[i] = peak_summer_roof[i] / peak_winter_roof[i] - 1

# Display the CV values for summer and winter using box plots
cv_summer = np.zeros((102, 2))
cv_winter = np.zeros((102, 2))
for i in range(102):
    cv_summer[i][0] = cv_summer_facade[i]
    cv_summer[i][1] = cv_summer_roof[i]
    cv_winter[i][0] = cv_winter_facade[i]
    cv_winter[i][1] = cv_winter_roof[i]

df_cv_summer = pd.DataFrame(cv_summer, columns=['Summer Facade', 'Summer Roof'])
df_cv_winter = pd.DataFrame(cv_winter, columns=['Winter Facade', 'Winter Roof'])

# Merge the dataframes
df_cv_combined = pd.concat([df_cv_summer, df_cv_winter], axis=1)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ['skyblue', 'mediumseagreen']  # Blue, Green corresponding to Roof and Facade
color_roof = colors[0]
color_facade = colors[1]

lw_axis = 0.8
lw_box = 1.0

# Extract data for each column
summer_facade_data = df_cv_combined['Summer Facade'].dropna().values
summer_roof_data   = df_cv_combined['Summer Roof'].dropna().values
winter_facade_data = df_cv_combined['Winter Facade'].dropna().values
winter_roof_data   = df_cv_combined['Winter Roof'].dropna().values

# Create the plot
font_options = {'size': fs - 3}
plt.rc('font', **font_options)
fig, ax = plt.subplots(figsize=np.array([figwidth * 2/3, figwidth * 0.6]) / 2.54,)  # Figure size in inches

# Define common properties for the box plot
common_props = dict(
    showfliers=False,      # Do not show outliers
    patch_artist=True,     # Allow custom fill color for boxes
    showmeans=True,        # Show the mean value
    widths=0.2,            # Box width
    medianprops=dict(color='orange', linewidth=lw_box),  # Median line style
    meanprops=dict(marker='x', markerfacecolor='black',
                   markeredgecolor='black', markersize=7, linewidth=lw_axis)
)

# Place two groups on the x-axis: 0 for Summer, 1 for Winter
x_positions = [0, 1]
offset_facade = -0.2
offset_roof   = +0.2

# 1) Summer Facade
bp_sf = ax.boxplot(
    [summer_facade_data], 
    positions=[x_positions[0] + offset_facade],
    **common_props
)
for box in bp_sf['boxes']:
    box.set_facecolor(color_facade)

# 2) Summer Roof
bp_sr = ax.boxplot(
    [summer_roof_data], 
    positions=[x_positions[0] + offset_roof],
    **common_props
)
for box in bp_sr['boxes']:
    box.set_facecolor(color_roof)

# 3) Winter Facade
bp_wf = ax.boxplot(
    [winter_facade_data], 
    positions=[x_positions[1] + offset_facade],
    **common_props
)
for box in bp_wf['boxes']:
    box.set_facecolor(color_facade)

# 4) Winter Roof
bp_wr = ax.boxplot(
    [winter_roof_data], 
    positions=[x_positions[1] + offset_roof],
    **common_props
)
for box in bp_wr['boxes']:
    box.set_facecolor(color_roof)

ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=lw_axis)

# Set ticks and labels
ax.set_xlim(-0.5, 1.5)
ax.set_xticks(x_positions)
ax.set_xticklabels(['Summer', 'Winter'], fontsize=fs-3)

ax.set_ylim(0.9, 1.5)
ax.set_yticks(np.arange(0.9, 1.6, 0.1))

# Axis labels
ax.set_title("Coefficient of variation", fontsize=fs-3)

# Manually create legend
patch_facade = mpatches.Patch(facecolor=color_facade, edgecolor='black', label='FPV')
patch_roof   = mpatches.Patch(facecolor=color_roof, edgecolor='black', label='RPV')
lg = ax.legend(handles=[patch_facade, patch_roof], fontsize=fs-6, loc='lower right')
lg.get_frame().set(linewidth=0.6, edgecolor='black', facecolor='none')

# Adjust subplot layout
plt.subplots_adjust(left=0.17, right=0.98, bottom=0.07, top=0.9,
		# wspace=0.45, hspace=0
        )

fig.savefig('Figs_new_supp/sFig2_time_metric_cv.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig2_time_metric_cv.png", dpi=600, bbox_inches='tight')

plt.show()


#%% Subfigure (b): Peak Power Duration
# Peak power duration data
peak_time_summer = np.zeros((102, 2))
peak_time_winter = np.zeros((102, 2))

for i in range(102):
    peak_time_summer[i][0] = avg_peak_time_summer_facade[i]
    peak_time_summer[i][1] = avg_peak_time_summer_roof[i]
    peak_time_winter[i][0] = avg_peak_time_winter_facade[i]
    peak_time_winter[i][1] = avg_peak_time_winter_roof[i]

df_peak_time_summer = pd.DataFrame(peak_time_summer, columns=['Summer Facade', 'Summer Roof'])
df_peak_time_winter = pd.DataFrame(peak_time_winter, columns=['Winter Facade', 'Winter Roof'])

# Merge Dataframes
df_peak_time_combined = pd.concat([df_peak_time_summer, df_peak_time_winter], axis=1)

lw_axis = 0.8
lw_box = 1.0

# Extracting data from columns
summer_facade_data = df_peak_time_combined['Summer Facade'].dropna().values
summer_roof_data   = df_peak_time_combined['Summer Roof'].dropna().values
winter_facade_data = df_peak_time_combined['Winter Facade'].dropna().values
winter_roof_data   = df_peak_time_combined['Winter Roof'].dropna().values

# Creating the plot
font_options = {'size': fs - 3}
plt.rc('font', **font_options)
fig, ax = plt.subplots(figsize=np.array([figwidth * 2/3, figwidth * 0.6]) / 2.54,)  # Figure size in inches

# Defining common properties for the box plot
common_props = dict(
    showfliers=False,      # Do not show outliers
    patch_artist=True,     # Allow custom box fill colors
    showmeans=True,        # Show the mean
    widths=0.2,            # Box width
    medianprops=dict(color='orange', linewidth=lw_box),  # Median line style
    meanprops=dict(marker='x', markerfacecolor='black',
                   markeredgecolor='black', markersize=7, linewidth=lw_axis)
)

# Positioning two groups on the x-axis: 0 for Summer, 1 for Winter
x_positions = [0, 1]
offset_facade = -0.2
offset_roof   = +0.2

# 1) Summer Facade
bp_sf = ax.boxplot(
    [summer_facade_data], 
    positions=[x_positions[0] + offset_facade],
    **common_props
)
for box in bp_sf['boxes']:
    box.set_facecolor(color_facade)

# 2) Summer Roof
bp_sr = ax.boxplot(
    [summer_roof_data], 
    positions=[x_positions[0] + offset_roof],
    **common_props
)
for box in bp_sr['boxes']:
    box.set_facecolor(color_roof)

# 3) Winter Facade
bp_wf = ax.boxplot(
    [winter_facade_data], 
    positions=[x_positions[1] + offset_facade],
    **common_props
)
for box in bp_wf['boxes']:
    box.set_facecolor(color_facade)

# 4) Winter Roof
bp_wr = ax.boxplot(
    [winter_roof_data], 
    positions=[x_positions[1] + offset_roof],
    **common_props
)
for box in bp_wr['boxes']:
    box.set_facecolor(color_roof)

ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=lw_axis)

# Setting ticks and labels
ax.set_xlim(-0.5, 1.5)
ax.set_xticks(x_positions)
ax.set_xticklabels(['Summer', 'Winter'], fontsize=fs-3)

ax.set_ylim(12, 15)
ax.set_yticks(np.arange(12, 15.5, 1.0))
ax.set_yticklabels(['12:00','13:00','14:00','15:00'])

# Axis labels
ax.set_title("Peak time", fontsize=fs - 3)

# Manually creating a legend
patch_facade = mpatches.Patch(facecolor=color_facade, edgecolor='black', label='FPV')
patch_roof   = mpatches.Patch(facecolor=color_roof, edgecolor='black', label='RPV')
lg = ax.legend(handles=[patch_facade, patch_roof], fontsize=fs-6, loc='upper left')
lg.get_frame().set(linewidth=0.6, edgecolor='black', facecolor='none')

# Adjusting subplot layout
plt.subplots_adjust(left=0.17, right=0.98, bottom=0.07, top=0.9,
		# wspace=0.45, hspace=0
        )

fig.savefig('Figs_new_supp/sFig2_time_metric_peak.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig2_time_metric_peak.png", dpi=600, bbox_inches='tight')

plt.show()


#%% Subfigure (c): Summer-to-winter peak ratio
import numpy as np
import pandas as pd

# Combine area data and peak ratio data
df_plot = area_df.iloc[:, :2].copy()  # km^2

# Calculate the ratio of facade and roof area to generation in summer and winter peak
df_plot['facade_summer_winter_peak'] = ratio_summer_winter_facade_peak + 1
df_plot['roof_summer_winter_peak'] = ratio_summer_winter_roof_peak + 1

# Merge population data
df_plot = df_plot.merge(population_df, left_index=True, right_index=True)

color_ratio = '#003366'
lw_axis = 0.8

# ----------------- Data preparation -----------------
# The index of df_plot is city name, including columns:
#   city_level, Area_facade-0(km2), Area_roof-0(km2)
df_vis = df_plot.copy()
df_vis = df_vis.rename(index={
    'Haerbin': 'Harbin',
    'Huhehaote': 'Hohhot',
    'Wulumuqi': 'Urumqi',
    'Xian': "Xi'an"
})

# 1) Sort by city_level (descending) and facade_power (descending)
df_vis.sort_values(
    by=['city_level', 'facade_summer_winter_peak'],
    ascending=[False, False],
    inplace=True
)

# 2) Split the cities into two parts
n = len(df_vis)
mid = n // 2
df_vis_part1 = df_vis.iloc[:mid]
df_vis_part2 = df_vis.iloc[mid:]

# Define a mapping dictionary for city levels
level_labels = {
    3: 'SLC',
    2: 'VLC',
    1: 'LC-I',
    0: 'LC-II'
}

# ----------------- Create figure and subplots -----------------
font_options = {'size': fs-12}
plt.rc('font', **font_options)

fig, axes = plt.subplots(
    1, 2,
    figsize=np.array([figwidth, figwidth * 1.2]) / 2.54,
)

parts = [df_vis_part1, df_vis_part2]

for i_ax, df_part in enumerate(parts):
    ax = axes[i_ax]

    # Construct a temporary table temp_part to keep city_level for annotation
    temp_part = df_part[['city_level','facade_summer_winter_peak', 'roof_summer_winter_peak']].copy()
    # Only two columns are used for plotting
    df_plot_part = temp_part[['facade_summer_winter_peak', 'roof_summer_winter_peak']]

    # Plot horizontal bar chart
    df_plot_part.plot(
        kind='barh',
        ax=ax,
        stacked=False,
        width=0.8,
        edgecolor='none',
        linewidth=lw_axis,
        alpha=0.8,
        color=colors[::-1],  # Facade, Roof
        legend=False,
    )

    # Draw vertical red guide line at x=1
    ax.axvline(x=1, linestyle='-', color='darkred', linewidth=lw, alpha=1)

    # Invert y-axis to show from top to bottom
    ax.invert_yaxis()

    ax.set_xlim(0, 2.5)
    ax.set_xticks(np.arange(0, 2.6, 0.5))
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Ratio of summer to winter peak (\u2013)", fontsize=fs-9)
    ax.set_ylabel(None)

    # -------------- Annotate city levels --------------
    # Reset index so row_idx corresponds to y-axis order in barh
    temp_part.reset_index(inplace=True)       
    temp_part['row_idx'] = temp_part.index    

    # Group by city_level to find min and max y-index
    grouped = temp_part.groupby('city_level')['row_idx']
    for lvl, rows in grouped:
        min_idx = rows.min()
        max_idx = rows.max()
        mid_idx = 0.5 * (min_idx + max_idx)  # Vertical midpoint

        # x_pos: horizontal position for arrow/label
        x_pos = ax.get_xlim()[-1] * 0.83
        arrow_len = 0.5
        h_len = ax.get_xlim()[-1] * 0.04  # Half length of horizontal line

        # (1) Top arrow
        ax.annotate(
            '',
            xy=(x_pos, min_idx),
            xytext=(x_pos, min_idx - arrow_len),
            arrowprops=dict(arrowstyle='<-', lw=lw_axis, color='black', mutation_scale=20),
            annotation_clip=False,
            transform=ax.transData,
        )
        ax.plot(
            [x_pos - h_len, x_pos + h_len],
            [min_idx-0.3, min_idx-0.3],
            color='black',
            lw=lw_axis,
            transform=ax.transData,
            clip_on=False
        )

        # (2) Vertical line
        ax.plot(
            [x_pos, x_pos], 
            [min_idx, max_idx],
            color='black', lw=lw_axis,
            transform=ax.transData,
            clip_on=False
        )

        # (3) Bottom arrow
        ax.annotate(
            '',
            xy=(x_pos, max_idx),
            xytext=(x_pos, max_idx + arrow_len),
            arrowprops=dict(arrowstyle='<-', lw=lw_axis, color='black', mutation_scale=20),
            annotation_clip=False,
            transform=ax.transData
        )
        ax.plot(
            [x_pos - h_len, x_pos + h_len],
            [max_idx+0.3, max_idx+0.3],
            color='black',
            lw=lw_axis,
            transform=ax.transData,
            clip_on=False
        )

        # (4) Add label at midpoint
        text_label = level_labels.get(lvl, f'Level {lvl}')
        ax.text(
            x_pos, mid_idx,
            text_label,
            ha='center', va='center',
            fontsize=fs-9,
            transform=ax.transData,
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.5
            )
        )

# ----------------- Add shared legend -----------------
lg = fig.legend(
    handles=[
        mpatches.Patch(fc=colors[1], label='FPV peak ratio'),  # Green
        mpatches.Patch(fc=colors[0], label='RPV peak ratio'),  # Blue
    ],
    handleheight=0.7,
    handlelength=2,
    loc='lower center',
    ncol=3,
    fontsize=fs - 9,
    bbox_to_anchor=(0.5, 0.00)
)
frame = lg.get_frame()
frame.set_linewidth(0.6)
frame.set_edgecolor('black')
frame.set_facecolor('none')

# ----------------- Adjust layout and save -----------------
plt.subplots_adjust(left=0.10, right=0.98, bottom=0.1, top=0.95,
		wspace=0.45, hspace=0
        )

fig.savefig('Figs_new_supp/sFig2_time_metric_bar.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig2_time_metric_bar.png", dpi=600, bbox_inches='tight')

plt.show()
# Bottom




