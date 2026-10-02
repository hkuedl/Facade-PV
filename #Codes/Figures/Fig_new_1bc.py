import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
from matplotlib import rcParams
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
CITY_STATISTICS_FILE = Path(
    os.environ.get(
        "FPV_CITY_STATISTICS_FILE",
        OPTIMIZATION_DATA_DIR / "Other_parameters" / "City_statistic.xlsx",
    )
).expanduser()
MAIN_CASE_DIR = FIG_INPUT_DIR
PLANNING_RESULTS_DIR = Path(
    os.environ.get("FPV_OPT_RESULTS_DIR", OPTIMIZATION_DATA_DIR / "Planning_results")
).expanduser()
FIG_OUTPUT_DIR = Path(
    os.environ.get("FPV_FIG_OUTPUT_DIR", SCRIPT_DIR / "Figs_new")
).expanduser()
FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OPTIMIZATION_DATA_DIR)

font_path = SCRIPT_DIR / "arial.ttf"
if font_path.exists():
    custom_font = fm.FontProperties(fname=str(font_path))
    fm.fontManager.addfont(str(font_path))
    rcParams["font.family"] = custom_font.get_name()

ratio_plot = 3

figwidth =  8.5 * ratio_plot
fs = 8 * ratio_plot
lw = 0.4 * ratio_plot
lw2 = 0.75 * ratio_plot
lw3 = 1 * ratio_plot

grid_alpha = 0.5

ABNORMAL_VOLUME_CITY_INDICES = {2, 6, 8, 69, 92, 97}

# ------------------- Read area and capacuty data -------------------
# Corresponds to roof area, maximum facade area, and filtered facade area

area_df = pd.read_excel('Fig_input_data/City_statistic.xlsx', sheet_name='Key_information',
        usecols=[0, 3, 4, 5], index_col=[0])

cap_df = pd.read_excel('Fig_input_data/City_Cap.xlsx', index_col=[0])

area_data = area_df.to_numpy()
cap_data = cap_df.to_numpy()

population_df = pd.read_excel('Fig_input_data/City_info.xlsx', sheet_name=0,
        usecols=list(np.arange(8)))  # The last column 'City' is the index of the cities in the study

population_df.set_index('City', inplace=True)

scale_mapping = {
    "II\u578b\u5927\u57ce\u5e02": 0,
    "I\u578b\u5927\u57ce\u5e02": 1,
    "\u7279\u5927\u57ce\u5e02": 2,
    "\u8d85\u5927\u57ce\u5e02": 3
}

population_df["city_level"] = population_df["\u89c4\u6a21\u7b49\u7ea7"].map(scale_mapping)

file_path = 'Fig_input_data/City_statistic.xlsx'  # Path to the Excel file
building_volume_df = pd.read_excel(file_path, index_col=0)
total_building_volume = building_volume_df.iloc[:, 0].copy()
key_information = pd.read_excel(file_path, sheet_name=2, index_col=0)
total_building_area = key_information['Area_roof-0(km2)']
average_building_height = total_building_volume / total_building_area * 1000
average_building_height.name = 'average_building_height'

# ------------------- Read power curve data -------------------
# Read .npy files
file_path = '.'
ideal_facade_data_1_path='Fig_input_data/Power_facade_ideal_1.npy'
ideal_facade_data_1 = np.sum((np.load(ideal_facade_data_1_path)),axis=1)/1e6

real_facade_data_1_path='Fig_input_data/Power_facade_1.npy'
real_facade_data_1 = np.sum((np.load(real_facade_data_1_path)),axis=1)/1e6

roof_data_1_path = 'Fig_input_data/Power_roof_1.npy'
roof_data_1 = np.sum(np.load(roof_data_1_path),axis=1)/1e6

cities_hku_dest = pd.read_hdf('Fig_input_data/cities_hku_dest.hdf', key='cities_hku_dest')

import os

# Specify the file path
file_path = "Fig_input_data/City_adcode.xlsx"

# Ensure the file exists
if os.path.exists(file_path):
    # Read the Excel data
    city_adcode_df = pd.read_excel(file_path, engine='openpyxl', index_col=0)
    print("Data loaded successfully!")
    print(city_adcode_df.head())  # Display the first few rows of the data
else:
    print(f"The file {file_path} does not exist. Please check if the path is correct.")

result_folder = r'Fig_input_data'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_roof_ideal_1_sum_all_df.h5')

# If the HDF file already exists, read the data directly; otherwise, perform data processing and save the result
Power_roof_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Power_roof_ideal_1_sum_all_df

result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_facade_ideal_1_sum_all_df.h5')

Power_facade_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Power_facade_ideal_1_sum_all_df


result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_feas_wea_all_df.h5')

Grid_feas_wea_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

solar_df = Grid_feas_wea_all_df.groupby(level=0).mean()/1000  # kWh/m²

Grid_feas_wea_all_df

df_plot = area_df.iloc[:, :2].copy()  # km^2
roof_power = roof_data_1  # TWh
facade_power = ideal_facade_data_1  # TWh
df_plot['roof_power'] = roof_power
df_plot['facade_power'] = facade_power

df_plot = df_plot.merge(average_building_height, left_index=True, right_index=True)

df_plot['facade_roof_ratio_area'] = df_plot['Area_facade-0(km2)'] / df_plot['Area_roof-0(km2)']
df_plot['facade_roof_ratio_generation'] = df_plot['facade_power'] / df_plot['roof_power']

import Functions
Carbon_evaluation = np.zeros((102, 1))
Carbon_xy = np.zeros((102, 3))
main_case_path = os.path.join(str(MAIN_CASE_DIR), 'main_case_city_totals.csv')
main_case = pd.read_csv(main_case_path).set_index('City')
main_case = main_case.reindex(df_plot.index)
if main_case['fpv_carbon_Mt'].isna().any():
    missing_cities = main_case.index[main_case['fpv_carbon_Mt'].isna()].tolist()
    raise ValueError(f'Missing cities in main case: {missing_cities}')

df_plot['facade_carbon'] = main_case['fpv_carbon_Mt'].values
for i in range(len(df_plot)):
    Carbon_evaluation[i, 0] = df_plot['facade_carbon'].iloc[i]
    Carbon_xy[i, -1] = df_plot['facade_carbon'].iloc[i]
    Carbon_xy[i, :2] = cities_hku_dest.loc[i, ['HKU_LONGITUDE', 'HKU_LATITUDE']].values

df_plot = df_plot.merge(city_adcode_df, left_index=True, right_index=True, how='left')

df_plot['city_adcode'] = df_plot['city_adcode'].astype(int)

df_plot


df_plot_level = area_df.iloc[:, :2].copy()

# Merge population data and solar radiation data
df_plot_level = df_plot_level.merge(population_df, left_index=True, right_index=True)
df_plot_level = df_plot_level.merge(solar_df, left_index=True, right_index=True)

# Sort by urban population in ascending order within each city level
df_plot_level[['\u57ce\u533a\u4eba\u53e3', '\u5e38\u4f4f\u4eba\u53e3', '\u57ce\u9547\u5316\u7387']] = df_plot_level[['\u57ce\u533a\u4eba\u53e3', '\u5e38\u4f4f\u4eba\u53e3', '\u57ce\u9547\u5316\u7387']].astype(float)
# df_plot_level.sort_values(['city_level', '\u57ce\u533a\u4eba\u53e3'], ascending=True, inplace=True)
# # Add 'city_adcode' column
df_plot_level = df_plot_level.merge(city_adcode_df, left_index=True, right_index=True, how='left')
df_plot_level['city_adcode'] = df_plot_level['city_adcode'].astype(int)

import geopandas as gpd
import pyproj
file_path_taiwan = 'Fig_input_data/\u53f0\u6e7e\u77e2\u91cf\u5730\u56feshp\u6570\u636e/'
taiwan = gpd.read_file(file_path_taiwan + '\u53f0\u6e7e\u7701-\u5e02\u77e2\u91cfshp.shp').to_crs(epsg=4326)

proj_aeqd = (
    "+proj=aeqd "
    "+lat_0=35 "   # Central latitude
    "+lon_0=105 "  # Central longitude
    "+datum=WGS84 "
    "+units=m "
    "+no_defs "
)
aeqd_crs = pyproj.CRS.from_proj4(proj_aeqd)

taiwan_aeqd = taiwan.to_crs(aeqd_crs)


#%%
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import frykit.plot as fplt
import frykit.shp as fshp


def get_cn_city_table_compatible():
    try:
        return fshp.get_cn_city_table(data_source='tianditu')
    except TypeError:
        return fshp.get_cn_city_table()


def get_cn_city_compatible(city_adcode):
    try:
        return fshp.get_cn_city(city_adcode, data_source='tianditu')
    except TypeError:
        return fshp.get_cn_city(city_adcode)


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


def normalize_city_table(city_table):
    city_table = city_table.copy()
    if 'city_adcode' in city_table.columns:
        return city_table

    if city_table.index.name is not None:
        idx_name = str(city_table.index.name)
        normalized_idx_name = idx_name.lower().replace('_', '').replace('-', '')
        if normalized_idx_name in {'cityadcode', 'adcode', 'code'} or 'adcode' in normalized_idx_name:
            city_table = city_table.reset_index()
            if idx_name != 'city_adcode':
                city_table = city_table.rename(columns={idx_name: 'city_adcode'})
            return city_table

    normalized_columns = {
        str(col).lower().replace('_', '').replace('-', '').replace(' ', ''): col
        for col in city_table.columns
    }
    for key in [
        'cityadcode',
        'ctadcode',
        'adcode',
        'code',
        'citycode',
        '\u5e02\u4ee3\u7801',
        '\u57ce\u5e02\u4ee3\u7801',
        '\u884c\u653f\u533a\u5212\u4ee3\u7801',
        '\u884c\u653f\u4ee3\u7801',
    ]:
        if key in normalized_columns:
            return city_table.rename(columns={normalized_columns[key]: 'city_adcode'})

    raise KeyError(
        "Cannot find city_adcode in frykit city table. "
        f"Available columns: {list(city_table.columns)}; index name: {city_table.index.name}"
    )

# Parameter settings
col_plot = "facade_carbon"

cmap = plt.get_cmap('YlGn')
bad_color = "#FDFCFC"
cmap.set_bad(bad_color)

n_levels = df_plot_level['city_level'].nunique()
norm_level = plt.Normalize(vmin=-0.5, vmax=n_levels - 0.5)
colors_level = ["#D8D4D4", "#969393", "#5F5D5D", "#222121"]
cmap_popula = ListedColormap(colors_level)

lw_cn_map = lw * 0.1
lw_axis = lw * 0.5

# Data preparation
city_table = get_cn_city_table_compatible()  # Retrieve city boundary metadata
city_table = normalize_city_table(city_table)
city_table_with_values = city_table.merge(df_plot, left_on='city_adcode', right_on='city_adcode', how='left')  # Merge by city_adcode
city_adcode = city_table_with_values['city_adcode'].astype(int)
cities = get_cn_city_compatible(city_adcode)
data = city_table_with_values[col_plot]

city_table_with_values_level = city_table.merge(df_plot_level, left_on='city_adcode', right_on='city_adcode', how='left')  # Merge by city_adcode
# city_adcode_level = city_table_with_values_level['city_adcode'].astype(int)
# cities_level = fshp.get_cn_city(city_adcode_level, data_source='tianditu')
data_level = city_table_with_values_level['city_level']

vmin, vmax = np.floor(data.min() * 1) / 1, np.ceil(data.max() * 1) / 1
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Set map projection
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = fplt.PLATE_CARREE

# Set tick marks
xticks = np.arange(-180, 181, 10)
yticks = np.arange(-90, 91, 10)

font_options = {'size': fs}
plt.rc('font', **font_options)

fig = plt.figure(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.5)
main_ax = fig.add_subplot(projection=map_crs)

fplt.set_map_ticks(main_ax, (74, 136, 17, 55), xticks, yticks)
main_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.1, ls='--', color='gray')
main_ax.set_adjustable('datalim')
main_ax.tick_params(axis='x', labelsize=10)  
main_ax.tick_params(axis='y', labelsize=10)  
#main_ax.set_xticks([])
#main_ax.set_yticks([])
#main_ax.axis('off')

# Prepare inset map
mini_ax = fplt.add_mini_axes(main_ax, shrink=0.35)
mini_ax.spines[:].set_linewidth(lw_axis)
mini_ax.set_extent((105, 122, 2, 25), data_crs)
mini_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.1, ls='--', color='gray')


# Add map features
for ax in [main_ax, mini_ax]:
    fplt.add_cn_border(ax, lw=lw_cn_map, fc='none', zorder=-10)
    add_cn_line_compatible(ax, lw=lw_cn_map+1)

for ax in [main_ax, mini_ax]:
    add_geometries_compatible(
        ax, cities, array=data_level,
        cmap=cmap_popula, norm=norm_level,
        ec='black', lw=lw_cn_map,
    )
    taiwan_aeqd.plot(
        ax=ax, color='none', edgecolor='black',
        linewidth=lw_cn_map,
    )
    for ik in range(102):
        lon, lat = Carbon_xy[ik, 0], Carbon_xy[ik, 1]
        normal_car = (Carbon_xy[ik,2]-min(Carbon_xy[:,2]))/(max(Carbon_xy[:,2])-min(Carbon_xy[:,2]))
        ax.scatter(lon,lat,s=30, marker='o', color=cmap(normal_car),transform=data_crs)

main_ax.plot([Carbon_xy[3,0],Carbon_xy[3,0]+10],[Carbon_xy[3,1],Carbon_xy[3,1]],color='black', lw=lw_axis,linestyle = '--',transform=data_crs)
main_ax.text(Carbon_xy[3,0]+11, Carbon_xy[3,1]-1,'Beijing (SLC)\n   '+str(round(Carbon_xy[3,2],1))+'MT',fontsize = fs-15,color='black', transform=data_crs)

main_ax.plot([Carbon_xy[52,0],Carbon_xy[52,0]+6],[Carbon_xy[52,1],Carbon_xy[52,1]-4],color='black', lw=lw_axis,linestyle = '--',transform=data_crs)
main_ax.text(Carbon_xy[52,0]+7, Carbon_xy[52,1]-5,'Qingdao (VLC)\n    '+str(round(Carbon_xy[52,2],1))+'MT',fontsize = fs-15,color='black', transform=data_crs)

main_ax.plot([Carbon_xy[68,0],Carbon_xy[68,0]+9],[Carbon_xy[68,1],Carbon_xy[68,1]-3],color='black', lw=lw_axis,linestyle = '--',transform=data_crs)
main_ax.text(Carbon_xy[68,0]+10, Carbon_xy[68,1]-4,'Tianjin (SLC)\n    '+str(round(Carbon_xy[68,2],1))+'MT',fontsize = fs-15,color='black', transform=data_crs)

# main_ax.plot([Carbon_xy[95,0],Carbon_xy[95,0]+14],[Carbon_xy[95,1],Carbon_xy[95,1]-3],color='black', lw=lw_axis,linestyle = '--',transform=data_crs)
# main_ax.text(Carbon_xy[95,0]+14.5, Carbon_xy[95,1]-4,'Zhengzhou (VLC)\n      '+str(round(Carbon_xy[95,2],1))+'MT',fontsize = fs-15,color='black', transform=data_crs)



main_ax.text(-0.04,1.03,'b',transform=main_ax.transAxes, fontsize=fs, fontweight='bold', va='top', ha='right')

main_ax.set_title(
    'Nationwide carbon mitigation potential',
    y=0.93,
    fontsize=fs-8,
    weight='normal',
    fontweight='bold'
)
# Add legend
import matplotlib.patches as mpatches
labels_level = ['1-3 (LC-II)', '3-5 (LC-I)', '5-10 (VLC)', '>10 (SLC)']
patches = []
for color, label in zip(colors_level, labels_level):
    patch = mpatches.Patch(fc=color, ec='k',
            lw=lw_axis, label=label)
    patches.append(patch)
main_ax.legend(
    handles=patches[::-1],
    loc=(-0.0, -0.0),
    frameon=False,
    handleheight=0.7,      # Default is 0.7
    handlelength=1.5,      # Default is 2
    fontsize=fs-13,
    title='Population\n  (Million)',
    title_fontsize=fs-13,
    labelspacing=0.3,      # Vertical spacing between legend entries
    handletextpad=0.5,     # Space between handle and text
)


# --------------------- Create colorbar ---------------------
# Main colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
#cax = fig.add_axes([0.08 + 0.06, 0.12, 0.25, 0.03])  # Add space on the left for NaN block
cax = fig.add_axes([0.26, 0.06, 0.03, 0.14])
cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_ticks(np.linspace(vmin, vmax, 2))
cbar.ax.tick_params(labelsize=fs - 13)
cbar.outline.set_visible(True)
cbar.outline.set_linewidth(lw_axis)

# Align tick labels: left-aligned for min, right-aligned for max
tick_labels = cbar.ax.get_yticklabels()
tick_labels[0].set_verticalalignment('bottom')
tick_labels[-1].set_verticalalignment('top')

# Add NaN color block (gray)
# na_cax = fig.add_axes([0.08, 0.12, 0.045, 0.03])  # Position of mini colorbar
# na_cax.set_facecolor(bad_color)
# na_cax.set_xticks([])
# na_cax.set_yticks([])
# na_cax.spines[:].set_linewidth(lw_axis)
# na_cax.text(0.5, -1.3, 'N/A', ha='center', va='center', fontsize=fs - 9, transform=na_cax.transAxes)

# Add unit label
main_ax.text(
    0.19, 0.21,  # Position: just above the NaN block
    'Carbon mitigation\n   (Million ton)',                   # Annotation text
    transform=fig.transFigure,
    ha='left', va='bottom', fontsize=fs-13
)

# Adjust layout
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
#plt.tight_layout(rect=[0, 0, 0.9, 0.9])
fig.savefig(os.path.join(str(FIG_OUTPUT_DIR), 'Fig1b_left.pdf'),format='pdf',dpi=600,bbox_inches='tight')
plt.show()

#%%

path = str(ML_RESULTS_DIR) + os.sep
Statis_all = pd.read_excel(CITY_STATISTICS_FILE,sheet_name = 'Key_information', index_col=0)

City_SLC_list = []
indices = Statis_all.index[Statis_all.iloc[:,-1] == ['\u8d85\u5927\u57ce\u5e02','\u7279\u5927\u57ce\u5e02','I\u578b\u5927\u57ce\u5e02','II\u578b\u5927\u57ce\u5e02'][0]]
for index in indices:
    City_SLC_list.append(Statis_all.index.get_loc(index))

City_VLC_list = []
indices1 = Statis_all.index[Statis_all.iloc[:,-1] == ['\u8d85\u5927\u57ce\u5e02','\u7279\u5927\u57ce\u5e02','I\u578b\u5927\u57ce\u5e02','II\u578b\u5927\u57ce\u5e02'][1]]
for index in indices1:
    City_VLC_list.append(Statis_all.index.get_loc(index))

City_LCI_list = []
indices2 = Statis_all.index[Statis_all.iloc[:,-1] == ['\u8d85\u5927\u57ce\u5e02','\u7279\u5927\u57ce\u5e02','I\u578b\u5927\u57ce\u5e02','II\u578b\u5927\u57ce\u5e02'][2]]
for index in indices2:
    City_LCI_list.append(Statis_all.index.get_loc(index))

City_LCII_list = []
indices3 = Statis_all.index[Statis_all.iloc[:,-1] == ['\u8d85\u5927\u57ce\u5e02','\u7279\u5927\u57ce\u5e02','I\u578b\u5927\u57ce\u5e02','II\u578b\u5927\u57ce\u5e02'][3]]
for index in indices3:
    City_LCII_list.append(Statis_all.index.get_loc(index))

data_list = [Carbon_evaluation[City_SLC_list,0],Carbon_evaluation[City_VLC_list,0],Carbon_evaluation[City_LCI_list,0],Carbon_evaluation[City_LCII_list,0]]

fig = plt.figure(figsize=(8, 9))

plt.title("Carbon mitigation in different city types", fontsize=fs, y=0.94, fontweight='bold')

plt.ylim(0, 40)
plt.boxplot(data_list, widths=0.5, patch_artist=True, boxprops=dict(facecolor='mediumseagreen', alpha = 0.7), tick_labels=[""] * 4, showfliers = True)
plt.grid(axis="y", linestyle="--", alpha=1)
plt.xlabel("")
plt.ylabel("Million ton")

for i, data in enumerate(data_list):
    x = np.random.normal(i+1, 0.0, size=len(data))
    plt.scatter(x, data, color='grey', alpha=1, s=20)
    mean_val = np.mean(data)
    plt.scatter(i+1, mean_val, marker='*', s=200, color='blue', edgecolor='black', zorder=10)
    plt.text(i+1.1, mean_val, f'{mean_val:.2f}', ha='left', va='center', fontsize=fs-5,color = 'blue')

plt.xticks(ticks=np.arange(1,5), labels=['SLC','VLC','LC-I','LC-II'])
plt.tick_params(axis='y', labelsize=fs-5)
plt.tight_layout()

fig.savefig(os.path.join(str(FIG_OUTPUT_DIR), 'Fig1b_right.pdf'),format='pdf',dpi=600,bbox_inches='tight')
plt.show()


#%%
from scipy.io import savemat,loadmat

path = str(ML_RESULTS_DIR) + os.sep
city_path = str(CITY_FEATURE_DIR) + os.sep
path_type = str(ML_RESULTS_DIR / 'Power')
path_cap = str(ML_RESULTS_DIR / 'Capacity')
City_statistic = pd.read_excel(CITY_STATISTICS_FILE,sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel(CITY_STATISTICS_FILE,sheet_name = 'Key_information', index_col=0)

aggregate_cache_path = FIG_INPUT_DIR / "Fig1c_aggregate.npz"
if aggregate_cache_path.exists():
    cache = np.load(aggregate_cache_path)
    Total_Cap_total = cache["Total_Cap_total"]
    Total_Ele = cache["Total_Ele"]
    Total_Cost_total = cache["Total_Cost_total"]
    Total_price = cache["Total_price"]
    Total_price_true = cache["Total_price_true"]
    Total_Carbon = cache["Total_Carbon"]
    Total_Carbon_00 = cache["Total_Carbon_00"]
    Total_Carbon_00_ty = cache["Total_Carbon_00_ty"]
    Total_price_00 = cache["Total_price_00"]
    Total_price_00_true = cache["Total_price_00_true"]
    Total_area = cache["Total_area"]
else:
    Total_Cap_total = np.zeros((102,2,5))
    Total_Ele = np.zeros((102,5))
    Total_Cost_total = np.zeros((102,6,5))
    
    Total_price = np.zeros((102,2,5))
    Total_price_true = np.zeros((102,2,5))
    Total_Carbon = np.zeros((102,2,5))
    
    Total_Carbon_00 = np.zeros((102,2,5))
    Total_Carbon_00_ty = np.zeros((102,2,5))
    Total_price_00 = np.zeros((102,2,5))
    Total_price_00_true = np.zeros((102,2,5))
    
    Total_area = np.zeros((102,1))
    
    Statis_all = pd.read_excel(CITY_STATISTICS_FILE,sheet_name = 'Key_information', index_col=0)
    
    City_mega_list = []
    indices = Statis_all.index[Statis_all.iloc[:,-1] == ['\u8d85\u5927\u57ce\u5e02','\u7279\u5927\u57ce\u5e02','I\u578b\u5927\u57ce\u5e02','II\u578b\u5927\u57ce\u5e02'][0]]
    for index in indices:
        City_mega_list.append(Statis_all.index.get_loc(index))
    
    for cc in  range(102):  #City_mega_list:[3,15,56,59]:
        city_name = City_statistic.index[cc]
        print(city_name)
        C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
        C2 = path_cap+'/Cap_roof_'+city_name+'.npy'
    
        G_type = np.load(ML_RESULTS_DIR / 'Grid_type' / f'Grid_type_{city_name}.npy')
        Feas_read_sta = np.load(city_path+city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
        indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
        WWR = np.load(city_path+city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
        N_grid = G_type.shape[0]
        N_gg = np.where(G_type[:,0] != 888)[0]
        Total_area[cc,0] = np.sum(G_type[:,4])/1e6 #(km2)
    
        th_hh = 18
        th_std = [0,0]  #[np.percentile(G_type[N_gg,2], 50),np.percentile(G_type[N_gg,2], 50)]
        th_aa = [200,200]  # [np.percentile(G_type[N_gg,3], 50),np.percentile(G_type[N_gg,3], 50)]   #200 #120  # 55#120[120,120]  #  
        list_form = [list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
           list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]<th_aa[0]))[0]),\
           list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
           list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]<th_aa[0]))[0])]
    
        if cc in ABNORMAL_VOLUME_CITY_INDICES:
            data_path = str(PLANNING_RESULTS_DIR)+'/'+city_name+'_hybrid_n.mat'
        else:
            data_path = str(PLANNING_RESULTS_DIR)+'/'+city_name+'_hybrid.mat'
        Data = loadmat(data_path)
        R_Cap_r = Data['R_Cap_r']  #np.zeros((N_grid, 2, Y))
        R_Cap_f = Data['R_Cap_f']  #np.zeros((N_grid, 3, Y))
        R_Ele = Data['R_Ele']      #np.zeros((N_grid, Y))
        R_Pow_f = Data['R_Pow_f']  #np.zeros((N_grid,Y, D, T))
        R_Pow_r = Data['R_Pow_r']  #np.zeros((N_grid, 2, Y, D, T))
        R_Car = Data['R_Car']     #np.zeros((N_grid,2,Y))
        R_Cost = Data['R_Cost']    #np.zeros((N_grid,2,Y))
        R_Cost_true = Data['R_Cost_true']    #np.zeros((N_grid,2,Y))
    
        Total_Cap_total[cc,:,:] = np.sum(R_Cap_f[:,[0,2],:],axis = 0)
        Total_Carbon[cc,:,:] = 1e3*np.sum(R_Car[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
        Total_price[cc,:,:] = np.sum(R_Cost[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
        Total_price_true[cc,:,:] = np.sum(R_Cost_true[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
        Total_Carbon_00[cc,:,:] = np.sum(R_Car[:,:,:],axis = 0)
        Total_Carbon_00_ty[cc,:,:] = np.sum(R_Car[list_form[0],:,:],axis = 0)
        
        Total_price_00[cc,:,:] = np.sum(R_Cost[:,:,:],axis = 0)
        Total_price_00_true[cc,:,:] = np.sum(R_Cost_true[:,:,:],axis = 0)
    np.savez_compressed(
        aggregate_cache_path,
        Total_Cap_total=Total_Cap_total,
        Total_Ele=Total_Ele,
        Total_Cost_total=Total_Cost_total,
        Total_price=Total_price,
        Total_price_true=Total_price_true,
        Total_Carbon=Total_Carbon,
        Total_Carbon_00=Total_Carbon_00,
        Total_Carbon_00_ty=Total_Carbon_00_ty,
        Total_price_00=Total_price_00,
        Total_price_00_true=Total_price_00_true,
        Total_area=Total_area,
    )

s_font,s_title = 16,14

from matplotlib.patches import Patch
list_all_city_name,list_all_city_j,list_all_city_ii = [],[],[]
list_all_city_order = np.zeros((102,3))
list_all_city_order[:,0] = np.arange(102)
for i in range(4):
    indices = Statis_all.index[Statis_all.iloc[:,-1] == ['\u8d85\u5927\u57ce\u5e02','\u7279\u5927\u57ce\u5e02','I\u578b\u5927\u57ce\u5e02','II\u578b\u5927\u57ce\u5e02'][i]]
    for index in indices:
        i_loc = Statis_all.index.get_loc(index)
        list_all_city_order[i_loc,2] = 4-i
        list_all_city_order[i_loc,1] = Total_Cap_total[i_loc,0,-1]/1e6
sorted_indices = np.lexsort((-list_all_city_order[:, 1], -list_all_city_order[:, 2]))  # \u6ce8\u610f\u8d1f\u53f7\u5b9e\u73b0\u964d\u5e8f
data_sorted = list_all_city_order[sorted_indices]
list_all_city_j = list(data_sorted[:,0].astype(int))
list_all_city_ii = [list_all_city_j[:7],list_all_city_j[7:22],list_all_city_j[22:35],list_all_city_j[35:]]

aaCar_wo,aaCar_ww = [],[]
aaPri_woFPV,aaPri_wFPV = [],[]
bbCar_wo,bbCar_ww = [],[]
bbPri_woFPV,bbPri_wFPV = [],[]
for i in range(4):
    aaCar_wo.append(np.sum(Total_Carbon[list_all_city_ii[i],0,:],axis=1)/5)
    aaCar_ww.append(np.sum(Total_Carbon[list_all_city_ii[i],1,:],axis=1)/5)
    aaPri_woFPV.append(np.sum(Total_price_true[list_all_city_ii[i],0,:],axis=1)*1e10/5)
    aaPri_wFPV.append(np.sum(Total_price_true[list_all_city_ii[i],1,:],axis=1)*1e10/5)

#     bbCar_wo.append(np.sum(Total_Carbon_00[list_all_city_ii[i],0,:],axis=1))
#     bbCar_ww.append(np.sum(Total_Carbon_00[list_all_city_ii[i],1,:],axis=1)) #ton
#     bbPri_woFPV.append(np.sum(Total_price_00_true[list_all_city_ii[i],0,:],axis=1)*1e10) #CNY
#     bbPri_wFPV.append(np.sum(Total_price_00_true[list_all_city_ii[i],1,:],axis=1)*1e10)

# Debug totals from an older version used bb* arrays, which are not populated
# in the current figure workflow.
# print(np.sum(bbCar_wo[0])/1e6) #MT
# print(np.sum(bbPri_woFPV[0])/1e9) #BCNY
# print(np.sum(bbCar_ww[0])/1e6) #MT
# print(np.sum(bbPri_wFPV[0])/1e9) #BCNY



# print(np.argmax(bbCar_wo[0]-bbCar_ww[0])/1e6)
# print(np.argmax(bbPri_woFPV[0]-bbPri_wFPV[0])/1e9)
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig, ax = plt.subplots(1, 4, figsize=(20, 6))
for i in range(4):
    ax[i].scatter(aaPri_woFPV[i], aaCar_wo[i], color='deepskyblue', label='RS',s=80,alpha = 0.5)
    ax[i].scatter(aaPri_wFPV[i], aaCar_ww[i], color='seagreen', label='RS+F',s=80,alpha = 0.5)
    for ii in range(len(aaPri_woFPV[i])):
        ax[i].plot([aaPri_woFPV[i][ii], aaPri_wFPV[i][ii]], [aaCar_wo[i][ii], aaCar_ww[i][ii]], color='grey', linestyle='--', linewidth=0.5)
    ax[i].set_title(['SLC','VLC','LC-I','LC-II'][i], fontsize=s_font+10)
    ax[i].set_xlabel('Unit cost (CNY/kWh)', fontsize=s_font+8)
    ax[i].set_ylim(0.11, 0.22)
    ax[i].set_xlim(0.36, 0.65)
    
    if i == 0:
        ax[i].set_ylabel('Unit carbon (ton/MWh)', fontsize=s_font+8)
        ax[i].legend(fontsize=s_font+4)
    if i >= 1:
        ax[i].set_yticks([])


ax[0].text(0.25,0.23,'c',fontsize=fs+15, fontweight='bold', va='top', ha='right')


ax[0].text(aaPri_woFPV[0][0]-0.05, aaCar_wo[0][0]+0.005, 'Beijing' ,color='k')
ax[0].scatter(aaPri_woFPV[0][0], aaCar_wo[0][0], color='deepskyblue', s=150,alpha = 1)
ax[0].scatter(aaPri_wFPV[0][0], aaCar_ww[0][0], color='seagreen', s=150,alpha = 1)
ax[0].text(aaPri_woFPV[0][0]-0.05, aaCar_wo[0][0]-0.01, str(round(100*(aaCar_wo[0][0]-aaCar_ww[0][0])/aaCar_wo[0][0],1))+'%↓' ,color='k')
ax[0].hlines(y=aaCar_ww[0][0], xmin=aaPri_wFPV[0][0], xmax=aaPri_woFPV[0][0], color='grey', linestyle='--', linewidth=0.5)
ax[0].vlines(x = aaPri_woFPV[0][0], ymin=aaCar_ww[0][0], ymax=aaCar_wo[0][0], color='grey', linestyle='--', linewidth=0.5)


ax[1].text(aaPri_woFPV[1][0]-0.03, aaCar_wo[1][0]+0.005, 'Wuhan' ,color='k')
ax[1].scatter(aaPri_woFPV[1][0], aaCar_wo[1][0], color='deepskyblue', s=100,alpha = 1)
ax[1].scatter(aaPri_wFPV[1][0], aaCar_ww[1][0], color='seagreen', s=100,alpha = 1)
ax[1].text(aaPri_woFPV[1][0]-0.025, aaCar_wo[1][0]-0.02, str(round(100*(aaCar_wo[1][0]-aaCar_ww[1][0])/aaCar_wo[1][0],1))+'%↓' ,color='k')
ax[1].hlines(y=aaCar_ww[1][0], xmin=aaPri_wFPV[1][0], xmax=aaPri_woFPV[1][0], color='grey', linestyle='--', linewidth=0.5)
ax[1].vlines(x = aaPri_woFPV[1][0], ymin=aaCar_ww[1][0], ymax=aaCar_wo[1][0], color='grey', linestyle='--', linewidth=0.5)

plt.subplots_adjust(wspace=0.1)
plt.tight_layout()
fig.savefig(os.path.join(str(FIG_OUTPUT_DIR), 'Fig1c.pdf'), format='pdf', dpi=600, bbox_inches='tight')
plt.show()



