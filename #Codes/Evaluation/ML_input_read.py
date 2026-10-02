
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from pyproj import Transformer
from scipy.spatial import KDTree
import scipy.stats as st
from osgeo import gdal, gdalconst, ogr, osr
import os
from shapely.geometry import Polygon, shape, box
import geopandas as gpd
import time
from scipy.spatial.distance import cdist

def WWR_read(lon, lat, city_name, Landzone_i):
    a = np.array([lon, lat]).reshape(1, -1)
    Climate = pd.read_excel('Climate_WWR.xlsx', sheet_name='Climate')
    climate_zone_row = Climate[Climate.eq(city_name)].any(axis=1).to_numpy().nonzero()[0][0]
    climate_zone = Climate.iloc[climate_zone_row, -1]
    
    distances = cdist(a, Landzone_i[:, :2])
    # Find the index of the nearest coordinate
    idx = np.argmin(distances)
    if Landzone_i[idx, -2] == 4 or Landzone_i[idx, -2] == 5:  # Transport/Public
        WWR = 0.5
    elif Landzone_i[idx, -2] == 3:  # Industrial
        WWR = 0.4
    elif Landzone_i[idx, -2] == 2:  # Commercial Office
        WWR = 0.2
    elif Landzone_i[idx, -2] == 1:  # Industrial
        if climate_zone == 'A':
            WWR = 0.25
        elif climate_zone == 'B':
            WWR = 0.30
        elif climate_zone == 'C':
            WWR = 0.35
        elif climate_zone == 'D':
            WWR = 0.30
        elif climate_zone == 'E':
            WWR = 0.35
    return WWR, Landzone_i[idx, -1]

def Wea_read(name, lat, lon):
    folder_path = name
    Location = []
    Location_path = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            numbers = re.findall(r'\d+.\d+', dir_name)
            numbers = [float(num) for num in numbers]
            Location.append(numbers)
            Location_path.append(os.path.join(root, dir_name))
    Location = np.array(Location)
    
    if Location.size == 0 or Location.shape[0] == 1:
        return np.zeros((1, 365*15*5))
    
    distances = cdist(np.array([lat, lon]).reshape(1, -1), Location)

    # Find the index of the nearest coordinate
    idx = np.argmin(distances)
    Location_path_i = Location_path[idx]
    for file_name in os.listdir(Location_path_i):
        if '_TMY' in file_name and file_name.endswith('.txt'):
            file_path = os.path.join(Location_path_i, file_name)
            content = pd.read_csv(file_path)
    Fea_wea = np.zeros((1, 365*15*5))
    for day in range(365):
        for timeslot in range(15): # range(6,21):6am-20pm
            ii_time = day*15 + timeslot
            Wea_str = re.findall(r'\d+\.?\d*', content.iloc[day*24 + timeslot+6, 0])
            Fea_wea[0, ii_time*5:(ii_time+1)*5] = [day+1, timeslot+6, float(Wea_str[-10]), float(Wea_str[-3]), float(Wea_str[-2])]
    return Fea_wea

def clip_shp(xmin, ymin, xmax, ymax, gdf):
    # Ensure the coordinate system of the input shapefile is EPSG:4326
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)
    
    # Define the clipping area
    clip_area = gpd.GeoDataFrame(geometry=[Polygon([(xmin, ymin), 
                                                    (xmax, ymin), 
                                                    (xmax, ymax), 
                                                    (xmin, ymax), 
                                                    (xmin, ymin)])], 
                                 crs='EPSG:4326')
    
    # Perform the clipping operation
    clipped = gpd.overlay(gdf, clip_area, how='intersection')
    clipped = clipped[clipped.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
    # Ensure the coordinate system of the clipping result is EPSG:4326
    clipped = clipped.set_crs(epsg=4326, allow_override=True)
    
    # Convert the clipping result to EPSG:4547
    clipped = clipped.to_crs(epsg=4547)
    
    # Add debugging information
    print(f"Clip area bounds: {clip_area.total_bounds}")
    print(f"Clipped area bounds: {clipped.total_bounds}")
    print(f"Number of features in clipped area: {len(clipped)}")
    
    if clipped.empty:
        return None
    return clipped

path_in = 'simulation'
City_name = pd.read_excel('Climate_WWR.xlsx', sheet_name='Climate').iloc[:, 1].tolist()

name = "Beijing"

# "Beijing" can be replaced with "Wulumuqi" or "Beijing_major" for other cities.

time1 = time.time()
input_tiff = '{}_DEM.tif'.format(name)
input_tiff_2 = '{}_canopy.tif'.format(name)
input_shp = '{}.shp'.format(name)
gdf = gpd.read_file(input_shp)

gdal.UseExceptions()
input_ds = gdal.Open(input_tiff, gdalconst.GA_ReadOnly)
geotransform = input_ds.GetGeoTransform()
pixel_width = geotransform[1]
pixel_height = geotransform[5]

input_ds_2 = gdal.Open(input_tiff_2, gdalconst.GA_ReadOnly)
geotransform_2 = input_ds_2.GetGeoTransform()
pixel_width_2 = geotransform_2[1]
pixel_height_2 = geotransform_2[5]

width = input_ds.RasterXSize
height = input_ds.RasterYSize
width_2 = input_ds_2.RasterXSize
height_2 = input_ds_2.RasterYSize
transformer = Transformer.from_crs("EPSG:4326", "EPSG:4547", always_xy=True)
itransformer = Transformer.from_crs("EPSG:4547", "EPSG:4326", always_xy=True)

min_x, max_y = geotransform[0], geotransform[3]
max_x, min_y = geotransform[0] + width * geotransform[1], geotransform[3] + height * geotransform[5]
min_x_2, max_y_2 = geotransform_2[0], geotransform_2[3]
max_x_2,min_y_2 = geotransform_2[0] + width_2 * geotransform_2[1], geotransform_2[3] + height_2 * geotransform_2[5]

WIDTH = min(max_x,max_x_2) - max(min_x,min_x_2)
HEIGHT = min(max_y,max_y_2) - max(min_y,min_y_2)

a1, a2 = transformer.transform(geotransform[0], geotransform[3])
if name == 'Wulumuqi':
    b1, _ = itransformer.transform(a1+2200, a2)
    _, b2 = itransformer.transform(a1, a2-2200)
else:
    b1, _ = itransformer.transform(a1+2000, a2)
    _, b2 = itransformer.transform(a1, a2-2000)
tile_size_x = b1 - min_x  # Here determines the longitude change corresponding to 2km
tile_size_y = max_y - b2

time2 = time.time()

num_tiles_x = WIDTH // tile_size_x
num_tiles_y = HEIGHT // tile_size_y
counter = 0
out_counter = 0
test_count = 0
offset_dict = {}

# Start and end points, taking the maximum range value here, because subsequent steps will check whether the point is within the Beijing map, so it will not cause an overly large fitting result
start_x = min(min_x,min_x_2)
end_x = max(max_x, max_x_2)
start_y = min(min_y, min_y_2)
end_y = max(max_y, max_y_2)

# Calculate the tile width and height
tile_size_x_tif_1 = int(width*tile_size_x/WIDTH)
tile_size_y_tif_1 = int(height*tile_size_y/HEIGHT)
tile_size_x_tif_2 = int(width_2*tile_size_x/WIDTH)
tile_size_y_tif_2 = int(height_2*tile_size_y/HEIGHT)

for i in tqdm(range(int(num_tiles_x))):
    for j in range(int(num_tiles_y)):
        # Calculate the offset of the current tile
        offset_x_jw, offset_y_jw = start_x + i * tile_size_x, start_y + j * tile_size_y

        offset_x_tif_1 = int((offset_x_jw-start_x)/geotransform[1])
        offset_y_tif_1 = int(-(offset_y_jw-start_y)/geotransform[5])
        offset_x_tif_2 = int((offset_x_jw-start_x)/geotransform_2[1])
        offset_y_tif_2 = int(-(offset_y_jw-start_y)/geotransform_2[5])
        if (0 <= offset_x_tif_1 < width and 0 <= offset_y_tif_1 < height and
            0 <= offset_x_tif_2 < width_2 and 0 <= offset_y_tif_2 < height_2 and
            offset_x_tif_1 + tile_size_x_tif_1 <= width and offset_y_tif_1 + tile_size_y_tif_1 <= height and
            offset_x_tif_2 + tile_size_x_tif_2 <= width_2 and offset_y_tif_2 + tile_size_y_tif_2 <= height_2):
            # Since the current coordinate transformation is not linear, points on the boundary may move to the right in the metric coordinate system, so negative values that cannot be converted back need to be discarded.
            tile_data_tif_1 = input_ds.ReadAsArray(offset_x_tif_1 , offset_y_tif_1 , tile_size_x_tif_1, tile_size_y_tif_1)
            tile_data_tif_2 = input_ds_2.ReadAsArray(offset_x_tif_2 , offset_y_tif_2 , tile_size_x_tif_2, tile_size_y_tif_2)

            # Exclude boundary points
            zero_ratio = (tile_data_tif_1.size - np.count_nonzero(tile_data_tif_1)) / tile_data_tif_1.size
            if zero_ratio > 0.5: # Consider all points with data, that is, consider the boundary, which will result in an area being too large (taking Beijing as an example, if zero_ratio == 1, then 17000+ square kilometers are considered; if zero_ratio > 0.5, then 16300 square kilometers are considered)
                continue
            offset_dict[counter] = (offset_x_tif_1, offset_y_tif_1, offset_x_tif_2, offset_y_tif_2)

            # Update the counter
            counter += 1

total_number = counter

# Get keys and values of dictionary
keys = list(offset_dict.keys())
values = list(offset_dict.values())

# Record coordinates
result = []
for a, b, c, d in offset_dict.values():
    x = geotransform[0] + a * geotransform[1]
    u = geotransform[3] + (b + tile_size_y_tif_1) * geotransform[5]
    z = geotransform[0] + (a + tile_size_x_tif_1) * geotransform[1]
    y = geotransform[3] + b * geotransform[5]
    m = geotransform_2[0] + c * geotransform_2[1]
    n = geotransform_2[3] + (d + tile_size_y_tif_2) * geotransform_2[5]
    o = geotransform_2[0] + (c + tile_size_x_tif_2) * geotransform_2[1]
    p = geotransform_2[3] + d * geotransform_2[5]
    result.append([x, u, z, y, m, n, o, p])
clip_location = np.array(result)
time3 = time.time()
#%
Feas_static = np.zeros((1, 16+1+3*4)) # 16+1 static features (the one after WWR is a more specific land use) + 3*4 (3 types of buildings (0, 4, 12), 4 indicators (number, roof area, volume, surface area))
Feas_dyna = np.zeros((1, 5*365*15)) # 5 dynamic features, a total of 365*15 moments (6-20), each moment has 5 dimensional features!!!
cut_number = 0

for sam in tqdm(range(len(clip_location))):
    #1. Clip shapefile
    SHP = clip_shp(clip_location[sam][0], clip_location[sam][3], clip_location[sam][2], clip_location[sam][1], gdf)
    if SHP is None:
        print('No data in this area')
        continue
    else:
        cut_number += 1
        print('Processing the {}th tile'.format(cut_number))

    #2. Clip tiff file
    # Create a new spatial reference system
    min_lon, min_lat, max_lon, max_lat, _, _, _, _ = clip_location[sam]
    # Convert latitude and longitude coordinates to EPSG:3857 coordinates
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)

    #3. Create an in-memory output file - DEM's sam-th slice data
    output_tiff_new = '/vsimem/temp_dem.TIF'
    output_ds = gdal.Warp(output_tiff_new, input_ds, format='GTiff', outputBounds=[min_x, min_y, max_x, max_y], dstSRS='EPSG:4547')
    output_ds = None

    DEM = gdal.Open(output_tiff_new, gdalconst.GA_ReadOnly)

    DEM_gt = DEM.GetGeoTransform()

    # Extract features for the actual study area below
    lon, lat = 0.5 * (min_lon + max_lon), 0.5 * (min_lat + max_lat)
    
    if lon < 70:
        print('Longitude and latitude are incorrect!!! Reversed!!!')

    # Read shapefile
    Shp_hei, Shp_hei_4, Shp_hei_0, Shp_area, Shp_area_4, Shp_area_0, Shp_peri, Shp_peri_4, Shp_peri_0 = [], [], [], [], [], [], [], [], []
    Shp_cen = []

    # Traverse each feature (building)
    for idx, feat in SHP.iterrows():
        # Get the geometry of the building
        geom = feat["geometry"]
        if geom is None:
            print(f"Feature ID: {idx} has a None geometry")
            continue
        # Get coordinates of geometry centroid
        centroid = geom.centroid
        x_, y_ = centroid.x, centroid.y   # Already in 3857 -- metric system

        # Check if coordinates are within the DEM range
        if min_x <= x_ <= max_x and min_y <= y_ <= max_y:
            if feat["Height"] >= 12:
                Shp_hei.append(feat["Height"])
                Shp_area.append(geom.area)
                Shp_peri.append(geom.length)
                Shp_cen.append([x_, y_])
            if feat["Height"] >= 4:
                Shp_area_4.append(geom.area)
                Shp_hei_4.append(feat["Height"])
                Shp_peri_4.append(geom.length)
            if feat["Height"] >= 0:
                Shp_area_0.append(geom.area)
                Shp_hei_0.append(feat["Height"])
                Shp_peri_0.append(geom.length)
                
    sam_area = (max_x - min_x) * (max_y - min_y)  # 2000 * 2000
    
    if len(Shp_hei_0) <= 0:  # Filter out samples with few buildings
        DEM = None
        print('No enough buildings')
        continue    
    elif len(Shp_hei_0) > 0 and len(Shp_hei) <= 2:
        # WWR
        lands = pd.read_excel(path_in + '/Landuse.xlsx').to_numpy()    
        x_min, x_max = lon - 1.0, lon + 1.0
        y_min, y_max = lat - 1.0, lat + 1.0
        Landzone_i = lands[(lands[:, 0] >= x_min) & (lands[:, 0] <= x_max) & (lands[:, 1] >= y_min) & (lands[:, 1] <= y_max)]
        Fea_WWR, Fea_WWR_detail = WWR_read(lon, lat, name, Landzone_i)
        Fea_city = [lon, lat, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Fea_WWR, Fea_WWR_detail]
        
    else:
        Shp_cen = np.array(Shp_cen)
        # Urban density (m3/m2)
        Fea_density = sum(Shp_hei[i] * Shp_area[i] for i in range(len(Shp_hei))) / sam_area
        # Site coverage (%)
        Fea_coverage = 100 * sum(Shp_area) / sam_area
        # Mean building height (m)
        Fea_Mhei = sum(Shp_hei[i] * Shp_area[i] for i in range(len(Shp_hei))) / sum(Shp_area)
        # SD height (m)
        Fea_SDhei = np.std(np.array(Shp_hei))
        # SD area (m2)
        Fea_SDarea = np.std(np.array(Shp_area))
        # Direction: finally defined as the DEM height value for 8 directions, and then the STD of each direction, and finally the average is taken.
        # Fea_Direct_std = [np.std(DEM_ext[i]) for i in range(len(DEM_ext))]
        # Fea_Direct = sum(Fea_Direct_std) / len(Fea_Direct_std)
        # Complexity (m2/m2)
        Fea_complexity = sum(Shp_hei[i] * Shp_peri[i] for i in range(len(Shp_hei))) / sam_area
        # Compactness (m2/m3)
        Fea_compact = sum(Shp_hei[i] * Shp_peri[i] + Shp_area[i] for i in range(len(Shp_hei))) / sum(Shp_hei[i] * Shp_area[i] for i in range(len(Shp_hei)))
        # Number of building volumes (m3)
        Fea_number = sum(Shp_hei[i] * Shp_area[i] for i in range(len(Shp_hei))) / len(Shp_hei)
        # Mean outdoor distance (m)
        Fea_outdoor = []
        tree = KDTree(Shp_cen)
        for i in range(len(Shp_cen)):
            # Find nearest neighbor coordinates
            dist, idx_nn = tree.query(Shp_cen[i], k=2)
            nn_coord = Shp_cen[idx_nn[1]]
            Fea_outdoor.append(dist[1])
        Fea_mean_outdoor = sum(Fea_outdoor) / len(Fea_outdoor)
        # 12m ratio of all buildings
        Fea_12_ratio = len(Shp_hei) / (len(Shp_hei) + len(Shp_hei_0))
        # Skewness coefficient: median and mean of height
        if all(x == Shp_hei[0] for x in Shp_hei):
            Fea_skew = 0
        else:
            Fea_skew = st.skew(Shp_hei)
        # WWR
        lands = pd.read_excel(path_in + '/Landuse.xlsx').to_numpy()    
        x_min, x_max = lon - 1.0, lon + 1.0
        y_min, y_max = lat - 1.0, lat + 1.0
        Landzone_i = lands[(lands[:, 0] >= x_min) & (lands[:, 0] <= x_max) & (lands[:, 1] >= y_min) & (lands[:, 1] <= y_max)]
        Fea_WWR, Fea_WWR_detail = WWR_read(lon, lat, name, Landzone_i)
        # Summarize
        Fea_city = [lon, lat, Fea_density, Fea_coverage, Fea_Mhei, Fea_SDhei, Fea_SDarea, Fea_complexity, Fea_compact, Fea_number, Fea_mean_outdoor, Fea_12_ratio, Fea_skew, Fea_WWR, Fea_WWR_detail] 
    #4. Create an in-memory output file - canopy's sam-th slice data
    output_tiff_new_2 = '/vsimem/temp_canopy.TIF'
    output_ds_2 = gdal.Warp(output_tiff_new_2, input_ds_2, format='GTiff', outputBounds=[min_x, min_y, max_x, max_y], dstSRS='EPSG:4547')
    output_ds_2 = None

    VEG = gdal.Open(output_tiff_new_2, gdalconst.GA_ReadOnly)
    Veg_band = VEG.GetRasterBand(1).ReadAsArray()
    Veg_band = np.nan_to_num(Veg_band)
    for i in range(Veg_band.shape[0]):
        for j in range(Veg_band.shape[1]):
            if Veg_band[i, j] <= 1 or Veg_band[i, j] >= 50:
                Veg_band[i, j] = 0

    Veg_band_act = Veg_band.reshape(-1)
    Veg_band_act = Veg_band_act[Veg_band_act != 0]
    if len(Veg_band_act) == 0:
        Fea_o_mean = 0
        Fea_o_std = 0
    else:
        Fea_o_mean = sum(Veg_band_act)/len(Veg_band_act)
        Fea_o_std = np.std(Veg_band_act)
    Fea_others = [Fea_o_mean,Fea_o_std]

    Fea_info = [len(Shp_hei_0),len(Shp_hei_4),len(Shp_hei),sum(Shp_area_0),sum(Shp_area_4),sum(Shp_area),\
                sum(Shp_hei_0[i]*Shp_area_0[i] for i in range(len(Shp_hei_0))),sum(Shp_hei_4[i]*Shp_area_4[i] for i in range(len(Shp_hei_4))),sum(Shp_hei[i]*Shp_area[i] for i in range(len(Shp_hei))),\
                sum(Shp_hei_0[i]*Shp_peri_0[i] for i in range(len(Shp_hei_0))),sum(Shp_hei_4[i]*Shp_peri_4[i] for i in range(len(Shp_hei_4))),sum(Shp_hei[i]*Shp_peri[i] for i in range(len(Shp_hei)))]
    Fea_huiz = np.array(Fea_city + Fea_others + Fea_info).reshape(1, Feas_static.shape[1])
    Feas_static = np.vstack((Feas_static,Fea_huiz))

    Fea_wea = Wea_read(name, lat, lon)
    Feas_dyna = np.vstack((Feas_dyna,Fea_wea))
    
    VEG = None
    DEM = None

input_ds = None
input_ds_2 = None

Feas_static = np.delete(Feas_static, 0, axis=0)
Feas_dyna = np.delete(Feas_dyna, 0, axis=0)
Feas_ALL = np.hstack((Feas_static,Feas_dyna))
time4 = time.time()
np.save(path_in+'/'+name+'_ALL_Featuers_major_re.npy', Feas_ALL)

