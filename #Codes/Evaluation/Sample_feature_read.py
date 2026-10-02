#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
from osgeo import gdal,ogr,gdalconst
import re
import fiona
from shapely.geometry import shape
from pyproj import Transformer
from scipy.spatial import KDTree
import scipy.stats as st
import Functions
import os
import time

City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', 'Hefei', 'Huhehaote', 'Jinan', \
             'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

day_list = ['0115','0215','0315','0415','0515','0615','0715','0815','0915','1015','1115','1215']
day_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
time_list = ['08','10','12','14','16','18']

cc = City_name[14]

print(cc)

df_read_roof = pd.read_excel(cc+'_simulation_results_Final.xlsx',sheet_name = 'Inst_Roof').iloc[:,1:].values
df_read_facade = pd.read_excel(cc+'_simulation_results_Final.xlsx',sheet_name = 'Inst_Facade_floor').iloc[:,1:].values
df_read_facade_pow = [pd.read_excel(cc+'_simulation_results_Final.xlsx',sheet_name = 'Eco_Facade_'+str(case)).iloc[:,1:].values for case in range(6)]
Shp_area_all_0_12 = pd.read_excel(cc+'_simulation_results_Final.xlsx',sheet_name = 'Shp_area').values
Shp_num_all = pd.read_excel(cc+'_simulation_results_Final.xlsx',sheet_name = 'Build_info').values
Shp_num_all = np.sum(Shp_num_all[:,4:],axis = 1)

sam_select = Functions.sele_sam(df_read_facade,Shp_area_all_0_12,Shp_num_all)

Feas_quanbu = np.zeros((len(sam_select)*72,1+21+3+8))  
for sam_i in tqdm(range(len(sam_select))):
    sam = sam_select[sam_i]
    DSM = gdal.Open(cc+'_'+str(sam)+'_DSM.tif')    
    DSM_gt = DSM.GetGeoTransform()
    DSM_xmin = DSM_gt[0]+33*3
    DSM_ymax = DSM_gt[3]-33*3
    DSM_xmax = DSM_xmin + 2001
    DSM_ymin = DSM_ymax -2001
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
    lat,lon = transformer.transform(0.5*(DSM_xmin+DSM_xmax), 0.5*(DSM_ymin+DSM_ymax))

    Shp_hei,Shp_hei_0,Shp_area,Shp_peri = [],[],[],[]
    Shp_area_roof = []
    Shp_cen = []
    with fiona.open(str(sam)+'.shp') as SHP:
        crs = SHP.crs
        for feat in SHP:
            geom = shape(feat["geometry"])
            x,y = np.array(geom.centroid.coords)[0,0],np.array(geom.centroid.coords)[0,1]
            if x >= DSM_xmin and x <= DSM_xmax and y >= DSM_ymin and y <= DSM_ymax:
                if feat["properties"]["height"] >= 12:
                    Shp_hei.append(feat["properties"]["height"])
                    bbox = geom.bounds
                    Shp_area.append(geom.area)
                    Shp_peri.append(geom.length)
                    Shp_cen.append([x,y])
                else:
                    Shp_hei_0.append(feat["properties"]["height"])
                if feat["properties"]["height"] >= 4:
                    Shp_area_roof.append(geom.area)
    
    Shp_cen = np.array(Shp_cen)
    
    sam_area = (DSM_xmax-DSM_xmin)*(DSM_ymax-DSM_ymin)
    
    #Urban density (m3/m2)
    Fea_density = sum(Shp_hei[i]*Shp_area[i] for i in range(len(Shp_hei)))/sam_area
    #Site coverage (%)
    Fea_coverage = 100*sum(Shp_area)/sam_area
    #Mean building height (m)
    Fea_Mhei = sum(Shp_hei[i]*Shp_area[i] for i in range(len(Shp_hei)))/sum(Shp_area)
    #SD height (m)
    Fea_SDhei = np.std(np.array(Shp_hei))
    #SD area (m2)
    Fea_SDarea = np.std(np.array(Shp_area))
    #Fea_Direct_std = [np.std(DSM_ext[i]) for i in range(len(DSM_ext))]
    #Fea_Direct = sum(Fea_Direct_std)/len(Fea_Direct_std)
    #Complexity (m2/m2)
    Fea_complexity = sum(Shp_hei[i]*Shp_peri[i] for i in range(len(Shp_hei)))/sam_area
    #Compactness (m2/m3)
    Fea_compact = sum(Shp_hei[i]*Shp_peri[i]+Shp_area[i]  for i in range(len(Shp_hei))) / sum(Shp_hei[i]*Shp_area[i]  for i in range(len(Shp_hei)))
    #Number of building volumes (m3)
    Fea_number = sum(Shp_hei[i]*Shp_area[i] for i in range(len(Shp_hei)))/len(Shp_hei)
    #Mean outdoor distance (m)
    Fea_outdoor = []
    tree = KDTree(Shp_cen)
    for i in range(len(Shp_cen)):
        dist, idx = tree.query(Shp_cen[i], k=2)
        nn_coord = Shp_cen[idx[1]]
        Fea_outdoor.append(dist[1])
    Fea_mean_outdoor = sum(Fea_outdoor)/len(Fea_outdoor)
    Fea_12_ratio = len(Shp_hei)/(len(Shp_hei)+len(Shp_hei_0))
    if all(x == Shp_hei[0] for x in Shp_hei):
        Fea_skew = 0
    else:
        Fea_skew = st.skew(Shp_hei)    
    lands = pd.read_excel('Landuse.xlsx').to_numpy()    
    x_min, x_max = lon-0.5, lon+0.5
    y_min, y_max = lat-0.5, lat+0.5
    Landzone_i = lands[(lands[:, 0] >= x_min) & (lands[:, 0] <= x_max) & (lands[:, 1] >= y_min) & (lands[:, 1] <= y_max)]
    Fea_WWR,_ = Functions.WWR_read(lon,lat,cc,Landzone_i)
    Fea_city = [lon, lat, Fea_density,Fea_coverage,Fea_Mhei,Fea_SDhei,Fea_SDarea,Fea_complexity,Fea_compact,Fea_number,Fea_mean_outdoor,Fea_12_ratio,Fea_skew,Fea_WWR]
    Fea_info = [len(Shp_hei),sum(Shp_area_roof),sum(Shp_hei[i]*Shp_peri[i] for i in range(len(Shp_hei)))]
    Veg = gdal.Open(cc+'/'+cc+'_'+str(sam)+'/'+cc+'_'+str(sam)+'_canopy_duiqi.tif')
    Veg_band = Veg.GetRasterBand(1).ReadAsArray()
    Veg_band = np.nan_to_num(Veg_band)
    for i in range(Veg_band.shape[0]):
        for j in range(Veg_band.shape[1]):
            if Veg_band[i,j] <= 1 or Veg_band[i,j] >= 50:
                Veg_band[i,j] = 0
    Veg_band_act = Veg_band[33:-33,33:-33].reshape(-1)
    Veg_band_act = Veg_band_act[Veg_band_act != 0]
    if len(Veg_band_act) == 0:
        Fea_o_mean = 0
        Fea_o_std = 0
    else:
        Fea_o_mean = sum(Veg_band_act)/len(Veg_band_act)
        Fea_o_std = np.std(Veg_band_act)
    Fea_others = [Fea_o_mean,Fea_o_std]
    
    for j_day in range(len(day_list)):
        days = float(sum(day_days[:j_day+1])+15)
        for j_time in range(len(time_list)):
            wea_path = cc+'/'+cc+'_'+str(sam)+'/'+cc+'_'+str(sam)+'_'+day_list[j_day]+'_'+time_list[j_time]+\
                                  '/'+cc+'_'+str(sam)+'_weather_TMY_'+day_list[j_day]+'_'+time_list[j_time]+'.txt'
            if not os.path.exists(wea_path):
                Fea_time = [days,float(time_list[j_time]),0,0,0]
            else:
                Weather = pd.read_csv(wea_path)
                Wea_str = re.findall(r'\d+\.?\d*', Weather.iloc[0,0])
                Fea_time = [days,float(time_list[j_time]),float(Wea_str[-10]),float(Wea_str[-3]),float(Wea_str[-2])]
            Fea_all = [sam] + Fea_city + Fea_others + Fea_time   #1;14+2+5=21
            Fea_label = [df_read_roof[j_day*6+j_time,sam-1], df_read_facade[j_day*6+j_time,sam-1]] + [df_read_facade_pow[case_i][j_day*6+j_time,sam-1] for case_i in range(6)]
            Feas_quanbu[sam_i*72+j_day*6+j_time,:] = Fea_all+Fea_info+Fea_label


np.save(cc+'_ML_features.npy', Feas_quanbu)
