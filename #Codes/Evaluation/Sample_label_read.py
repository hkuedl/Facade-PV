#%%
import pandas as pd
import numpy as np
import re
from osgeo import gdal, osr, ogr
from tqdm import tqdm
import fiona
from shapely.geometry import shape
from scipy.spatial.distance import cdist
import os
import Functions
import time
import mmap

path_save = 'ALL_sample_results'
City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', \
             'Hefei', 'Huhehaote', 'Jinan', 'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', \
                 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

day_list = ['0115','0215','0315','0415','0515','0615','0715','0815','0915','1015','1115','1215']
time_list = ['08','10','12','14','16','18']


cc = City_name[14]  #0,30
print(cc)

TOU_time,TOU_price_indu,TOU_price_resi,TOU_price_coal = Functions.TOU_period(cc)  
for sam in range(1,65): 
    print(sam)
    if not os.path.exists(cc+'/'+cc+'_'+str(sam)):
        continue
    if sam == 1:

        df_read_roof_0 = pd.read_excel(cc+'/'+cc+'_simulation_results.xlsx',sheet_name = 'Roof')
        df_read_facade_12_floor = pd.read_excel(cc+'/'+cc+'_simulation_results.xlsx',sheet_name = 'Facade')

        df_read_facade_LCOE = [np.zeros((3*64,10000),dtype = float) for i in range(6)]
        df_read_facade_area = np.zeros((64,6)) 
        df_read_facade_capacity = np.zeros((64,6)) 
        df_read_facade_ele = []
        for i in range(6):
            df_read_facade_ele.append(pd.read_excel(cc+'/'+cc+'_simulation_results.xlsx',sheet_name = 'Facade'))

        Build_info = np.zeros((64,500))
        Shp_area_all_0_12 = pd.DataFrame(np.zeros((64,3)),columns=['Roof', 'Facade_floor_shp', 'Facade_floor_txt'])
    else:
        df_read_facade_12_floor,df_read_roof_0,Build_info,Shp_area_all_0_12,df_read_facade_area,df_read_facade_capacity,df_read_facade_ele,df_read_facade_LCOE = \
            Functions.read(0,path_save,cc)  
    
    Shp_area_all_0_12 = Shp_area_all_0_12.astype(float)
    for xx in range(6):
        df_read_facade_LCOE[xx]=df_read_facade_LCOE[xx].astype(float)
    
    Wall_path = cc+'/'+cc+'_'+str(sam)+'/'

    Wall_hei_threshold = 4
    Wall_hei_threshold_n = 12
    ds_shp = ogr.Open(Wall_path+str(sam)+'.shp')
    if ds_shp == None:
        continue
    df_read_sam_rad = np.zeros((72,100000))   

    Wall_hei = gdal.Open(Wall_path+cc+'_'+str(sam)+'_wall_hei.tif')
    band = Wall_hei.GetRasterBand(1)
    data_hei = band.ReadAsArray()
    s_t = time.time()
    Wall_data = []
    if np.max(data_hei) > 600:
        with open(Wall_path+cc+'_'+str(sam)+'_'+day_list[0]+'_'+time_list[0]+'/Energyyearwall.txt', 'r') as file:
            with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmapped_file:
                for line in iter(mmapped_file.readline, b""):
                    line = line.decode('utf-8')
                    Wall_data.append(line[:303])
        Wall_data = Wall_data[1:]
    else:
        with open(Wall_path+cc+'_'+str(sam)+'_'+day_list[0]+'_'+time_list[0]+'/Energyyearwall.txt', 'r') as file:
            with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmapped_file:
                for line in iter(mmapped_file.readline, b""):
                    line = line.decode('utf-8')
                    Wall_data.append(line)
        Wall_data = Wall_data[1:]
    e_t = time.time()
    print(e_t-s_t)
    
    wall_index = np.zeros((len(Wall_data),2))
    for i in range(len(Wall_data)):
        i_list = Wall_data[i]
        number_str = re.findall(r'\d+\.?\d*', i_list)
        number_value = [float(ii) for ii in number_str]
        wall_index[i,:] = [int(number_value[0]),int(number_value[1])]
    polygons = []
    area_percentage1 = []
    area_percentage1_12 = []

    ds_tif = gdal.Open(Wall_path+cc+'_'+str(sam)+'_'+day_list[0]+'_'+time_list[0]+'/Energyyearroof.tif')
    transform_tif = ds_tif.GetGeoTransform()
    transform_shp = ds_shp.GetLayer().GetSpatialRef()
    for i in range(33,ds_tif.RasterYSize-33):
        for j in range(33,ds_tif.RasterXSize-33):

            x = transform_tif[0] + j * transform_tif[1] + 1.5
            y = transform_tif[3] + i * transform_tif[5] - 1.5

            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x - 1.5, y - 1.5) 
            ring.AddPoint(x + 1.5, y - 1.5)
            ring.AddPoint(x + 1.5, y + 1.5) 
            ring.AddPoint(x - 1.5, y + 1.5) 
            ring.AddPoint(x - 1.5, y - 1.5) 
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            poly.TransformTo(transform_shp)

            poly.pixel_position = (i, j)
            polygons.append(poly)
    for feature in ds_shp.GetLayer(): 
        if feature.GetField('height') >= Wall_hei_threshold:  
            for poly in polygons: 

                if feature.GetGeometryRef().Intersects(poly):

                    intersection = feature.GetGeometryRef().Intersection(poly)
                    if intersection:
                        intersection_area = intersection.Area()

                        i_n, j_n = poly.pixel_position
                        area_percentage1.append([i_n,j_n,intersection_area / (3*3)])
                        if feature.GetField('height') >= Wall_hei_threshold_n:
                            area_percentage1_12.append([i_n,j_n,intersection_area / (3*3)])
    Shp_area,area_percentage4,_,wall_sele_all_0 = Functions.area(area_percentage1,wall_index,data_hei)
    _,area_percentage4_12,Wall_indexes,wall_sele_all = Functions.area(area_percentage1_12,wall_index,data_hei)

    if len(Wall_indexes) == 0:
        Functions.save(0, path_save,cc,df_read_facade_12_floor,df_read_roof_0,Build_info,Shp_area_all_0_12,df_read_facade_area,df_read_facade_capacity,df_read_facade_ele,df_read_facade_LCOE)
        del df_read_sam_rad
        del df_read_facade_LCOE
        continue

    Wall_indexes = Wall_indexes[Wall_indexes[:,2] >= Wall_hei_threshold_n]
    Wall_indexes = Wall_indexes[Wall_indexes[:,2] <= 600]

    Wall_ind_group = Functions.read_wall(Wall_indexes)

    for j_day in tqdm(range(len(day_list))):
        for j_time in range(len(time_list)):
            #####shanghai case
            # if sam == 20 and j_day == 9 and j_time == 5:
            #     continue
            # if sam == 20 and j_day == 10 and j_time == 4:
            #     continue
            # if sam == 36 and j_day == 9 and j_time == 5:
            #     continue
            # if sam == 38 and j_day == 11 and j_time == 4:
            #     continue
            # if sam == 44 and j_day == 5 and j_time == 2:
            #     continue
            # if sam == 54 and j_day == 10 and j_time == 5:
            #     continue
            s_t = time.time()
            file1 = Wall_path+cc+'_'+str(sam)+'_'+day_list[j_day]+'_'+time_list[j_time]+'/Energyyearroof.tif'
            file2 = Wall_path+cc+'_'+str(sam)+'_'+day_list[j_day]+'_'+time_list[j_time]+'/Energyyearwall.txt'
            if not os.path.exists(file1) or not os.path.exists(file2):
                Roof_solar_0 = 0
                Wall_solar_12_floor = [[0]]
            else:
                ds_tif = gdal.Open(file1)
                Wall_data = []
                if np.max(data_hei) > 600:
                    with open(file2, 'r') as file:
                        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmapped_file:
                            for line in iter(mmapped_file.readline, b""):
                                line = line.decode('utf-8')
                                Wall_data.append(line[:303])
                    Wall_data = Wall_data[1:]
                else:
                    with open(file2, 'r') as file:
                        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmapped_file:
                            for line in iter(mmapped_file.readline, b""):
                                line = line.decode('utf-8')
                                Wall_data.append(line)
                    Wall_data = Wall_data[1:]
                band = ds_tif.GetRasterBand(1)
                data = band.ReadAsArray()
                Roof_solar_0,_,_ = Functions.solar(data,area_percentage4,wall_sele_all_0,Wall_data)
                _,_,Wall_solar_12_floor = Functions.solar(data,area_percentage4_12,wall_sele_all,Wall_data)
            

            df_read_sam_rad[j_day*len(time_list)+j_time, :len(Wall_solar_12_floor)] = Wall_solar_12_floor[:]

            
            df_read_roof_0.iloc[j_day*len(time_list)+j_time,sam] = Roof_solar_0*9
            df_read_facade_12_floor.iloc[j_day*len(time_list)+j_time,sam] = sum(Wall_solar_12_floor)*9 # for i in range(len(Wall_solar_12_floor)))*9

            e_t = time.time()
            print(e_t-s_t)
    Shp_area_all_0_12.iloc[sam-1,0] = sum(Shp_area)

    Shp_area_all_0_12.iloc[sam-1,2] = 3*sum(Wall_indexes[:,2]-3.0)
    df_read_sam_rad_yes,df_read_LCOE,df_read_power,df_read_facade_capacity[sam-1,:] \
        = Functions.economy(df_read_sam_rad, Wall_ind_group, Wall_indexes,cc,transform_tif[0],transform_tif[3],TOU_price_indu,TOU_price_resi)
    effective_area = 0.9
    for case_i in range(6):
        index_case = np.argwhere(df_read_sam_rad_yes[:,case_i] == 1)
        df_read_facade_area[sam-1,case_i] = effective_area*3*sum(Wall_indexes[i,2]-3 for i in index_case[:,0])
        df_read_facade_ele[case_i].iloc[:,sam] = df_read_power[:,case_i]
        if len(df_read_LCOE[0]) <= df_read_facade_LCOE[0].shape[-1]:
            df_read_facade_LCOE[case_i][3*(sam-1),:len(df_read_LCOE[case_i])] = df_read_LCOE[case_i]
        elif len(df_read_LCOE[0]) <= 2*df_read_facade_LCOE[0].shape[-1]:
            df_read_facade_LCOE[case_i][3*(sam-1),:] = df_read_LCOE[case_i][:10000]
            df_read_facade_LCOE[case_i][3*(sam-1)+1,:(len(df_read_LCOE[case_i])-10000)] = df_read_LCOE[case_i][10000:len(df_read_LCOE[case_i])]


    DSM = gdal.Open(Wall_path+cc+'_'+str(sam)+'_DSM.tif')
    DSM_gt = DSM.GetGeoTransform() 

    DSM_xmin = DSM_gt[0]+33*3
    DSM_ymax = DSM_gt[3]-33*3
    DSM_xmax = DSM_xmin + 2001
    DSM_ymin = DSM_ymax - 2001
    Shp_hei,Shp_peri,Shp_hei_12,Shp_peri_12 = [],[],[],[]
    with fiona.open(Wall_path+str(sam)+'.shp') as SHP:
        crs = SHP.crs
        for feat in SHP:
            geom = shape(feat["geometry"])
            x,y = np.array(geom.centroid.coords)[0,0],np.array(geom.centroid.coords)[0,1]
            if x >= DSM_xmin and x <= DSM_xmax and y >= DSM_ymin and y <= DSM_ymax:
                hhei = feat["properties"]["height"]
                Build_info[sam-1,0] += 1
                Build_info[sam-1,int(hhei//3)] += 1
                if hhei >= Wall_hei_threshold:
                    Shp_hei.append(hhei)
                    Shp_peri.append(geom.length)
                    if hhei >= Wall_hei_threshold_n:
                        Shp_hei_12.append(hhei)
                        Shp_peri_12.append(geom.length)
    Shp_area_all_0_12.iloc[sam-1,1] = sum(Shp_peri_12[i]*(Shp_hei_12[i]-3.0) for i in range(len(Shp_peri_12)))
    Functions.save(0, path_save,cc,df_read_facade_12_floor,df_read_roof_0,Build_info,Shp_area_all_0_12,df_read_facade_area,df_read_facade_capacity,df_read_facade_ele,df_read_facade_LCOE)
    del df_read_sam_rad
    del df_read_facade_LCOE
