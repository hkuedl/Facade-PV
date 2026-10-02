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
from pyproj import Transformer
from scipy.interpolate import interp1d,lagrange,make_interp_spline,CubicSpline


def sele_sam(np_facade,Shp_area_all_0_12,Shp_num_all):
    dele_mean_facade = np.mean(np_facade,axis = 0)/1000
    dele_ratio = np.zeros((64,2))
    for i in range(64):
        dele_ratio[i,0] = i+1
        if Shp_area_all_0_12[i,2] == 0 or Shp_num_all[i] <= 1:
            dele_ratio[i,1] = 0
        else:
            dele_ratio[i,1] = dele_mean_facade[i]/(Shp_area_all_0_12[i,2]/1e6)
    dete_0 = np.where(dele_ratio[:,1] == 0)[0]
    dete_ratio_0 = np.delete(dele_ratio, dete_0, axis=0)
    dete_z_scores = (dete_ratio_0[:,1] - np.mean(dete_ratio_0[:,1]))/np.std(dete_ratio_0[:,1])
    dete_out_ind = np.where(abs(dete_z_scores) > 2)[0]
    dele_out = dete_ratio_0[dete_out_ind,0]
    dele_0 = dele_ratio[dete_0,0]
    sam_dele = list(dele_0) + list(dele_out)
    sam_select = [num for num in range(1,65) if num not in sam_dele]
    print(dele_0)
    print(dele_out)
    
    return sam_select

def save(name,path,cc,df_read_facade_12_floor,df_read_roof_0,Build_info,Shp_area_all_0_12,df_read_facade_area,df_read_facade_capacity,df_read_facade_ele,df_read_facade_LCOE):
    if name == 0:
        file = path+'/'+cc+'_simulation_results_Final.xlsx'
    else:
        file = path+'/'+cc+'_simulation_results_Final_0m.xlsx'
    writer = pd.ExcelWriter(file)
    df_read_facade_12_floor.to_excel(writer,sheet_name='Inst_Facade_floor',index=False)
    df_read_roof_0.to_excel(writer,sheet_name='Inst_Roof',index=False)
    Build_info_pd = pd.DataFrame(Build_info)
    Build_info_pd.to_excel(writer,sheet_name='Build_info',index=False)
    Shp_area_all_0_12.to_excel(writer,sheet_name='Shp_area',index=False)
    df_read_facade_area_pd = pd.DataFrame(df_read_facade_area)
    df_read_facade_area_pd.to_excel(writer,sheet_name='Eco_area',index=False)
    df_read_facade_capacity_pd = pd.DataFrame(df_read_facade_capacity)
    df_read_facade_capacity_pd.to_excel(writer,sheet_name='Eco_capacity',index=False)
    for i in range(6):
        df_read_facade_ele[i].to_excel(writer,sheet_name='Eco_Facade_'+str(i),index=False)
        df_read_facade_LCOE_pd = pd.DataFrame(df_read_facade_LCOE[i])
        df_read_facade_LCOE_pd.to_excel(writer,sheet_name='Eco_LCOE_'+str(i),index=False)
        
    writer.close()

def read(name,path,cc):
    if name == 0:
        file = path+'/'+cc+'_simulation_results_Final.xlsx'
    else:
        file = path+'/'+cc+'_simulation_results_Final_0m.xlsx'
    df_read_roof_0 = pd.read_excel(file,sheet_name = 'Inst_Roof')
    df_read_facade_12_floor = pd.read_excel(file,sheet_name = 'Inst_Facade_floor')
    df_read_facade_LCOE = []
    df_read_facade_area = pd.read_excel(file,sheet_name = 'Eco_area').to_numpy()
    df_read_facade_capacity = pd.read_excel(file,sheet_name = 'Eco_capacity').to_numpy()
    
    df_read_facade_ele = []
    for i in range(6):
        df_read_facade_ele.append(pd.read_excel(file,sheet_name = 'Eco_Facade_'+str(i)))
        df_read_facade_LCOE.append(pd.read_excel(file,sheet_name = 'Eco_LCOE_'+str(i)).to_numpy())
    Build_info = pd.read_excel(file,sheet_name = 'Build_info').to_numpy()
    Shp_area_all_0_12 = pd.read_excel(file,sheet_name = 'Shp_area')
    return df_read_facade_12_floor,df_read_roof_0,Build_info,Shp_area_all_0_12,df_read_facade_area,df_read_facade_capacity,df_read_facade_ele,df_read_facade_LCOE

def area(area_percentage1,wall_index,data_hei):
    Shp_area = []
    area_percentage3 = np.zeros((len(area_percentage1),3))
    for i in range(area_percentage3.shape[0]):
        area_percentage3[i,:] = area_percentage1[i][:]
    
    [_,area_index] = np.unique(area_percentage3[:,:2],axis=0, return_index=True)
    
    area_percentage4 = area_percentage3[area_index,:]
    
    unique_elements, inverse_indices = np.unique(area_percentage3[:,:2],axis=0, return_inverse=True)
    grouped_indices = [np.where(inverse_indices == i)[0] for i in range(len(unique_elements))]
    
    jjj1,jjj2 = 0,0
    for i in range(len(grouped_indices)):
        if len(grouped_indices[i]) > 1:
            jjj1 += 1
            new_sum = area_percentage3[grouped_indices[i][:],2]
            if sum(new_sum) <= 1:
                jjj2 += 1
                area_percentage4[i,2] = sum(new_sum)
            else:
                area_percentage4[i,2] = 1
        Shp_area.append(area_percentage4[i,2]*9.0)

    wall_indexes = []
    wall_sele_all = []
    for ff in range(area_percentage4.shape[0]):
        Build_loc = [(int(area_percentage4[ff,0])+1, int(area_percentage4[ff,1])+1)]
        dist_matrix = cdist(Build_loc, wall_index)
        wall_sele = np.argmin(dist_matrix)
        if wall_sele not in wall_sele_all:
            if data_hei[int(wall_index[wall_sele,0]-1),int(wall_index[wall_sele,1]-1)] <= 600:
                wall_sele_all.append(wall_sele)
                wall_indexes.append([wall_index[wall_sele,0],wall_index[wall_sele,1],data_hei[int(wall_index[wall_sele,0]-1),int(wall_index[wall_sele,1]-1)]])
    return Shp_area,area_percentage4,np.array(wall_indexes),wall_sele_all


def solar(data,area_percentage4,wall_sele_all,Wall_data):
    Roof_solar = 0
    Wall_solar = 0
    Wall_solar_floor = []
    
    for ff in range(area_percentage4.shape[0]):
        ff_data = data[int(area_percentage4[ff,0]), int(area_percentage4[ff,1])]
        if ff_data > 0:
            Roof_solar += ff_data * area_percentage4[ff,2]
    for wall_sele in wall_sele_all:
        i_list = Wall_data[wall_sele]
        number_str = re.findall(r'\d+\.?\d*', i_list)
        number_value = [float(i) for i in number_str]
        Wall_solar += sum(number_value[2:])
        Wall_solar_floor.append(sum(number_value[3:]))

    return Roof_solar,Wall_solar,Wall_solar_floor

def read_wall(data):
    df = pd.DataFrame(data[:,:2], columns=['Column1', 'Column2'])
    df.sort_values(['Column1', 'Column2'], inplace=True)
    df['Diff'] = df.groupby('Column1')['Column2'].diff().fillna(0)
    df['NewGroup'] = ((df['Diff'] > 1) | (df['Column1'] != df['Column1'].shift())).astype(int)
    df['GroupLabel'] = df['NewGroup'].cumsum()
    data_new = []
    data_new_all = []
    for _, group in df.groupby('GroupLabel'):
        if group.shape[0] >= 2:
            data_new.append(list(group.index.values))
            data_new_all += list(group.index.values)
    
    df1 = df.drop(data_new_all).iloc[:,:2]
    df1.sort_values(['Column2', 'Column1'], inplace=True)
    df1['Diff'] = df1.groupby('Column2')['Column1'].diff().fillna(0)
    df1['NewGroup'] = ((df1['Diff'] > 1) | (df1['Column2'] != df1['Column2'].shift())).astype(int)
    df1['GroupLabel'] = df1['NewGroup'].cumsum()    
    for name, group in df1.groupby('GroupLabel'):
        #print(f"Group {name}:")
        data_new.append(list(group.index.values))
        data_new_all += list(group.index.values)
    return data_new

def interpol(y_year,med):
    ynew_year = np.zeros((11*12,))
    x = np.array([0, 1, 2, 3, 4, 5])  
    for i in range(12):
        y = y_year[i*6:(i+1)*6]
        f_lin = interp1d(x, y)
        f_lag = lagrange(x, y)
        f_spl = make_interp_spline(x, y)
        f_cs = CubicSpline(x, y)
        xnew = np.linspace(0, 5, num = 11)  
        if med == 'Lin':
            ynew = f_lin(xnew)
        elif med == 'Lag':
            ynew = f_lag(xnew)
        elif med == 'Cs':
            ynew = f_cs(xnew)
        ynew_year[i*11:(i+1)*11] = ynew
    return ynew_year

def WWR_read(lon,lat,city_name,Landzone_i):
    a = np.array([lon,lat]).reshape(1, -1)
    Climate = pd.read_excel('Climate_WWR.xlsx',sheet_name = 'Climate')
    climate_zone_row = Climate[Climate.eq(city_name)].any(axis=1).to_numpy().nonzero()[0][0]
    climate_zone = Climate.iloc[climate_zone_row,-1]
    
    distances = cdist(a, Landzone_i[:,:2])
    idx = np.argmin(distances)
    if Landzone_i[idx,-2] == 4 or Landzone_i[idx,-2] == 5:
        WWR = 0.5
    elif Landzone_i[idx,-2] == 3:
        WWR = 0.4
    elif Landzone_i[idx,-2] == 2:
        WWR = 0.2
    elif Landzone_i[idx,-2] == 1:
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
    return WWR,Landzone_i[idx,-2]

def LCOE(Power,price1,price2,area1,area2,WWR):
    year_op,year_rec = 25,15
    shuaiji = [0.01]+[0.004 for i in range(year_op-1)]
    const_cost = price1*(1-WWR)*area1+price2*WWR*area2
    IC1,IC2,IC3 = const_cost, const_cost*0.05/(1.08**year_rec), sum(0.047*(area1*(1-WWR)+area2*WWR)/(1.08)**i for i in range(1,year_op+1))
    IC4 = sum(Power*(1-sum(shuaiji[:i]))/(1.08)**i for i in range(1,year_op+1))
    return (IC1-IC2+IC3)/IC4

def economy(df_read_sam_rad_ana,Wall_ind_group,wall_indexes,cc,lon,lat,TOU_price_indu,TOU_price_resi):
    lands = pd.read_excel('Landuse.xlsx').to_numpy()
    
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
    y_ini,x_ini = transformer.transform(lon, lat)
    
    x_min, x_max = x_ini-0.5, x_ini+1.0
    y_min, y_max = y_ini-1.0, y_ini+0.5
    
    Landzone_i = lands[(lands[:, 0] >= x_min) & (lands[:, 0] <= x_max) & (lands[:, 1] >= y_min) & (lands[:, 1] <= y_max)]
    eff_wall,eff_win,eff_grid = [0.2,0.21,0.22,0.23,0.24,0.25],[0.15,0.16,0.17,0.18,0.19,0.2], 0.8
    pri_wall,are_wall,pri_win,are_win = [3.18,2.98,2.75,2.70,2.65,2.60], 180, [4.8,4.50,4.15,4.08,4.00,3.92], 140
    
    df_read_yes = np.zeros((df_read_sam_rad_ana.shape[1],6))
    df_read_LCOE = [[],[],[],[],[],[]]
    df_read_sam_power = [np.zeros((df_read_sam_rad_ana.shape[0],df_read_sam_rad_ana.shape[1])) for _ in range(6)]
    df_power = np.zeros((72,6))
    df_capacity = np.zeros((df_read_sam_rad_ana.shape[1],6))
    
    # Baseline used for the public ML archive; the planning workflow scales
    # these arrays to the saved main installation case.
    effective_area = 0.90

    for ii in range(len(Wall_ind_group)):
        lon_i = [lon+3*i for i in wall_indexes[Wall_ind_group[ii],0]]
        lat_i = [lat-3*i for i in wall_indexes[Wall_ind_group[ii],1]]
        lon_i = sum(lon_i)/len(lon_i)
        lat_i = sum(lat_i)/len(lat_i)
        y, x = transformer.transform(lon_i, lat_i)
        WWR,i_type = WWR_read(x,y,cc,Landzone_i)
        
        df_select = df_read_sam_rad_ana[:,Wall_ind_group[ii]]
        df_select_pol = np.zeros((11*12,len(Wall_ind_group[ii])))
        for ij in range(len(Wall_ind_group[ii])):
            df_select_pol[:,ij] = interpol(df_select[:,ij],'Cs')
        df_select_pol = np.sum(df_select_pol,axis = 1)
        df_select_pol = df_select_pol.reshape(12, 11)
        if i_type == 1:
            df_select_money = np.multiply(TOU_price_resi[:,8:19],df_select_pol)
        else:
            df_select_money = np.multiply(TOU_price_indu[:,8:19],df_select_pol)
        rad = 9*np.sum(df_select_money)/(3*sum(wall_indexes[Wall_ind_group[ii],2]-3))
        rad_year = rad*365/12 
        
        for case_i in range(6):
            if rad == 0:
                df_read_yes[Wall_ind_group[ii],case_i] = -1
                df_read_LCOE[case_i].append(0)
                df_read_sam_power[case_i][:,Wall_ind_group[ii]] = 0
                df_capacity[Wall_ind_group[ii],case_i] = 0
            else:
                C_power = effective_area*rad_year*(WWR*eff_win[case_i] + (1-WWR)*eff_wall[case_i])*eff_grid   #1\u5e74\u53d1\u591a\u5c11\u7535kWh/m2/year
                C_LCOE = LCOE(C_power, pri_wall[case_i], pri_win[case_i], are_wall, are_win, WWR)

                df_read_sam_power[case_i][:,Wall_ind_group[ii]] = 9*effective_area*df_read_sam_rad_ana[:,Wall_ind_group[ii]]*(WWR*eff_win[case_i] + (1-WWR)*eff_wall[case_i])*eff_grid
                
                if C_LCOE >= 1:
                    df_read_yes[Wall_ind_group[ii],case_i] = -1
                    df_capacity[Wall_ind_group[ii],case_i] = 0
                else:
                    df_read_yes[Wall_ind_group[ii],case_i] = 1
                    df_capacity[Wall_ind_group[ii],case_i] = effective_area*3*(wall_indexes[Wall_ind_group[ii],2]-3)*(WWR*are_win+(1-WWR)*are_wall)
                df_read_LCOE[case_i].append(C_LCOE)

    for case_i in range(6):
        index_case = np.argwhere(df_read_yes[:,case_i] == 1)
        for i in range(72):
            df_power[i,case_i] = np.sum(df_read_sam_power[case_i][i,index_case])
    return df_read_yes,df_read_LCOE,df_power,np.sum(df_capacity,axis = 0)

def TOU_period(cc):
    price_file = pd.read_excel('Realtime Price.xlsx',sheet_name = 'Sheet1')
    row_index = price_file[price_file.eq(cc)].any(axis=1).idxmax()
    cc_province = price_file.iloc[row_index,0]
    TOU_time = np.zeros((12,24))
    numbers = list(range(1,13))
    if cc_province == '\u5e7f\u4e1c':
        for i in [7,8,9]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,1,3,3,2,1,1,2,2,3,3,3,3,3]
        for i in [num for num in numbers if num not in {7,8,9}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,2,3,3,2,2,2,2,2,3,3,3,3,3]
    if cc_province == '\u6c5f\u82cf':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,2,2,2,3,3,3,3,3,3,2,2,2,2,2,3,3]
    if cc_province == '\u5c71\u897f':
        for i in [1,7,8,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,2,2,2,4,4,3,3,3,3,2,1,1,2,2,2,3]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,2,2,2,4,4,3,3,3,3,2,2,2,2,2,2,3]
    if cc_province == '\u6d59\u6c5f':
        for i in [1,7,8,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,2,1,1,4,4,3,3,1,1,2,2,2,2,2,2,3]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,2,2,2,4,4,2,2,2,2,3,3,3,3,3,3,3]
    if cc_province == '\u91cd\u5e86':
        for i in [1,7,8,9,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,3,2,1,1,2,2,2,3,3,3,2,2,3,3]
        for i in [num for num in numbers if num not in {1,7,8,9,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,3,2,2,2,2,2,2,3,3,3,2,2,3,3]
    if cc_province == '\u5929\u6d25':
        for i in [7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,2,2,1,3,3,3,3,1,2,2,2,2,3,3,4]
        for i in [num for num in numbers if num not in {7,8}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,2,2,2,3,3,3,3,2,2,2,2,2,3,3,4]
    if cc_province == '\u6e56\u5357':
        for i in [1,7,8,9,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,2,2,2,3,3,3,3,1,1,1,1,2,4]
        for i in [num for num in numbers if num not in {1,7,8,9,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,2,2,2,3,3,3,3,2,2,2,2,2,4]
    if cc_province == '\u6d77\u5357':
        for i in [5,6,7]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,3,2,2,2,2,1,1,3,4]
        for i in [num for num in numbers if num not in {5,6,7}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,3,2,2,2,2,2,2,3,4]
    if cc_province == '\u9655\u897f':
        for i in [7,8]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,4,4,4,4,2,2,2,2,1,1,2,2,3]
        for i in [1,12]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,4,4,4,4,2,2,2,1,1,2,2,2,3]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,4,4,4,4,2,2,2,2,2,2,2,2,3]
    if cc_province == '\u9ed1\u9f99\u6c5f':
        for i in [1,7,8,9,11,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,2,3,3,2,2,2,3,3,3,3,2,1,1,2,3,3,3,4]
        for i in [num for num in numbers if num not in {1,7,8,9,11,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,2,3,3,2,2,2,3,3,3,3,2,2,2,2,3,3,3,4]
    if cc_province == '\u8fbd\u5b81':
        for i in [1,7,8,11,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,3,3,3,2,2,2,3,4,3,3,3,2,1,1,2,2,3,4,4]
        for i in [num for num in numbers if num not in {1,7,8,11,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,3,3,3,2,2,2,3,4,3,3,3,2,2,2,2,2,3,4,4]
    if cc_province == '\u5409\u6797':
        for i in [1,2,7,8,11,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,1,1,2,2,2,3,3,4]
        for i in [num for num in numbers if num not in {1,2,7,8,11,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,2,2,2,2,2,3,3,4]
    if cc_province == '\u56db\u5ddd':
        for i in [7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,1,1,2,2,2,2,3,3,4]
        for i in [1,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,2,2,2,2,1,1,3,3,4]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,2,2,2,2,2,2,3,3,4]
    if cc_province == '\u5180\u5317':
        for i in [6,7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,3,2,1,1,1,2,2,3,4]
        for i in [1,11,12]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,4,3,2,2,3,3,4,4,3,3,2,1,1,2,2,2,3,3]
        for i in [num for num in numbers if num not in {1,6,7,8,11,12}]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,4,3,2,2,3,3,4,4,3,3,2,2,2,2,2,2,3,3]
    if cc_province == '\u798f\u5efa':
        for i in [7,8,9]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,1,3,3,3,2,2,1,2,2,3,2,3,3]
        for i in [num for num in numbers if num not in {7,8,9}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,2,3,3,3,2,2,2,2,2,3,2,3,3]
    if cc_province == '\u5317\u4eac':
        for i in [7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,1,1,3,3,3,1,2,2,2,2,2,3,4]
        for i in [1,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,2,1,1,1,2,3,4]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,2,2,2,2,2,3,4]
    if cc_province == '\u9752\u6d77':
        for i in [1,2,3,10,11,12]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,1,4,4,4,4,4,4,4,4,2,2,1,1,2,2,3]
        for i in [num for num in numbers if num not in {1,2,3,10,11,12}]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,4,4,4,4,4,2,2,2,2,2,2,3]
    if cc_province == '\u4e91\u5357':
        for i in [1,3,4,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,2,2,1,3,3,3,3,3,2,1,2,2,2,3,4]
        for i in [num for num in numbers if num not in {1,3,4,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,2,2,2,3,3,3,3,3,2,2,2,2,2,3,4]
    if cc_province == '\u8499\u4e1c':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,3,2,2,2,3,3,4,4,4,3,3,3,2,2,2,2,2,3,3]
    if cc_province == '\u8499\u897f':
        for i in [6,7,8]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,2,2,3,3,3,4,4,4,4,4,3,3,2,2,2,2,3,3]
        for i in [num for num in numbers if num not in {6,7,8}]:
            TOU_time[i-1,:] = [4,4,4,4,3,3,2,2,3,3,3,4,4,4,4,4,3,3,2,2,2,2,3,3]
    if cc_province == '\u65b0\u7586':
        for i in [7]:
            TOU_time[i-1,:] = [3,3,3,3,4,4,4,4,2,2,2,3,3,4,5,5,4,3,3,2,2,1,1,2]
        for i in [1,11,12]:
            TOU_time[i-1,:] = [3,3,3,3,4,4,4,4,2,2,2,3,3,4,4,4,4,3,3,1,1,2,2,2]
        for i in [5,6,8]:
            TOU_time[i-1,:] = [3,3,3,3,4,4,4,4,2,2,2,3,3,4,5,5,4,3,3,2,2,2,2,2]
        for i in [num for num in numbers if num not in {1,5,6,7,8,11,12}]:
            TOU_time[i-1,:] = [3,3,3,3,4,4,4,4,2,2,2,3,3,4,4,4,4,3,3,2,2,2,2,2]
    if cc_province == '\u5c71\u4e1c':
        for i in [1,2,12]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,3,3,3,4,5,5,5,4,3,1,1,1,2,2,3,3,3]
        for i in [3,4,5]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,3,3,3,4,5,5,5,4,3,3,1,1,1,2,2,3,3]
        for i in [6,7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,2,1,1,1,1,1,3,3]
        for i in [9,10,11]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,3,3,3,4,5,5,5,4,3,2,1,1,2,2,3,3,3]
    if cc_province == '\u8d35\u5dde':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,2,2,3,3,3,3,2,2,2,2,2,3,3]
    if cc_province == '\u5e7f\u897f':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,2,2,3,3,3,3,2,2,2,2,2,2,4]
    if cc_province == '\u5b81\u590f':
        for i in [7,8]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,4,4,4,4,4,2,2,1,1,2,2,3]
        for i in [12]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,4,4,4,4,4,2,1,1,2,2,2,3]
        for i in [4,5,9]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,5,5,4,4,4,2,2,2,2,2,2,3]
        for i in [num for num in numbers if num not in {4,5,7,8,9,12}]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,4,4,4,4,4,2,2,2,2,2,2,3]
    if cc_province == '\u7518\u8083':
        for i in numbers:
            TOU_time[i-1,:] = [3,3,3,3,3,3,2,2,3,3,4,4,4,4,4,4,3,3,2,2,2,2,2,3]
    if cc_province == '\u4e0a\u6d77':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4]
    if cc_province == '\u5b89\u5fbd':
        for i in [7,8,9]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,2,2,2,2,1,1,2,2]
        for i in [1,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,2,2,2,2,1,1,2,2,4]
        for i in [num for num in numbers if num not in {1,7,8,9,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,2,2,2,3,3,3,3,3,2,2,2,2,2,3,3,4]
    if cc_province == '\u6e56\u5317':
        for i in [7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,4,4,3,3,2,2,2,2,1,1,2,2]
        for i in [num for num in numbers if num not in {7,8}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,4,4,3,3,2,2,1,1,2,2,2,2]
    if cc_province == '\u6c5f\u897f':
        for i in [1,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,3,3,1,1,2,3,3,3]
        for i in [7,8,9]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,1,1,3]
        for i in [num for num in numbers if num not in {1,7,8,9,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,3,3]
    if cc_province == '\u6cb3\u5357':
        for i in [1,2,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,2,1,1,2,2,2,2,2]
        for i in [3,4,5,9,10,11]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,4,4,4,3,3,2,2,2,2,2,2,2,2]
        for i in [6,7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,2,2,2,2,1,1,1,2]
    if cc_province == '\u6cb3\u5317':
        for i in [6,7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,2,2,2,2,1,1,1,2,3]
        for i in [1,2,12]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,3,4,4,4,3,2,1,1,2,2,2,2,2]
        for i in [num for num in numbers if num not in {1,2,6,7,8,12}]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,3,4,4,4,3,2,2,2,2,2,2,2,2]
    
    TOU_price_indu = np.zeros((12,24))
    TOU_price_resi = np.zeros((12,24))
    TOU_price_coal = np.zeros((12,24))
    
    TOU_price_coal[:,:] = price_file.iloc[row_index,2]
    for mm in range(1,13):
        mm0 = 11 if mm == 12 else mm
        TOU = pd.read_excel('Realtime Price.xlsx',sheet_name = str(mm0))
        if cc_province in TOU['\u5730\u533a'].values:
            row_index = TOU[TOU['\u5730\u533a'] == cc_province].index[0]
            for ii in range(24):
                tou_i = 4 if int(TOU_time[mm-1,ii]) == 5 else int(TOU_time[mm-1,ii])
                TOU_price_indu[mm-1,ii] = TOU.iloc[row_index,tou_i]
            TOU_price_resi[mm-1,:] = 0.3*TOU.iloc[row_index,3] + 0.7*TOU.iloc[row_index,4]
        else:
            print(f"\u5b57\u7b26\u4e32 '{cc_province}{mm}' \u4e0d\u5b58\u5728")

    return TOU_time,TOU_price_indu,TOU_price_resi,TOU_price_coal






def economy_4_relation(df_read_sam_rad_ana,Wall_ind_group,wall_indexes,cc,lon,lat,TOU_price_indu,TOU_price_resi,case_i):

    lands = pd.read_excel('Landuse.xlsx').to_numpy()
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
    y_ini,x_ini = transformer.transform(lon, lat)
    
    x_min, x_max = x_ini-0.5, x_ini+1.0
    y_min, y_max = y_ini-1.0, y_ini+0.5
    
    Landzone_i = lands[(lands[:, 0] >= x_min) & (lands[:, 0] <= x_max) & (lands[:, 1] >= y_min) & (lands[:, 1] <= y_max)]
    eff_wall,eff_win,eff_grid = [0.2,0.21,0.22,0.23,0.24,0.25],[0.15,0.16,0.17,0.18,0.19,0.2], 0.8
    pri_wall,are_wall,pri_win,are_win = [3.18,2.98,2.75,2.70,2.65,2.60], 180, [4.8,4.50,4.15,4.08,4.00,3.92], 140
    
    df_read_yes = np.zeros((df_read_sam_rad_ana.shape[1],6))
    df_read_LCOE = [[],[],[],[],[],[]]
    df_read_sam_power = [np.zeros((df_read_sam_rad_ana.shape[0],df_read_sam_rad_ana.shape[1])) for _ in range(6)]
    df_power = np.zeros((df_read_sam_rad_ana.shape[1],6))
    df_capacity = np.zeros((df_read_sam_rad_ana.shape[1],6))
    
    effective_area = 0.9

    for ii in range(len(Wall_ind_group)):
        lon_i = [lon+3*i for i in wall_indexes[Wall_ind_group[ii],0]]
        lat_i = [lat-3*i for i in wall_indexes[Wall_ind_group[ii],1]]
        lon_i = sum(lon_i)/len(lon_i)
        lat_i = sum(lat_i)/len(lat_i)
        y, x = transformer.transform(lon_i, lat_i)
        WWR,i_type = WWR_read(x,y,cc,Landzone_i)
        
        df_select = df_read_sam_rad_ana[:,Wall_ind_group[ii]]
        df_select_pol = np.zeros((11*12,len(Wall_ind_group[ii])))
        for ij in range(len(Wall_ind_group[ii])):
            df_select_pol[:,ij] = interpol(df_select[:,ij],'Cs')
        df_select_pol = np.sum(df_select_pol,axis = 1)
        df_select_pol = df_select_pol.reshape(12, 11)
        if i_type == 1:
            df_select_money = np.multiply(TOU_price_resi[:,8:19],df_select_pol)
        else:
            df_select_money = np.multiply(TOU_price_indu[:,8:19],df_select_pol)
        rad = 9*np.sum(df_select_money)/(3*sum(wall_indexes[Wall_ind_group[ii],2]-3))
        rad_year = rad*365/12
        if rad == 0:
            df_read_yes[Wall_ind_group[ii],case_i] = -1
            df_read_LCOE[case_i].append(0)
            df_read_sam_power[case_i][:,Wall_ind_group[ii]] = 0
            df_capacity[Wall_ind_group[ii],case_i] = effective_area*3*(wall_indexes[Wall_ind_group[ii],2]-3)*(WWR*are_win+(1-WWR)*are_wall)
        else:
            C_power = effective_area*rad_year*(WWR*eff_win[case_i] + (1-WWR)*eff_wall[case_i])*eff_grid
            C_LCOE = LCOE(C_power, pri_wall[case_i], pri_win[case_i], are_wall, are_win, WWR)
            df_read_sam_power[case_i][:,Wall_ind_group[ii]] = 9*effective_area*df_read_sam_rad_ana[:,Wall_ind_group[ii]]*(WWR*eff_win[case_i] + (1-WWR)*eff_wall[case_i])*eff_grid
            
            if C_LCOE >= 1:
                df_read_yes[Wall_ind_group[ii],case_i] = -1
            else:
                df_read_yes[Wall_ind_group[ii],case_i] = 1
            df_capacity[Wall_ind_group[ii],case_i] = effective_area*3*(wall_indexes[Wall_ind_group[ii],2]-3)*(WWR*are_win+(1-WWR)*are_wall)
            df_read_LCOE[case_i].append(C_LCOE)

    for i in range(df_power.shape[0]):
        pp = df_read_sam_power[case_i][:,i]
        pp1 = interpol(pp,'Cs')
        df_power[i,case_i] = sum(pp1)*365/12
    return df_read_yes,df_read_LCOE,df_power,df_capacity

