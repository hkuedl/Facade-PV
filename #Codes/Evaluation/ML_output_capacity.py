#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import shap
from sklearn.model_selection import GridSearchCV
import joblib

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
    
    return sam_select

def adjust_r2_score(y_test, y_te_pred, p):
    n = len(y_test)
    r2 = r2_score(y_test, y_te_pred)
    return 1-((1-r2)*(n-1))/(n-p-1)

def model_read(model_name):
    if model_name == 'DT':
        model = DecisionTreeRegressor(random_state=20, max_depth = 10)
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'random_state': [20]}
    elif model_name == 'LR':
        model = LinearRegression()
        param_grid = {}
    elif model_name == 'KNN':
        model = KNeighborsRegressor(n_neighbors = 3)
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'random_state': [20]}
    elif model_name == 'RF':
        model = RandomForestRegressor(n_estimators=100, random_state=20, max_depth = 10, min_samples_split = 2)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'random_state': [20]}
    elif model_name == 'SVR':
        model = SVR(kernel="rbf", C=10, gamma=0.1, epsilon=0.1)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4]
            }
    return model,param_grid

def re_input_new(Feas_input):
    indices_non_zero = np.where(Feas_input[:,11] != 0)[0]
    i_dele = [3]
    Feas_input_new = np.delete(Feas_input[indices_non_zero,:], i_dele, axis=1)
    return Feas_input_new

ALL_city = pd.read_excel('Climate_WWR.xlsx',sheet_name = 'Climate').iloc[:,1].tolist()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(20)

City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', 'Hefei', 'Huhehaote', 'Jinan', \
             'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

day_list = ['0115','0215','0315','0415','0515','0615','0715','0815','0915','1015','1115','1215']
day_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
time_list = ['08','10','12','14','16','18']

eff_wall,eff_win,eff_grid = [0.2,0.21,0.22,0.23,0.24,0.25],[0.15,0.16,0.17,0.18,0.19,0.2],0.8
are_wall = 180
are_win = 140
# Baseline filtered-AOI availability used to create the large public ML output
# archive. The RPV35/FPV70 main case is applied downstream by
# Multi_period_planning.py using main_case_city_totals.csv.
BASELINE_FILTERED_FACADE_INSTALLABILITY = 0.90
effective_area = BASELINE_FILTERED_FACADE_INSTALLABILITY

ALL_city_list = [i for i in range(102)]

def Land_type(Feas_read_use,Feas_read_info):
    types = [101,201,202,301,401,402,403,501,502,503,504,505]
    Landuse_volu = []  #np.zeros((1,12+1))
    Landuse_num = []  #np.zeros((1,12+1))
    Landuse_volu.append(sum(Feas_read_info[:,6])/1e9)
    Landuse_num.append(sum(Feas_read_info[:,0]))
    for ii in range(len(types)):
        type_index = np.where(Feas_read_use == types[ii])[0]
        Landuse_volu.append(sum(Feas_read_info[type_index,6])/1e9)
        Landuse_num.append(sum(Feas_read_info[type_index,0]))
    return Landuse_volu,Landuse_num

for cc_i in range(len(ALL_city_list)):
    city_ii = ALL_city_list[cc_i]
    city_name = ALL_city[city_ii]
    print(city_name)
    Feas_read_use = np.load(city_name+'_ALL_Featuers.npy')[:,14:15]
    if city_ii in [0,4,10,11,1,3,93,94,5,13,12,7,9]:
        Feas_read_info = np.load(city_name+'_ALL_Featuers_supplementary.npy')
    else:
        Feas_read_info = np.load(city_name+'_ALL_Featuers.npy')[:,range(17,29)]
    
    Feas_input = np.load(city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
    indices_non_zero = np.where(Feas_input[:,11] != 0)[0]
    Statis_Cap_Roof = 0.35*are_wall*Feas_read_info[:,3]/1e9
    Statis_Cap_Facade = np.zeros((len(indices_non_zero),7))
    WWR = np.load(city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
    Capa_facade = (are_win*WWR + are_wall*(1-WWR))*Feas_read_info[indices_non_zero,11:12]/1e9
    Statis_Cap_Facade[:,6:7] = Capa_facade*(1-0.1)*effective_area
    for i_lab in [2,3,4,5,6,7]:
        print(i_lab)
        case = i_lab - 2
        load_model_capa = joblib.load('RF' + '_Capa_' + str(i_lab) +'.joblib')
        Feas_input_capa = re_input_new(Feas_input)
        Feas_output_capa = load_model_capa.predict(Feas_input_capa)
        Statis_Cap_Facade[:,case] = Feas_output_capa/1e3
    
    np.save('Cap_facade_'+city_name+'.npy',Statis_Cap_Facade)
    np.save('Cap_roof_'+city_name+'.npy',Statis_Cap_Roof)
