#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import shap
from sklearn.model_selection import GridSearchCV
import joblib


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
        model = RandomForestRegressor(n_estimators=100, random_state=20, max_depth = 20, min_samples_split = 2)
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

def re_input_new(i_lab,Feas_input,Feas_read_info):
    indices_non_zero = np.where(Feas_input[:,11] != 0)[0]
    if i_lab >= 1:
        i_dele = [3]
        Feas_input_new = np.delete(Feas_input[indices_non_zero,:], i_dele, axis=1)
    if i_lab == 0:
        Feas_input_new = Feas_input[:,[0,1,3,16,17,18,19,20]]
        Feas_input_new[:,2] = Feas_read_info[:,0]
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

City_sele = [_ for _ in range(0,30)]

Feas_read_fea = []
Feas_read_lab = []
Feas_read_inf = []
Feas_read_eff_area = []

for cc in City_sele:
    city_name = City_name[cc]
    Feas_read_fea.append(np.load(city_name+'_ML_features.npy')[:,1:22])
    Feas_read_lab.append(np.load(city_name+'_ML_features.npy')[:,25:]/1e3)
    Feas_read_inf.append(np.load(city_name+'_ML_features.npy')[:,23:24]/1e6)

Feas_input = np.vstack((Feas_read_fea))
Feas_output = np.vstack((Feas_read_lab))
Feas_read_inf = np.vstack((Feas_read_inf))
print("Array contains NaN:",np.isnan(Feas_output).any())
if np.isnan(Feas_output).any():
    Feas_input = np.nan_to_num(Feas_input, nan=0.0)
    Feas_output = np.nan_to_num(Feas_output, nan=0.0)

i_lab = 2

print('Sample numbers')
print(Feas_input.shape[0]/72)

pcc_fea_lab = []
for i in range(2,14):
    pcc_fea_lab.append(np.corrcoef(Feas_input[:,i], Feas_output[:,i_lab])[0,1])

pccs_fea_fea = np.corrcoef(Feas_input[:,2:14].T)

Feas_input = re_input_new(i_lab,Feas_input,Feas_read_inf)


eff_wall,eff_win,eff_grid = [0.2,0.21,0.22,0.23,0.24,0.25],[0.15,0.16,0.17,0.18,0.19,0.2],0.8
are_wall = 180  #W/m2
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


case = i_lab - 2

load_model_roof = joblib.load('RF' + '_' + str(0) +'.joblib')
load_model_facade = joblib.load('RF' + '_new_' + str(case+2) +'.joblib')
load_model_facade_ideal = joblib.load('RF' + '_new_' + str(1) +'.joblib')

Abnor_city = [2,8,92,97,6]
Abnor_data = np.array([[2211.18, 1408.73, 1189.29, 8.08],[322.11, 185.29, 159.99, 1.26],\
                       [1024.77, 679.62, 529.76, 4.17],[1873.33, 1103.92, 1007.07, 8.34],\
                       [752.92,469.28,367.97,2.64]])

for cc_i in range(102):  #len(ALL_city_list)):
    city_ii = ALL_city_list[cc_i]
    city_name = ALL_city[city_ii]
    print(city_name)
    Feas_read_use = np.load(city_name+'_ALL_Featuers.npy')[:,14:15]
    if city_ii in [0,4,10,11,1,3,93,94,5,13,12,7,9]:
        Feas_read_info = np.load(city_name+'_ALL_Featuers_supplementary.npy')
    else:
        Feas_read_info = np.load(city_name+'_ALL_Featuers.npy')[:,range(17,29)]
    Capa_roof = 0.35*are_wall*sum(Feas_read_info[:,3])/1e9   #GW
    Feas_read_sta = np.load(city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
    Feas_read_dyn = np.load(city_name+'_ALL_Featuers.npy')[:,29:]
    Feas_read_roof = np.load(city_name+'_ALL_Featuers.npy')[:,20:21]/1e6   #0m
    indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
    WWR = np.load(city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
    Feas_grid_type = np.hstack((Feas_read_use[indices_non_zero,:], Feas_read_sta[indices_non_zero,4:6], Feas_read_info[indices_non_zero,2:3], Feas_read_info[indices_non_zero,11:12], Feas_read_sta[indices_non_zero,3:4],Feas_read_sta[indices_non_zero,:2]))
    
    i_dele = [3]
    Type_static = np.delete(Feas_read_sta[indices_non_zero,:], i_dele, axis=1)
    Type_weaii = np.zeros((len(indices_non_zero),3,365,15))
    Type_wea = np.zeros((len(indices_non_zero),3))
    for day in range(365):
        for hour in range(15):
            Type_weaii[:,0,day,hour] = Feas_read_dyn[indices_non_zero,(5*(day*15+hour)+2)]
            Type_weaii[:,1,day,hour] = Feas_read_dyn[indices_non_zero,(5*(day*15+hour)+3)]
            Type_weaii[:,2,day,hour] = Feas_read_dyn[indices_non_zero,(5*(day*15+hour)+4)]            
    Type_wea = np.sum(Type_weaii[:,:,:,:],axis = (2,3))
    np.save(city_name+'_static.npy',Type_static)
    np.save(city_name+'_wea.npy',Type_wea)
    
    Feas_output_roof = np.zeros((Feas_read_sta.shape[0],8760))
    Feas_output_facade = np.zeros((len(indices_non_zero),8760))
    Feas_output_facade_ideal = np.zeros((len(indices_non_zero),8760))
    for day in range(365):
        for hour in range(15):
            Feas_dyn_i = Feas_read_dyn[:,5*(day*15+hour):5*(day*15+hour+1)]
            Feas_input = np.hstack((Feas_read_sta,Feas_dyn_i))
            Feas_input_roof = re_input_new(0,Feas_input,Feas_read_roof)
            Feas_input_facade = re_input_new(case+2,Feas_input,Feas_read_roof)
            Feas_input_facade_ideal = re_input_new(1,Feas_input,Feas_read_roof)
            
            Feas_output_roof[:,day*24+6+hour] = load_model_roof.predict(Feas_input_roof)*0.35*eff_wall[case]*eff_grid
            Feas_output_facade[:,day*24+6+hour] = load_model_facade.predict(Feas_input_facade)
            Feas_output_facade_ideal[:,day*24+6+hour:day*24+6+hour+1] = load_model_facade_ideal.predict(Feas_input_facade_ideal).reshape(-1,1)*(eff_wall[case]*(1-WWR)+eff_win[case]*WWR)*eff_grid*effective_area
    
    np.save('Grid_type'+'_'+city_name+'.npy',Feas_grid_type)
    np.save('N_P_facade_ideal_'+str(i_lab-1)+'_'+city_name+'.npy',Feas_output_facade_ideal)
    np.save('N_P_roof_'+str(i_lab-1)+'_'+city_name+'.npy',Feas_output_roof)
    np.save('N_P_facade_'+str(i_lab-1)+'_'+city_name+'.npy',Feas_output_facade)
