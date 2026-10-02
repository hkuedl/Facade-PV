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

City_sele = [_ for _ in range(0,30)] #+ [_ for _ in range(20,30)]

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

i_lab = 1  #0: Rooftop solar; 1: Facade solar; 2-7: economic power

print('Sample numbers')
print(Feas_input.shape[0]/72)

pcc_fea_lab = []
for i in range(2,14):
    pcc_fea_lab.append(np.corrcoef(Feas_input[:,i], Feas_output[:,i_lab])[0,1])

pccs_fea_fea = np.corrcoef(Feas_input[:,2:14].T)

Feas_input = re_input_new(i_lab,Feas_input,Feas_read_inf)

X_train, X_test, y_train, y_test = train_test_split(Feas_input, Feas_output[:,i_lab:i_lab+1], test_size=0.2, random_state=20)

models_name = ['DT','LR','KNN','RF','XGB','SVR']
model_name = 'RF'  #'DT','LR','KNN'
fold = 'FTrue'
minmax = 'False'
Grid_ser = 'FTrue'

X_train, X_test, y_train, y_test = train_test_split(Feas_input, Feas_output[:,i_lab:i_lab+1], test_size=0.2, random_state=20)
y_train, y_test = y_train[:,0],y_test[:,0]

if model_name == 'XGB':
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 20,
        'learning_rate': 0.6,
        'seed':20
    }
    model_xgb = xgb.train(params, dtrain)
    y_tr_pred,y_te_pred = model_xgb.predict(dtrain),model_xgb.predict(dtest)

    Xgb_acc = np.zeros((4,2))
    Xgb_acc[:,0] = [np.sqrt(mean_squared_error(y_train, y_tr_pred)), mean_absolute_error(y_train, y_tr_pred), r2_score(y_train, y_tr_pred), adjust_r2_score(y_train, y_tr_pred,Feas_input.shape[1])]
    Xgb_acc[:,1] = [np.sqrt(mean_squared_error(y_test, y_te_pred)), mean_absolute_error(y_test, y_te_pred), r2_score(y_test, y_te_pred), adjust_r2_score(y_test, y_te_pred,Feas_input.shape[1])]
    print('Results of training set')
    print(Xgb_acc[:,0])
    print('Results of test set')
    print(Xgb_acc[:,1])
else:
    Train_acc = np.zeros((8,12))
    model,param_grid = model_read(model_name)
    if Grid_ser == 'True':
        
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        print("Best Parameters:", grid_search.best_params_)
        model = grid_search.best_estimator_
    
    if fold == 'True':
        kfold = KFold(n_splits = 10, shuffle=True, random_state=20)
        i_flag = 0
        for train_index, test_index in kfold.split(y_train):
            if Grid_ser == 'True':
                model = grid_search.best_estimator_
            print("Train:", train_index, "Validation:",test_index)
            X_train_i = X_train[train_index,:]
            X_test_i = X_train[test_index,:]
            y_train_i = y_train[train_index]
            y_test_i = y_train[test_index]
            model.fit(X_train_i, y_train_i)
            y_te_pred_i = model.predict(X_test_i)
            Train_acc[0:4,i_flag] = [np.sqrt(mean_squared_error(y_test_i, y_te_pred_i)), mean_absolute_error(y_test_i, y_te_pred_i), r2_score(y_test_i, y_te_pred_i), adjust_r2_score(y_test_i, y_te_pred_i,Feas_input.shape[1])]
            i_flag += 1
        Train_acc[0:4,10],Train_acc[0:4,11] = np.mean(Train_acc[0:4,:10],axis = 1),np.std(Train_acc[0:4,:10],axis = 1)
    
    model.fit(X_train, y_train)
    y_te_pred = model.predict(X_test)
    y_tr_pred = model.predict(X_train)
    Train_acc[4:,0] = [np.sqrt(mean_squared_error(y_train, y_tr_pred)), mean_absolute_error(y_train, y_tr_pred), r2_score(y_train, y_tr_pred), adjust_r2_score(y_train, y_tr_pred,Feas_input.shape[1])]
    Train_acc[4:,1] = [np.sqrt(mean_squared_error(y_test, y_te_pred)), mean_absolute_error(y_test, y_te_pred), r2_score(y_test, y_te_pred), adjust_r2_score(y_test, y_te_pred,Feas_input.shape[1])]

    print('Results of training set')
    print(Train_acc[4:,0])
    print('Results of test set')
    print(Train_acc[4:,1])
    print('Test data')
    print([np.mean(y_test),np.std(y_test)])

aTrain_acc_re = np.round(Train_acc,4)

y_err = np.abs(y_test - y_te_pred)
y_err_ratio = []
for i in range(len(y_test)):
    if y_test[i] == 0:
        y_err_ratio.append(0)
    else:
        y_err_ratio.append(y_err[i]/y_test[i])
y_err_ratio = np.array(y_err_ratio)
y_err_sort = y_err[np.argsort(y_err)[::-1]]
y_err_ratio_sort = y_err_ratio[np.argsort(y_err)[::-1]]
