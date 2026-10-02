"""Solve the 2030-2050 multi-period planning model.

The public workflow uses the main installation case (RPV 35% and FPV 70%
of the filtered facade AOI) and the land-use-informed annual loads. Large
baseline power arrays remain unchanged and are scaled to the saved main-case
city totals before optimization.

Set FPV_DATA_ROOT to the downloaded ##Data directory. Optional environment
variables can override individual input and output directories.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.io import savemat

import Functions


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(
    os.environ.get("FPV_DATA_ROOT", REPOSITORY_ROOT / "##Data")
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
LOAD_DIR = Path(
    os.environ.get(
        "FPV_LOAD_OUTPUT_DIR",
        DATA_ROOT / "Optimization" / "Loads" / "Landuse_Loads",
    )
).expanduser()
OPT_RESULTS_DIR = Path(
    os.environ.get(
        "FPV_OPT_OUTPUT_DIR",
        DATA_ROOT / "Optimization" / "Planning_results",
    )
).expanduser()
FIG_INPUT_DIR = Path(
    os.environ.get(
        "FPV_FIG_INPUT_DIR",
        DATA_ROOT / "Optimization" / "Fig_input_data",
    )
).expanduser()
CITY_STATISTICS_FILE = Path(
    os.environ.get(
        "FPV_CITY_STATISTICS_FILE",
        DATA_ROOT / "Optimization" / "Other_parameters" / "City_statistic.xlsx",
    )
).expanduser()
CITY_NORTH_FILE = Path(
    os.environ.get(
        "FPV_CITY_NORTH_FILE",
        DATA_ROOT / "Optimization" / "Loads" / "City_north_south.xlsx",
    )
).expanduser()
PRICE_FILE = Path(
    os.environ.get(
        "FPV_PRICE_FILE",
        DATA_ROOT / "Optimization" / "Electricity_price" / "Realtime Price.xlsx",
    )
).expanduser()
MAIN_CASE_FILE = Path(
    os.environ.get(
        "FPV_MAIN_CASE_FILE",
        FIG_INPUT_DIR / "main_case_city_totals.csv",
    )
).expanduser()

parser = argparse.ArgumentParser(
    description="Run the multi-period planning model for a city range."
)
parser.add_argument("--city-start", type=int, default=0, help="Inclusive city index.")
parser.add_argument("--city-end", type=int, default=102, help="Exclusive city index.")
parser.add_argument(
    "--solver",
    default=os.environ.get("FPV_OPT_SOLVER", "gurobi_direct"),
    help="Pyomo solver name. The solver license is configured outside this script.",
)
args = parser.parse_args()
if not 0 <= args.city_start < args.city_end <= 102:
    parser.error("Require 0 <= city-start < city-end <= 102.")

OPT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
path = str(ML_RESULTS_DIR) + os.sep
city_path = str(CITY_FEATURE_DIR) + os.sep
path_load = str(LOAD_DIR) + os.sep
path_opt = str(OPT_RESULTS_DIR) + os.sep
path_type = str(ML_RESULTS_DIR / "Power")
path_cap = str(ML_RESULTS_DIR / "Capacity")
path_main_case = str(MAIN_CASE_FILE)


def _scale_to_saved_total(target, raw_total, city_name, label):
    """Return the factor that matches each raw city total to the saved main case."""
    target = float(target)
    raw_total = float(raw_total)
    if not np.isfinite(target) or target < 0:
        raise ValueError(f'{city_name}: invalid saved {label} target: {target}')
    if not np.isfinite(raw_total) or raw_total <= 0:
        if target == 0:
            return 0.0
        raise ValueError(f'{city_name}: invalid raw {label} total: {raw_total}')
    return target / raw_total


# Read the saved main case directly, so the FPV/RPV ratios are defined in one place.
main_case_data = pd.read_csv(path_main_case)
required_main_columns = {
    'City', 'rpv_installability', 'fpv_installability_of_filtered_AOI',
    'rpv_capacity_GW', 'fpv_capacity_GW',
    'rpv_generation_TWh', 'fpv_generation_TWh'
}
missing_main_columns = required_main_columns.difference(main_case_data.columns)
if missing_main_columns:
    raise KeyError(
        f'{path_main_case} is missing columns: {sorted(missing_main_columns)}'
    )
main_case_data['City'] = main_case_data['City'].astype(str).str.strip()
if main_case_data['City'].duplicated().any():
    duplicated = main_case_data.loc[main_case_data['City'].duplicated(), 'City'].tolist()
    raise ValueError(f'Duplicated cities in main case: {duplicated}')
if not np.allclose(main_case_data['rpv_installability'].to_numpy(float), 0.35):
    raise ValueError('The saved main case is not uniformly RPV=0.35.')
if not np.allclose(
    main_case_data['fpv_installability_of_filtered_AOI'].to_numpy(float), 0.70
):
    raise ValueError('The saved main case is not uniformly FPV=0.70.')
main_case_data = main_case_data.set_index('City')

ABNORMAL_VOLUME_CITY_INDICES = {2, 6, 8, 69, 92, 97}
HEATING_ELECTRIFICATION_NORTH = 0.4
HEATING_ELECTRIFICATION_SOUTH = 0.6
City_statistic = pd.read_excel(CITY_STATISTICS_FILE, sheet_name="Class_Volume", index_col=0)
Statis_all = pd.read_excel(CITY_STATISTICS_FILE, sheet_name="Key_information", index_col=0)
are_wall,are_win = 180, 140  #W/m2

pri_change = [1,0.8,0.8,0.8,0.75,0.7]
wall_0 = [3.18]+[2.98,2.75,2.70,2.65,2.60]
win_0 = [4.80]+[4.50,4.15,4.08,4.00,3.92]
cap_ini = 370*1e6
K1_roof = [[1+0.006*i for i in [5,10,15,20,25]],[1+0.01*i for i in [5,10,15,20,25]]]
K1_pri_wall = [[1e3*wall_0[i] for i in range(6)],[pri_change[i]*1e3*wall_0[i] for i in range(6)]]
K1_pri_win  = [[1e3*win_0[i] for i in range(6)],[pri_change[i]*1e3*win_0[i] for i in range(6)]]

K3_Load = [1+0.02*i for i in [5,10,15,20,25]]

K2_C_pri = [[120,165,280,430,610],[240,330,560,860,1220]]

D_days = 12
mon_d = [31,28,31,30,31,30,31,31,30,31,30,31]
K2_C_factor = [np.zeros((102,5)),np.zeros((102,5))]
K2_C_factor_2025 = np.zeros((102,1))
K3_TOU_indu,K3_TOU_resi,K3_net = np.zeros((102,365,24)),np.zeros((102,365,24)),np.zeros((102,1))
for cc in range(102):
    city_name = City_statistic.index[cc]
    #resi_ratio = City_statistic.iloc[cc,1]/City_statistic.iloc[cc,0]
    _,K3_TOU_indu_i,K3_TOU_resi_i,K3_net_i,Carbon_F = Functions.TOU_period(city_name, PRICE_FILE)
    K3_TOU_indu[cc,:,:] = np.vstack([np.tile(K3_TOU_indu_i[month], (days, 1)) for month, days in enumerate(mon_d)])
    K3_TOU_resi[cc,:,:] = np.vstack([np.tile(K3_TOU_resi_i[month], (days, 1)) for month, days in enumerate(mon_d)])
    K3_net[cc,0] = K3_net_i*1e0
    K2_C_factor_2025[cc,0] = 1e-3*Carbon_F
    K2_C_factor[0][cc,:] = [1e-3*Carbon_F*0.74,1e-3*Carbon_F*0.51,1e-3*Carbon_F*0.33,1e-3*Carbon_F*0.17,1e-3*Carbon_F*0.17*0.06]
    K2_C_factor[1][cc,:] = [1e-3*Carbon_F*0.58,1e-3*Carbon_F*0.30,1e-3*Carbon_F*0.13,1e-3*Carbon_F*0.006,1e-3*Carbon_F*0.17*0]

# Clu_center,Clu_days = Functions.read_cluster_allyear()
# np.save('Clu_center.npy', Clu_center)
# np.save('Clu_days.npy', Clu_days)
Clu_center = np.load(FIG_INPUT_DIR / "Clu_center.npy")
Clu_days = np.load(FIG_INPUT_DIR / "Clu_days.npy")

###45/82;38/94;29/103;20/110;11/120;4/130;;;;;;45/82;35/110;20/120;9/130;0.5/140;0
###0.74;0.51;0.33;0.17;0.06;;;;;0.58;0.30;0.13;0.006;0

K3_pri_sto0 = [550,500,495,485,470]
K3_pri_sto  = [1e0*i*7.2 for i in K3_pri_sto0]
K3_main_pri_3 = [0.02,0.04,0.02]

EER_c, EER_h = 3.3, 3.0
city_north = pd.read_excel(CITY_NORTH_FILE, index_col=0)

K3_year_op = [25,25,25]

eta_ = 0.9
T_ESS = 4
epsi_ = [0.2,0.2,0.3,0.4,0.5]
Y = 5
D,T = 12,24
r = 0.08

#for cc in range(17):
#for cc in range(17,34):
#for cc in range(34,51):
#for cc in range(51,68):
#for cc in range(68,85):
for cc in range(args.city_start, args.city_end):
    city_name = City_statistic.index[cc]
    print(city_name)
    if city_north.loc[city_name, 'North'] == 0:
        heating_electrification_rate = [HEATING_ELECTRIFICATION_SOUTH for _ in range(Y)]
    else:
        heating_electrification_rate = [HEATING_ELECTRIFICATION_NORTH for _ in range(Y)]
    
    city_key = str(city_name).strip()
    if city_key not in main_case_data.index:
        raise KeyError(f'{city_name} is missing from {path_main_case}')
    main_case_city = main_case_data.loc[city_key]

    C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
    C2 = path_cap+'/Cap_roof_'+city_name+'.npy'

    G_type = np.load(ML_RESULTS_DIR / "Grid_type" / f"Grid_type_{city_name}.npy")
    Feas_read_sta = np.load(city_path+city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
    indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]

    #WWR = Statis_all.iloc[cc,-2]
    WWR = np.load(city_path+city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
    
    N_grid = len(indices_non_zero)

    cap_roof_raw = np.load(C2)[indices_non_zero]
    cap_facade_raw = np.load(C1)
    if cap_roof_raw.shape[0] != N_grid or cap_facade_raw.shape[0] != N_grid:
        raise ValueError(
            f'{city_name}: inconsistent grid counts: N_grid={N_grid}, '
            f'roof_capacity={cap_roof_raw.shape[0]}, facade_capacity={cap_facade_raw.shape[0]}'
        )

    roof_capacity_scale = _scale_to_saved_total(
        main_case_city['rpv_capacity_GW'], np.sum(cap_roof_raw), city_name, 'RPV capacity'
    )
    facade_capacity_scale = _scale_to_saved_total(
        main_case_city['fpv_capacity_GW'], np.sum(cap_facade_raw[:, -1]),
        city_name, 'FPV capacity'
    )
    cap_roof_main = cap_roof_raw * roof_capacity_scale
    cap_facade_main = cap_facade_raw * facade_capacity_scale

    roof_power_case1_raw = np.load(
        path_type+'1/N_P_roof_1_'+city_name+'.npy'
    )[indices_non_zero, :]
    facade_power_case1_raw = np.load(
        path_type+'1/N_P_facade_ideal_1_'+city_name+'.npy'
    )
    if roof_power_case1_raw.shape[0] != N_grid or facade_power_case1_raw.shape[0] != N_grid:
        raise ValueError(
            f'{city_name}: inconsistent power grid counts: N_grid={N_grid}, '
            f'roof_power={roof_power_case1_raw.shape[0]}, '
            f'facade_power={facade_power_case1_raw.shape[0]}'
        )

    roof_generation_scale = _scale_to_saved_total(
        main_case_city['rpv_generation_TWh'],
        np.sum(roof_power_case1_raw) / 1e6, city_name, 'RPV generation'
    )
    facade_generation_scale = _scale_to_saved_total(
        main_case_city['fpv_generation_TWh'],
        np.sum(facade_power_case1_raw) / 1e6, city_name, 'FPV generation'
    )

    print(
        f"Main case targets | RPV: {main_case_city['rpv_capacity_GW']:.4f} GW, "
        f"{main_case_city['rpv_generation_TWh']:.4f} TWh; "
        f"FPV: {main_case_city['fpv_capacity_GW']:.4f} GW, "
        f"{main_case_city['fpv_generation_TWh']:.4f} TWh"
    )

    K3_load_ele = np.zeros((N_grid,D_days,24))
    K3_load_heat = np.zeros((N_grid,D_days,24))
    D_load_cc = np.load(path_load+'Load_'+city_name+'_hybrid.npy')/1e0

    K3_D_roof = [np.zeros((N_grid,D_days,24)) for _ in range(6)]
    K3_D_fa_id = [np.zeros((N_grid,D_days,24)) for _ in range(6)]
    K3_D_fa_s = [[np.zeros((N_grid,D_days,24)) for _ in range(6)],[np.zeros((N_grid,D_days,24)) for _ in range(6)]]
    for mm in range(12):
        K3_load_ele[:,mm,:] = (D_load_cc[:,0,:] + D_load_cc[:,2,:]/EER_c)[:,Clu_center[cc,mm]*24:(Clu_center[cc,mm]+1)*24]
        K3_load_heat[:,mm,:] = (1/EER_h)*D_load_cc[:,1,:][:,Clu_center[cc,mm]*24:(Clu_center[cc,mm]+1)*24]

    K3_load = [K3_load_ele+K3_load_heat*heating_electrification_rate[iy] for iy in range(5)]

    A_Cap,A_Pow = Functions.regulate(city_name, ML_RESULTS_DIR)
    A_Cap = [np.asarray(x) * facade_capacity_scale for x in A_Cap]
    A_Pow = [np.asarray(x) * facade_generation_scale for x in A_Pow]
    
    for case in range(6):
        P1 = path_type+str(case+1)+'/N_P_facade_ideal_'+str(case+1)+'_'+city_name+'.npy'
        #P2 = path_type+str(case+1)+'/N_P_facade_'+str(case+1)+'_'+city_name+'.npy'
        P3 = path_type+str(case+1)+'/N_P_roof_'+str(case+1)+'_'+city_name+'.npy'
        roof_power_case = np.load(P3)[indices_non_zero, :] * roof_generation_scale
        facade_power_case = np.load(P1) * facade_generation_scale
        for i in range(12):
            Clu_i = Clu_center[cc,i]
            K3_D_roof[case][:,i,:] = (1/1e-3)*roof_power_case[:,Clu_i*24:(Clu_i+1)*24]
            K3_D_fa_id[case][:,i,:] = (1/1e-3)*facade_power_case[:,Clu_i*24:(Clu_i+1)*24]
            K3_D_fa_s[0][case][:,i,:] = (1/1e-3)*A_Pow[case][:,Clu_i*24:(Clu_i+1)*24]
            K3_D_fa_s[1][case][:,i,:] = (1/pri_change[case])*(1/1e-3)*A_Pow[case][:,Clu_i*24:(Clu_i+1)*24]
    
    R_Cap_r = np.zeros((N_grid, 2, Y))
    R_Cap_s = np.zeros((N_grid, 2, Y))
    
    R_Cap_f = np.zeros((N_grid, 3, Y))
    R_Ele = np.zeros((N_grid, Y))
    R_Pow_f = np.zeros((N_grid,Y, D, T))     #GWh
    
    R_Pow = np.zeros((N_grid, Y, D, T))
    R_Pow_r = np.zeros((N_grid, 2, Y, D, T))
    R_Pow_ch = np.zeros((N_grid,2,Y, D, T))     #GWh
    R_Pow_dis = np.zeros((N_grid,2,Y, D, T))     #GWh
    R_Pow_G = np.zeros((N_grid,2,Y, D, T))     #GWh
    R_Pow_Buy = np.zeros((N_grid,2,Y, D, T))     #GWh
    R_Pow_AB = np.zeros((N_grid,2,Y, D, T))     #GWh
    
    R_Car = np.zeros((N_grid,2,Y))
    R_Car_t = np.zeros((N_grid,2,Y,D,T))
    R_Cost = np.zeros((N_grid,2,Y))
    R_Cost_true = np.zeros((N_grid,2,Y))

    N_gg = np.where(G_type[:,0] != 888)[0]
    print('Non-Industrial ratio: {}'.format(len(N_gg)/G_type.shape[0]))

    for nn in N_gg:
        print(nn)
        #resi_ratio = City_statistic.iloc[cc,1]/City_statistic.iloc[cc,0]
        for type in [0,1]:
            # K3_faca_delta = [0.1,0.1,0.2,0.3,0.3]
            # K3_sto_delta = [0.3, 0.3, 0.3, 0.3, 0.3]
            K3_roof_delta = [1, 1, 1, 1, 1]
            K3_faca_delta = [1,1,1,1,1]
            K3_sto_delta = [1, 1, 1, 1, 1]
            K1_C_roof = cap_roof_main[nn]*1e6
            K1_C_roof_max = K1_C_roof
            if type == 0:
                K1_C_fa_id = 0*cap_facade_main[nn,-1]*1e6
                K1_C_fa_0 = np.array([0 for _ in range(6)])
            elif type == 1:
                K1_C_fa_id = cap_facade_main[nn,-1]*1e6
                K1_C_fa_0 = np.array([A_Cap[i][nn]*1e6 for i in range(6)])
            K3_C_sto = [0.5*K1_C_roof_max for _ in range(5)]
            K1_C_fa_s = [K1_C_fa_0,K1_C_fa_0/pri_change]
            for case1 in [0]:
                for case2 in [0]:
                    time1 = time.time()
                    mm = pyo.ConcreteModel()
                    mm.Y = pyo.RangeSet(0, Y-1)
                    
                    mm.cost = pyo.RangeSet(0, 3)
                    mm.D = pyo.RangeSet(0, D-1)
                    mm.T = pyo.RangeSet(0, T-1)
                    mm.T1 = pyo.RangeSet(0, T)
                    mm.V_CI_0 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_CI_1 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_CI_2 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.true = pyo.RangeSet(0, 2)
                    mm.V_CI_true = pyo.Var(mm.true, mm.Y,domain=pyo.NonNegativeReals)

                    mm.V_Car = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_Car_t = pyo.Var(mm.Y,mm.D,mm.T,domain=pyo.NonNegativeReals)
                    
                    mm.V_CO = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_CO_i = pyo.Var(mm.cost,mm.Y,domain=pyo.NonNegativeReals)
                    
                    mm.V_dUp_0 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_dUn_0 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_dUp_1 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_dUn_1 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_dUp_2 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_dUn_2 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_U_0 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_U_1 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)
                    mm.V_U_2 = pyo.Var(mm.Y,domain=pyo.NonNegativeReals)

                    mm.V_PB = pyo.Var(mm.Y,mm.D,mm.T,domain=pyo.NonNegativeReals)
                    mm.V_PG = pyo.Var(mm.Y,mm.D,mm.T,domain=pyo.NonNegativeReals)
                    mm.V_PAB = pyo.Var(mm.Y,mm.D,mm.T,domain=pyo.NonNegativeReals)
                    mm.V_Pr,mm.V_Pf = pyo.Var(mm.Y,mm.D,mm.T,domain=pyo.NonNegativeReals),pyo.Var(mm.Y,mm.D,mm.T,domain=pyo.NonNegativeReals)
                    mm.V_Pch,mm.V_Pdis = pyo.Var(mm.Y,mm.D,mm.T,domain=pyo.NonNegativeReals),pyo.Var(mm.Y,mm.D,mm.T,domain=pyo.NonNegativeReals)
                    mm.V_S = pyo.Var(mm.Y,mm.D,mm.T1,domain=pyo.NonNegativeReals)
                    mm.V_z1 = pyo.Var(mm.Y,domain=pyo.Binary,initialize=1)
                    mm.V_z2 = pyo.Var(mm.Y,domain=pyo.Binary,initialize=1)

                    def obj(mm):
                        return (sum((mm.V_CI_0[y]+mm.V_CI_1[y]+mm.V_CI_2[y])*(1+r)**(-5*(y+1)) for y in mm.Y) \
                                + sum(mm.V_CO[y]*(1+r)**(-5*(y+1)) for y in mm.Y))
                    mm.OBJ = pyo.Objective(rule = obj, sense=pyo.minimize)

                    def C_CI_0(mm,y):
                        cap_price = K1_pri_wall[case1][y+1]
                        cc1 = sum((1+r)**(-x) for x in range(1,25-5*y+1))*(r/(1-(1+r)**(-K3_year_op[0])))
                        cc2 = sum((1+r)**(-x) for x in range(1,6))
                        return mm.V_CI_0[y] == cc1*cap_price*mm.V_dUp_0[y] +\
                            cc2*K3_main_pri_3[0]*cap_price*mm.V_U_0[y]
                    mm.C_CI_0 = pyo.Constraint(mm.Y,rule = C_CI_0)

                    def C_CI_2(mm,y):
                        cap_price = K3_pri_sto[y]
                        cc1 = sum((1+r)**(-x) for x in range(1,25-5*y+1))*(r/(1-(1+r)**(-K3_year_op[2])))
                        cc2 = sum((1+r)**(-x) for x in range(1,6))
                        return mm.V_CI_2[y] == cc1*cap_price*mm.V_dUp_2[y] +\
                            cc2*K3_main_pri_3[2]*cap_price*mm.V_U_2[y]
                    mm.C_CI_2 = pyo.Constraint(mm.Y,rule = C_CI_2)

                    def C_CI_1(mm,y):
                        cap_price = (K1_pri_wall[case1][y+1]*(1-WWR[nn,0])+K1_pri_win[case1][y+1]*WWR[nn,0])
                        cc1 = sum((1+r)**(-x) for x in range(1,25-5*y+1))*(r/(1-(1+r)**(-K3_year_op[1])))
                        cc2 = sum((1+r)**(-x) for x in range(1,6))
                        return mm.V_CI_1[y] == cc1*cap_price*mm.V_dUp_1[y] +\
                            cc2*K3_main_pri_3[1]*cap_price*mm.V_U_1[y]
                    mm.C_CI_1 = pyo.Constraint(mm.Y,rule = C_CI_1)

                    def C_CI_true(mm,cost,y):
                        cc2 = sum((1+r)**(-x) for x in range(1,6))
                        if cost == 0:
                            cap_price = K1_pri_wall[case1][y+1]
                            cc1_true = sum((1+r)**(-x) for x in range(1,6))*(r/(1-(1+r)**(-K3_year_op[0])))
                            return mm.V_CI_true[cost, y] == cc1_true*cap_price*mm.V_U_0[y] +\
                                cc2*K3_main_pri_3[0]*cap_price*mm.V_U_0[y]
                        elif cost == 1:
                            cap_price = (K1_pri_wall[case1][y+1]*(1-WWR[nn,0])+K1_pri_win[case1][y+1]*WWR[nn,0])
                            cc1_true = sum((1+r)**(-x) for x in range(1,6))*(r/(1-(1+r)**(-K3_year_op[1])))
                            return mm.V_CI_true[cost, y] == cc1_true*cap_price*mm.V_U_1[y] +\
                                cc2*K3_main_pri_3[1]*cap_price*mm.V_U_1[y]
                        elif cost == 2:
                            cap_price = K3_pri_sto[y]
                            cc1_true = sum((1+r)**(-x) for x in range(1,6))*(r/(1-(1+r)**(-K3_year_op[2])))
                            return mm.V_CI_true[cost, y] == cc1_true*cap_price*mm.V_U_2[y] +\
                                cc2*K3_main_pri_3[2]*cap_price*mm.V_U_2[y]                       
                    mm.C_CI_true = pyo.Constraint(mm.true,mm.Y,rule = C_CI_true)

                    def C_CO(mm,y):
                        return mm.V_CO[y] == mm.V_CO_i[0,y] + mm.V_CO_i[1,y] - mm.V_CO_i[2,y] #+ mm.V_CO_i[3,y]
                    mm.C_CO = pyo.Constraint(mm.Y,rule = C_CO)

                    def C_CO_0(mm,y):
                        # if G_type[nn,0] == 101:
                        #     price = K3_TOU_resi
                        # else:
                        #     price = K3_TOU_indu
                        price = 0.8*K3_TOU_resi + 0.2*K3_TOU_indu
                        return mm.V_CO_i[0,y] == sum(5*Clu_days[cc,d]*price[cc,Clu_center[cc,d],t]*mm.V_PB[y,d,t] for t in mm.T for d in mm.D)
                    mm.C_CO_0 = pyo.Constraint(mm.Y,rule = C_CO_0)
                    def C_CO_1(mm,y):
                        return mm.V_CO_i[1,y] == sum(5*Clu_days[cc,d]*K2_C_pri[case2][y]*K2_C_factor[case2][cc,y]*mm.V_PB[y,d,t] for t in mm.T for d in mm.D)
                    mm.C_CO_1 = pyo.Constraint(mm.Y,rule = C_CO_1)
                    def C_CO_2(mm,y):
                        return mm.V_CO_i[2,y] == sum(5*Clu_days[cc,d]*K3_net[cc,0]*mm.V_PG[y,d,t] for t in mm.T for d in mm.D)
                    mm.C_CO_2 = pyo.Constraint(mm.Y,rule = C_CO_2)
                    #     return mm.V_CO_i[3,y] == sum(5*mon_d[d]*K3_net[cc,0]*mm.V_PAB[y,d,t] for t in mm.T for d in mm.D)
                    #mm.C_CO_3 = pyo.Constraint(mm.Y,rule = C_CO_3)
                    
                    def C_Car(mm,y):
                        return mm.V_Car[y] == sum(5*Clu_days[cc,d]*K2_C_factor[case2][cc,y]*mm.V_PB[y,d,t] for t in mm.T for d in mm.D)
                    mm.C_Car = pyo.Constraint(mm.Y,rule = C_Car)
                    def C_Car_t(mm,y,d,t):
                        return mm.V_Car_t[y,d,t] == K2_C_factor[case2][cc,y]*mm.V_PB[y,d,t]
                    mm.C_Car_t = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Car_t)
                    

                    #K1_C_roof_sum = sum(np.array(pd.read_excel(path+'City_Cap.xlsx', index_col=0).iloc[:,0]))
                    #U_ini = [cap_ini*K1_C_roof/K1_C_roof_sum,0,0]
                    U_ini = [0,0,0]
                    def C_Plan1_0(mm,y):
                        if y == 0:
                            return mm.V_U_0[y] == U_ini[0] + mm.V_dUp_0[y]-mm.V_dUn_0[y]
                        else:
                            return mm.V_U_0[y] == mm.V_U_0[y-1]+mm.V_dUp_0[y]-mm.V_dUn_0[y]
                    mm.C_Plan1_0 = pyo.Constraint(mm.Y,rule = C_Plan1_0)
                    def C_Plan1_1(mm,y):
                        if y == 0:
                            return mm.V_U_1[y] == U_ini[1] + mm.V_dUp_1[y]-mm.V_dUn_1[y]
                        else:
                            return mm.V_U_1[y] == mm.V_U_1[y-1]+mm.V_dUp_1[y]-mm.V_dUn_1[y]
                    mm.C_Plan1_1 = pyo.Constraint(mm.Y,rule = C_Plan1_1)
                    def C_Plan1_2(mm,y):
                        if y == 0:
                            return mm.V_U_2[y] == U_ini[2] + mm.V_dUp_2[y]-mm.V_dUn_2[y]
                        else:
                            return mm.V_U_2[y] == mm.V_U_2[y-1]+mm.V_dUp_2[y]-mm.V_dUn_2[y]
                    mm.C_Plan1_2 = pyo.Constraint(mm.Y,rule = C_Plan1_2)                

                    def C_Plan2_0(mm,y):
                        return mm.V_U_0[y] <= K1_C_roof*K1_roof[case1][y]
                    mm.C_Plan2_0 = pyo.Constraint(mm.Y,rule = C_Plan2_0)
                    def C_Plan2_1(mm,y):
                        return mm.V_U_1[y] <= K1_C_fa_id
                    mm.C_Plan2_1 = pyo.Constraint(mm.Y,rule = C_Plan2_1)
                    def C_Plan2_2(mm,y):
                        return mm.V_U_2[y] <= K3_C_sto[y]
                    mm.C_Plan2_2 = pyo.Constraint(mm.Y,rule = C_Plan2_2)

                    def C_Plan3_0(mm,y):
                        return mm.V_dUp_0[y] <= K1_C_roof*K1_roof[case1][y]*K3_roof_delta[y]
                    mm.C_Plan3_0 = pyo.Constraint(mm.Y,rule = C_Plan3_0)
                    def C_Plan3_1(mm,y):
                        return mm.V_dUp_1[y] <= K1_C_fa_id*K3_faca_delta[y]
                    mm.C_Plan3_1 = pyo.Constraint(mm.Y,rule = C_Plan3_1)
                    def C_Plan3_2(mm,y):
                        return mm.V_dUp_2[y] <= K3_C_sto[y]*K3_sto_delta[y]
                    mm.C_Plan3_2 = pyo.Constraint(mm.Y,rule = C_Plan3_2)
                    def C_Plan4_0(mm,y):
                        return mm.V_dUn_0[y] <= K1_C_roof*K1_roof[case1][y]*K3_roof_delta[y]
                    mm.C_Plan4_0 = pyo.Constraint(mm.Y,rule = C_Plan4_0)
                    def C_Plan4_1(mm,y):
                        return mm.V_dUn_1[y] <= K1_C_fa_id*K3_faca_delta[y]
                    mm.C_Plan4_1 = pyo.Constraint(mm.Y,rule = C_Plan4_1)
                    def C_Plan4_2(mm,y):
                        return mm.V_dUn_2[y] <= K3_C_sto[y]*K3_sto_delta[y]
                    mm.C_Plan4_2 = pyo.Constraint(mm.Y,rule = C_Plan4_2)
                    
                    def C_Op_load(mm,y,d,t):
                        #K3_load = (K3_load_ele+K3_load_heat*heating_electrification_rate[y])
                        Load = K3_load[y][nn,:,:]*K3_Load[y]
                        return Load[d,t] == mm.V_Pr[y,d,t] + mm.V_Pf[y,d,t] + mm.V_PB[y,d,t] - mm.V_PAB[y,d,t] - mm.V_PG[y,d,t] + mm.V_Pdis[y,d,t] - mm.V_Pch[y,d,t]
                    mm.C_Op_load = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_load)

                    def C_Op_PV_r(mm,y,d,t):
                        U_max = K1_C_roof*K1_roof[case1][y]
                        P_max = K3_D_roof[y+1][nn,:,:]*K1_roof[case1][y]
                        return mm.V_Pr[y,d,t] == P_max[d,t]*mm.V_U_0[y]/U_max
                    mm.C_Op_PV_r = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_PV_r)
                    
                    if type == 1:
                        # def C_Op_PV_ff(mm,y,d,t):
                        #     U_max = K1_C_fa_id
                        #     P_max = K3_D_fa_id[y+1][nn,:,:]
                        #     return mm.V_Pf[y,d,t] == P_max[d,t]*mm.V_U_1[y]/U_max
                        # mm.C_Op_PV_ff = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_PV_ff)
                        M_b = 1e8
                        def C_Op_PV_f1(mm,y):
                            return mm.V_z1[y] + mm.V_z2[y] == 1
                        mm.C_Op_PV_f1 = pyo.Constraint(mm.Y,rule = C_Op_PV_f1)
                        def C_Op_PV_f2(mm,y):
                            return mm.V_U_1[y] >= 0*mm.V_z1[y]
                        mm.C_Op_PV_f2 = pyo.Constraint(mm.Y,rule = C_Op_PV_f2)
                        def C_Op_PV_f3(mm,y):
                            U_s_max = K1_C_fa_s[case1][y+1]
                            return mm.V_U_1[y] <= U_s_max*mm.V_z1[y] + M_b*(1-mm.V_z1[y])
                        mm.C_Op_PV_f3 = pyo.Constraint(mm.Y,rule = C_Op_PV_f3)
                        def C_Op_PV_f4(mm,y):
                            U_s_max = K1_C_fa_s[case1][y+1]
                            return mm.V_U_1[y] >= U_s_max*mm.V_z2[y]
                        mm.C_Op_PV_f4 = pyo.Constraint(mm.Y,rule = C_Op_PV_f4)
                        def C_Op_PV_f5(mm,y):
                            U_max = K1_C_fa_id
                            return mm.V_U_1[y] <= U_max*mm.V_z2[y] + M_b*(1-mm.V_z2[y])
                        mm.C_Op_PV_f5 = pyo.Constraint(mm.Y,rule = C_Op_PV_f5)
                        def C_Op_PV_f6(mm,y,d,t):
                            U_max = K1_C_fa_id
                            U_s_max = K1_C_fa_s[case1][y+1]
                            P_max = K3_D_fa_id[y+1][nn,:,:]
                            P_s_max = K3_D_fa_s[case1][y+1][nn,:,:]
                            return mm.V_Pf[y,d,t] == (P_s_max[d,t]/U_s_max)*mm.V_U_1[y]*mm.V_z1[y] \
                                + (P_s_max[d,t]+(P_max[d,t]-P_s_max[d,t])*(mm.V_U_1[y]-U_s_max)/(U_max-U_s_max))*mm.V_z2[y]
                        mm.C_Op_PV_f6 = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_PV_f6)
                    else:
                        def C_Op_PV_f7(mm,y,d,t):
                            return mm.V_Pf[y,d,t] == 0
                        mm.C_Op_PV_f7 = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_PV_f7)
                    
                    def C_Op_s1(mm,y,d,t):
                        return mm.V_S[y,d,t+1] == mm.V_S[y,d,t] + mm.V_Pch[y,d,t]*eta_ - mm.V_Pdis[y,d,t]/eta_
                    mm.C_Op_s1 = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_s1)
                    def C_Op_s2(mm,y,d,t):
                        return mm.V_Pch[y,d,t] <= mm.V_U_2[y]
                    mm.C_Op_s2 = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_s2)
                    def C_Op_s3(mm,y,d,t):
                        return mm.V_Pdis[y,d,t] <= mm.V_U_2[y]
                    mm.C_Op_s3 = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_s3)
                    def C_Op_s4(mm,y,d,t1):
                        return mm.V_S[y,d,t1] <= mm.V_U_2[y]*T_ESS
                    mm.C_Op_s4 = pyo.Constraint(mm.Y,mm.D,mm.T1,rule = C_Op_s4)
                    def C_Op_s5(mm,y,d):
                        return mm.V_S[y,d,0] == mm.V_S[y,d,24]
                    mm.C_Op_s5 = pyo.Constraint(mm.Y,mm.D,rule = C_Op_s5)
                    def C_Op_s6(mm,y,d):
                        return mm.V_S[y,d,0] == 0.5*mm.V_U_2[y]*T_ESS
                    mm.C_Op_s6 = pyo.Constraint(mm.Y,mm.D,rule = C_Op_s6)

                    def C_Op_PG(mm,y,d,t):
                        return mm.V_PG[y,d,t] <= epsi_[y]*(mm.V_PG[y,d,t]+mm.V_PAB[y,d,t])
                    mm.C_Op_PG = pyo.Constraint(mm.Y,mm.D,mm.T,rule = C_Op_PG)

                    opt = pyo.SolverFactory(args.solver)
                    instance = mm.create_instance()
                    results = opt.solve(instance)  #, tee=True)
                    
                    time2 = time.time()
                    print(time2-time1)

            # print(pyo.value(instance.OBJ))
            if type == 1:
            #     print(pyo.value(instance.V_U_0[4])/(K1_C_roof*K1_roof[case1][4]))
                print(pyo.value(instance.V_U_1[4])/K1_C_fa_s[case1][5])
            
            for yy in range(5):
                #print([pyo.value(instance.V_U_0[yy]),pyo.value(instance.V_U_1[yy]),pyo.value(instance.V_U_2[yy])])
                R_Cap_r[nn,type,yy],R_Cap_s[nn,type,yy] = pyo.value(instance.V_U_0[yy]),pyo.value(instance.V_U_2[yy])
                if type == 1:
                    R_Cap_f[nn,0,yy] = pyo.value(instance.V_U_1[yy])
                    R_Cap_f[nn,1,yy] = K1_C_fa_s[case1][yy+1]
                    R_Cap_f[nn,2,yy] = K1_C_fa_id
                    R_Ele[nn,yy] = sum(5*Clu_days[cc,d]*(K3_load[yy][nn,d,t]*K3_Load[yy]) for d in range(D) for t in range(24))
                    R_Pow[nn,yy,:,:] = K3_load[yy][nn,:,:]*K3_Load[yy]
                    R_Pow_f[nn,yy,:,:] = np.array(pyo.value(instance.V_Pf[yy,:,:])).reshape(D,T)
                R_Pow_ch[nn,type,yy,:,:],R_Pow_dis[nn,type,yy,:,:] = np.array(pyo.value(instance.V_Pch[yy,:,:])).reshape(D,T),np.array(pyo.value(instance.V_Pdis[yy,:,:])).reshape(D,T)
                R_Pow_G[nn,type,yy,:,:],R_Pow_Buy[nn,type,yy,:,:],R_Pow_AB[nn,type,yy,:,:] = np.array(pyo.value(instance.V_PG[yy,:,:])).reshape(D,T),np.array(pyo.value(instance.V_PB[yy,:,:])).reshape(D,T),np.array(pyo.value(instance.V_PAB[yy,:,:])).reshape(D,T)
                R_Pow_r[nn,type,yy,:,:] = np.array(pyo.value(instance.V_Pr[yy,:,:])).reshape(D,T)
                R_Car[nn,type,yy] = pyo.value(instance.V_Car[yy])
                R_Car_t[nn,type,yy,:,:] = np.array(pyo.value(instance.V_Car_t[yy,:,:])).reshape(D,T)         
                R_Cost[nn,type,yy] = (pyo.value(instance.V_CI_0[yy])+pyo.value(instance.V_CI_1[yy])+pyo.value(instance.V_CI_2[yy]) + pyo.value(instance.V_CO[yy]))/1e10
                R_Cost_true[nn,type,yy] = (pyo.value(instance.V_CI_true[0, yy]) + pyo.value(instance.V_CI_true[1, yy]) + pyo.value(instance.V_CI_true[2, yy]) + pyo.value(instance.V_CO[yy]))/1e10

    result_suffix = '_hybrid_n.mat' if cc in ABNORMAL_VOLUME_CITY_INDICES else '_hybrid.mat'
    savemat(path_opt+city_name+result_suffix,{'R_Cap_r': R_Cap_r,'R_Cap_s': R_Cap_s,'R_Cap_f': R_Cap_f,'R_Ele': R_Ele,'R_Pow': R_Pow,\
                          'R_Pow_f': R_Pow_f,'R_Pow_ch': R_Pow_ch,'R_Pow_dis':R_Pow_dis,'R_Pow_G':R_Pow_G,\
                            'R_Pow_r':R_Pow_r, 'R_Pow_Buy':R_Pow_Buy,'R_Pow_AB':R_Pow_AB,'R_Car':R_Car,'R_Car_t':R_Car_t,'R_Cost':R_Cost,'R_Cost_true':R_Cost_true})


