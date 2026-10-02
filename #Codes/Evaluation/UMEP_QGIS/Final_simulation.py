# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:13:34 2024

@author: 24986
"""

import time

City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', 'Hefei', 'Huhehaote', 'Jinan', 'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

day_list = ['0115','0215','0315','0415','0515','0615','0715','0815','0915','1015','1115','1215']

time_list = ['08','10','12','14','16','18']

weather_type = '_weather_TMY_'
#Weather_type = '_weather_2023_'

for cc in City_name[25:26]:
    print(cc)
    for i in range(59,65):
        for j_day in day_list:
            for j_time in time_list:
                print(i)
                print('period: '+j_day+'_'+j_time)
                start_time = time.time()
                processing.run("umep:Solar Radiation: Solar Energy of Builing Envelopes (SEBE)", {'INPUT_DSM':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_DSM.TIF','INPUT_CDSM':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_canopy_duiqi.TIF','TRANS_VEG':3,'INPUT_TDSM':None,'INPUT_THEIGHT':25,'INPUT_HEIGHT':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_wall_hei.TIF','INPUT_ASPECT':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_wall_asp.TIF','ALBEDO':0.15,'INPUTMET':cc+'\\'+cc+'_'+str(i)+'\\'+cc+'_'+str(i)+'_'+j_day+'_'+j_time+'\\'+cc+'_'+str(i)+weather_type+j_day+'_'+j_time+'.txt','ONLYGLOBAL':False,'UTC':8,'SAVESKYIRR':False,'IRR_FILE':'TEMPORARY_OUTPUT','OUTPUT_DIR':cc+'\\'+cc+'_'+str(i)+'\\'+cc+'_'+str(i)+'_'+j_day+'_'+j_time})
                end_time = time.time()
                print('finish with time {}'.format(end_time-start_time))
                
