# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:58:47 2024

@author: 24986
"""

import numpy as np
import pandas as pd

City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', 'Hefei', 'Huhehaote', 'Jinan', 'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

for cc in City_name:
    print(cc)
    df_read = np.array(pd.read_excel(cc+'\\'+cc+'_sample_DEM_coordinate'+'.xlsx'))
    for i in range(1,65):
        print(i)
        [xmin,xmax,ymin,ymax] = df_read[i-1,:]
        xmax = xmin+2199
        ymin = ymax-2199
        processing.run("gdal:cliprasterbyextent", {'INPUT':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_canopy_3m.TIF','PROJWIN':str(xmin)+','+str(xmax)+','+str(ymin)+','+str(ymax)+' [EPSG:3857]','OVERCRS':False,'NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_canopy_duiqi_Fin.tif'})
        print('finish')



