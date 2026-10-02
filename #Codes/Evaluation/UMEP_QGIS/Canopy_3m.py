# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:54:36 2024

@author: 24986
"""

City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', 'Hefei', 'Huhehaote', 'Jinan', 'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

for cc in City_name:
    print(cc)
    for i in range(1,65):
        print(i)
        processing.run("native:alignrasters", {'LAYERS':[{'inputFile': cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_canopy.TIF','outputFile': cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_canopy_3m.tif','resampleMethod': 0,'rescale': False}],'REFERENCE_LAYER':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_DSM.TIF','CRS':QgsCoordinateReferenceSystem('EPSG:3857'),'CELL_SIZE_X':None,'CELL_SIZE_Y':None,'GRID_OFFSET_X':None,'GRID_OFFSET_Y':None,'EXTENT':None})
        print('finish')
