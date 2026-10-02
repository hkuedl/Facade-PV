# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:45:43 2024

@author: 24986
"""


City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', 'Hefei', 'Huhehaote', 'Jinan', 'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

for cc in City_name[5:]:
    print(cc)
    for i in range(1,65):
        print(i)
        processing.run("umep:Urban Geometry: Wall Height and Aspect", {'INPUT':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_DSM.TIF','INPUT_LIMIT':3,'OUTPUT_HEIGHT':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_wall_hei.tif','OUTPUT_ASPECT':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_wall_asp.tif'})
        print('finish')
