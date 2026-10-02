
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
        processing.run("umep:Spatial Data: DSM Generator", {'INPUT_DEM':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_DEM.TIF','INPUT_POLYGONLAYER':cc+'/'+cc+'_'+str(i)+'/'+str(i)+'.shp','INPUT_FIELD':'height','USE_OSM':False,'BUILDING_LEVEL':3.1,'EXTENT':str(xmin)+','+str(xmax)+','+str(ymin)+','+str(ymax)+' [EPSG:3857]','PIXEL_RESOLUTION':3,'OUTPUT_DSM':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_DSM.tif'})
        print('finish')
