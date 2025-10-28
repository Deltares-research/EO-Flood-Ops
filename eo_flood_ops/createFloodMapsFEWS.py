# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 2025

@author: haag (arjen.haag@deltares.nl)
"""


# required libraries
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio import features
#import xarray as xr
#import rioxarray as xrio
import matplotlib.pyplot as plt

FILE_POINTS_DATA = 'Export_FloodMapPoints_Bosna_201902.csv'
FILE_POINTS_LOCS = 'Selectie_points_Bosna_edit.shp'
FILE_POLYS = 'Polygon_Bosna_ID.shp'
FILE_DEM = 'bosna_FABDEM.tif'
#FILE_DEM = 'bosna_copdem.tif'

time0 = '2019-02-04 16:00:00'
#TIMES = ['2019-02-04 15:00:00', '2019-02-04 16:00:00', '2019-02-04 17:00:00']

FILE_OUT = FILE_DEM.replace('.tif','_') + time0.replace('-','').replace(':','').replace(' ','T') + '.tif'

#TEST_LOC = 'Floodmap_HEC_RAS_BOSNA_217' # near Doboj gauging station
TEST_LOC = 'Floodmap_HEC_RAS_BOSNA_218' # near Doboj gauging station
#TEST_LOC = 'Floodmap_HEC_RAS_BOSNA_289' # near Samac gauging station
#TEST_LOC = 'Floodmap_HEC_RAS_BOSNA_290' # near Samac gauging station

gdf_points = gpd.read_file(FILE_POINTS_LOCS)
#gdf_points.plot()
#plt.show()

gdf_polys = gpd.read_file(FILE_POLYS)
#gdf_polys.plot()
#plt.show()

#fig = plt.figure(figsize=(12, 8))
#ax = fig.add_subplot()
#gdf_polys.plot(ax=ax, color="white", edgecolor="black")#, alpha=0.5)
#gdf_points.plot(ax=ax, color="blue", alpha=0.5)
#plt.show()
#import pdb; pdb.set_trace()

df = pd.read_csv(FILE_POINTS_DATA, skiprows=[1])
#df['GMT'] = pd.to_datetime(df['GMT'])
#plt.plot(df['GMT'], df[TEST_LOC])
#plt.show()

cols = df.columns.values
new_cols = [item.split('_point')[0] for item in cols]
df.columns = new_cols

#df.transpose()
#df.set_index('GMT').transpose()
#df.set_index('GMT').transpose().reset_index().rename(columns={'index':'ID_1'})

#df.loc[df['GMT']==time0].drop('GMT', axis=1).transpose()
#gdf_polys_time0 = 

gdf_polys_data = gdf_polys.merge(df.set_index('GMT').transpose().reset_index().rename(columns={'index':'ID_1'}), on="ID_1")

gdf_polys_data_time0 = gdf_polys_data[['ID_1', 'geometry', time0]]
#gdf_polys_data_time0.plot(column=time0, cmap='Spectral', legend=True)
#plt.show()

#import pdb; pdb.set_trace()

#DEM = xrio.open_rasterio(FILE_DEM)
DEM = rio.open(FILE_DEM)

meta = DEM.meta.copy()
meta.update(compress='lzw')

with rio.open(FILE_OUT, 'w+', **meta) as out:
    out_arr = out.read(1)

    # this is where we create a generator of geom, value pairs to use in rasterizing
    shapes = ((geom,value) for geom, value in zip(gdf_polys_data_time0.geometry, gdf_polys_data_time0[time0]))

    waterlevels = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
    floodmap = waterlevels - DEM.read(1)
    floodmap = np.maximum(floodmap, 0)
    
    plt.imshow(np.maximum(floodmap,0), cmap='Blues')
    plt.show()
    import pdb; pdb.set_trace()
    
    out.write_band(1, floodmap)

import pdb; pdb.set_trace()
