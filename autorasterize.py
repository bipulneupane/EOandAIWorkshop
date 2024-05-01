#!/usr/bin/env python3
#  auto-rasterize
#
#     Huriel Reichel - huriel.ruan@gmail.com
#     Nils Hamel - nils.hamel@bluewin.ch
#     Copyright (c) 2020 Republic and Canton of Geneva
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from osgeo import gdal
from osgeo import ogr


def _rasterize(input, output, xmin, ymin, xmax, ymax, pixel):
    # input = "Datasets\\2020 Building Footprints\\shapefile_new.shp" 
    # output = "testt2.png"  
    # xmin = 314852
    # ymax = 5817273 
    # xmax = 323406
    # ymin = 5808587 
    # pixel = 1

    # Open the data source
    orig_data_source = ogr.Open(input)
    # Make a copy of the layer's data source because we'll need to 
    # modify its attributes table
    source_ds = ogr.GetDriverByName("Memory").CopyDataSource(
            orig_data_source, "")
    source_layer = source_ds.GetLayer(0)
    source_srs = source_layer.GetSpatialRef() 
    

    # Create the destination data source
    x_res = int((xmax - xmin) / pixel)
    y_res = int((ymax - ymin) / pixel)      
    
    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res,
            y_res, 3, gdal.GDT_Byte)

    target_ds.SetGeoTransform((
            xmin, pixel, 0,
            ymax, 0, -pixel,
        ))
    target_ds.GetRasterBand(1).SetNoDataValue(0)
    target_ds.GetRasterBand(2).SetNoDataValue(0)
    target_ds.GetRasterBand(3).SetNoDataValue(0)
    if source_srs:
        # Make the target raster have the same projection as the source
        target_ds.SetProjection(source_srs.ExportToWkt())
    else:
        # Source has no projection (needs GDAL >= 1.7.0 to work)
        target_ds.SetProjection('LOCAL_CS["arbitrary"]')
    # Rasterize
    err = gdal.RasterizeLayer(target_ds, (3, 2, 1), source_layer,
            burn_values=(255, 255, 255))
    if err != 0:
        raise Exception("error rasterizing layer: %s" % err)

def rasterize(input, output, xmin, ymin, xmax, ymax, pixel):
    print('rasterizing based on reference raster')
    _rasterize(input, output, xmin, ymin, xmax, ymax, pixel)
