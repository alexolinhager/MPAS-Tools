#!/usr/bin/env python

import xarray as xr
import os
import numpy as np
import subprocess
from argparse import ArgumentParser
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from mpas_tools.logging import check_call
import tifffile as tiff

class validateWithSpec :

    def __init__(self):
        print("Gathering Information ...")
        parser = ArgumentParser(prog='validate_mali_with_spec.py', 
                                description='Calculates a validation score for mali subglacial hydrology simulations using radar specularity content')
        parser.add_argument("--maliFile", dest="maliFile", required=True, help="MALI output file to validate")
        parser.add_argument("--specTiff", dest="specTiff", required=True, help='Tiff file containing specularity content classified according to Young 2016:'
                            'Values of 0 represent specularity content below 20%; values of 3.3 represent specularity content above 20% and energy 1' 
                            'microsecond below the bed 15 dB lower than the bed echo, and values of 6.7 represent specularity content above 20% and energy' 
                            '1 microsecond below the bed 15 dB within than the bed echo.')
        parser.add_argument("--compRes", dest="compRes", type=float, help="Grid resolution on which to interpolate and validate (meters)", default="5000")
        parser.add_argument("--ntasks", dest="ntasks", type=str, help="Number of processors to use with ESMF_RegridWeightGen", default='128')
        args = parser.parse_args()
        self.options = args

    def create_dest_scrip_file(self):
        res = self.options.compRes

        ds_mali = xr.open_dataset(self.options.maliFile, decode_times=False, decode_cf=False)
        xCell = ds_mali['xCell'][:].values
        yCell = ds_mali['yCell'][:].values

        xmin = np.min(xCell)
        xmax = np.max(xCell)
        ymin = np.min(yCell)
        ymax = np.max(yCell)

        # Define cell centers 
        x_center = np.arange(xmin, xmax, res)
        y_center = np.arange(ymin, ymax, res)

        ds_scrip = xr.Dataset()
        x = xr.DataArray(x_center.astype('float64'), dims=("nx"))
        y = xr.DataArray(y_center.astype('float64'), dims=("ny"))
        ds_scrip['x'] = x
        ds_scrip['y'] = y
        ds_scrip.to_netcdf('dest.scrip.initial.tmp.nc')
        ds_scrip.close()
        subprocess.run(["create_scrip_file_from_planar_rectangular_grid", "-i", "dest.scrip.initial.tmp.nc",
                        "-s", "dest.scrip.tmp.nc", "-p", "ais-bedmap2", "-r", "2"])
        
        # Convert to radians from degrees
        ds_scrip = xr.open_dataset('dest.scrip.tmp.nc')
        center_lon = ds_scrip['grid_center_lon']
        center_lat = ds_scrip['grid_center_lat']
        corner_lon = ds_scrip['grid_corner_lon']
        corner_lat = ds_scrip['grid_corner_lat']
        
        mask = center_lon < 0
        center_lon[mask] = center_lon[mask] + 360
        corner_lon[mask] = corner_lon[mask] + 360

        center_lon = center_lon * 2*np.pi / 360
        center_lat = center_lat * 2*np.pi / 360
        corner_lon = corner_lon * 2*np.pi / 360
        corner_lat = corner_lat * 2*np.pi / 360

        ds_scrip['center_lon'] = center_lon
        ds_scrip['center_lat'] = center_lat
        ds_scrip['corner_lon'] = corner_lon
        ds_scrip['corner_lat'] = corner_lat

        ds_scrip.to_netcdf('dest.scrip.nc')
        ds_scrip.close()

    def interpolate_mali(self):
        scrip_from_mpas(self.options.maliFile, 'mali.scrip.nc')
        args = ['srun',
                '-n', self.options.ntasks, 'ESMF_RegridWeightGen',
                '--source', 'mali.scrip.nc',
                '--destination', 'dest.scrip.nc',
                '--weight', 'mali_to_dest.nc',
                '--method', 'bilinear',
                '--netcdf4',
                "--dst_regional", "--src_regional", '--ignore_unmapped']
        check_call(args)

        args_remap = ["ncremap",
                "-i", self.options.maliFile,
                "-o", 'mali_remapped.nc',
                "-m", 'mali_to_dest.nc',
                "-v", 'waterThickness']
        check_call(args_remap)    

    def interpolate_spec(self):
        # Open geoTiff and convert to netcdf
        with tiff.TiffFile("self.options.specTiff") as tif:
            page = tif.pages[0]
            specData = page.asarray()
            scale = page.tags["ModelPixelScaleTag"].value
            tiepoint = page.tags["ModelTiepointTag"].value

            pixelWidth = scale[0]
            pixelHeight = scale[1]

            i,j,k,x0,y0,z0 = tiepoint

            rows,cols = specData.shape

            x = x0 + np.arange(cols) * pixelWidth
            y = y0 - np.arange(rows) * pixelHeight

        specData[np.isnan(specData)] = 0
        
        ds_spec = xr.Dataset()
        x = xr.DataArray(x_center.astype('float64'), dims=("nx"))
        y = xr.DataArray(y_center.astype('float64'), dims=("ny"))
        ds_spec['x'] = x
        ds_spec['y'] = y
        ds_spec['spec'] = xr.DataArray(np.array(specData).astype('float64'), dims=("nx","ny"))
        specMask = np.zeros(np.array(spec).shape)
        specMask[spec == 3.3] = 1
        specMask[spec == 6.7] = 2
        ds_spec['specMask'] = xr.DataArray(specMask.astype('int32'), dims=("nx","ny"))
        ds_scrip.to_netcdf('spec.nc')
        ds_scrip.close()

        # ais-bedmap2 is same as EPSG:3031 (Antarctic Polar Stereographic)
        subprocess.run(["create_scrip_file_from_planar_rectangular_grid", "-i", "spec.nc",
                        "-s", "spec.scrip.tmp.nc", "-p", "ais-bedmap2", "-r", "2"])
        
        # Convert to radians from degrees
        ds_scrip = xr.open_dataset('spec.scrip.tmp.nc')
        center_lon = ds_scrip['grid_center_lon']
        center_lat = ds_scrip['grid_center_lat']
        corner_lon = ds_scrip['grid_corner_lon']
        corner_lat = ds_scrip['grid_corner_lat']
        
        mask = center_lon < 0
        center_lon[mask] = center_lon[mask] + 360
        corner_lon[mask] = corner_lon[mask] + 360

        center_lon = center_lon * 2*np.pi / 360
        center_lat = center_lat * 2*np.pi / 360
        corner_lon = corner_lon * 2*np.pi / 360
        corner_lat = corner_lat * 2*np.pi / 360

        ds_scrip['center_lon'] = center_lon
        ds_scrip['center_lat'] = center_lat
        ds_scrip['corner_lon'] = corner_lon
        ds_scrip['corner_lat'] = corner_lat

        ds_scrip.to_netcdf('spec.scrip.nc')
        ds_scrip.close()

        args = ['srun',
                '-n', self.options.ntasks, 'ESMF_RegridWeightGen',
                '--source', 'spec.scrip.nc',
                '--destination', 'dest.scrip.nc',
                '--weight', 'spec_to_dest.nc',
                '--method', 'conserve',
                '--netcdf4',
                "--dst_regional", "--src_regional", '--ignore_unmapped']
        check_call(args)

        args_remap = ["ncremap",
                "-i", 'spec.nc',
                "-o", 'spec_remapped.nc',
                "-m", 'spec_to_dest.nc']
        check_call(args_remap)    

    #def validate(self):
        #ds_spec = xr.open_dataset('spec_remapped.nc')
        #ds_mali = xr.open_dataset('mali_remapped.nc')

        #spec = ds_spec['spec'][:].values
        #waterThickness = ds_mali['waterThickness'][:].values


def main():
    run = validateWithSpec()
    
    run.create_dest_scrip_file()
    
    run.interpolate_mali()

    run.interpolate_spec()

    #run.validate()

if __name__ == "__main__":
    main()

                            
