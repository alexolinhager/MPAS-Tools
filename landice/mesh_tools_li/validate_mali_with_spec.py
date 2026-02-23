#!/usr/bin/env python

print("Importing toolboxes")

import xarray as xr
import os
import numpy as np
import subprocess
from argparse import ArgumentParser
import tifffile as tiff
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d

class validateWithSpec:
    def __init__(self):
        print("Gathering Information ...")
        parser = ArgumentParser(prog='validate_mali_with_spec.py', 
                                description='Calculates a validation score for mali subglacial hydrology simulations using radar specularity content')
        parser.add_argument("--maliFile", dest="maliFile", required=True, help="MALI output file to validate")
        parser.add_argument("--specTiff", dest="specTiff", required=True, help='Tiff file containing specularity content classified according to Young 2016:'
                            'Values of 0 represent specularity content below 20%; values of 3.3 represent specularity content above 20% and energy 1' 
                            'microsecond below the bed 15 dB lower than the bed echo, and values of 6.7 represent specularity content above 20% and energy' 
                            '1 microsecond below the bed 15 dB within than the bed echo.')
        parser.add_argument("--compRes", dest="compRes", type=float, help="Grid resolution on which to interpolate and validate (meters)", default=5000.0)
        parser.add_argument("--Wr", dest="Wr", type=float, help="Simulation bed bump height", default=0.1)
        args = parser.parse_args()
        self.options = args

    def interpolate_to_common_grid(self):
        # establish common grid
        res = self.options.compRes

        ds_mali = xr.open_dataset(self.options.maliFile, decode_times=False, decode_cf=False)
        xCell = ds_mali['xCell'][:].values
        yCell = ds_mali['yCell'][:].values
        
        xmin = np.min(xCell)
        xmax = np.max(xCell)
        ymin = np.min(yCell)
        ymax = np.max(yCell)

        x_edges = np.arange(xmin, xmax + res, res)
        y_edges = np.arange(ymin, ymax + res, res)
        
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Remap MALI
        Xgrid, Ygrid = np.meshgrid(x_centers, y_centers)
        
        W = ds_mali['waterThickness'][-1,:].values # Assume last time step for now
        W_remapped = griddata(points=(xCell, yCell),
                                  values=W,
                              xi=(Xgrid, Ygrid),
                              method='linear')
        
        Z = ds_mali['bedTopography'][-1,:].values # Assume last time step for now
        Z_remapped = griddata(points=(xCell, yCell),
                                  values=Z,
                              xi=(Xgrid, Ygrid),
                              method='linear')
        
        H = ds_mali['thickness'][-1,:].values # Assume last time step for now
        H_remapped = griddata(points=(xCell, yCell),
                                  values=H,
                              xi=(Xgrid, Ygrid),
                              method='linear')

        # Open geoTiff and convert to netcdf
        with tiff.TiffFile(self.options.specTiff) as tif:
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

        specData = specData.astype(float)
        specData[specData == 0] = np.nan

        [Xspec, Yspec] = np.meshgrid(x,y)
        specData = specData.ravel()
        Xspec = Xspec.ravel()
        Yspec = Yspec.ravel()

        mask = np.isfinite(specData)
        specData = specData[mask]
        Xspec = Xspec[mask]
        Yspec = Yspec[mask]

        spec_remapped, x_edges_out, y_edges_out, binnum = binned_statistic_2d(
                Xspec, Yspec, specData,
                statistic='mean',
                bins=[x_edges,y_edges]
                )
        spec_remapped = spec_remapped.T
        print(f"spec shape: {spec_remapped.shape}")
        print(f"H shape: {H_remapped.shape}")
        print(f"Z shape: {Z_remapped.shape}")

        # Filter specularity data
        floating = (910/1028) * H_remapped + Z_remapped <= 0 
        spec_remapped[floating] = np.nan
        spec_remapped[H_remapped == 0] = np.nan

        east_AIS = Xgrid >= 0
       
        east_valid = east_AIS & np.isfinite(spec_remapped) & np.isfinite(W_remapped)
        west_valid = ~east_AIS & np.isfinite(spec_remapped) & np.isfinite(W_remapped)

        # calculate Rwt
        Rwt_e = W_remapped[east_valid] / self.options.Wr
        Rwt_w = W_remapped[west_valid] / self.options.Wr

        # Define comparison thresholds
        Sthresh = 3.33 # Physically-based specularity threshold
        Rthresh = np.arange(0.95, 1.0, 0.01)

        Strue_e = spec_remapped[east_valid] >= Sthresh 
        Sfalse_e = spec_remapped[east_valid] < Sthresh
        Strue_w = spec_remapped[west_valid] >= Sthresh
        Sfalse_w = spec_remapped[west_valid] < Sthresh
        
        Strue_e = Strue_e[:,None]
        Sfalse_e = Sfalse_e[:,None]
        Strue_w = Strue_w[:,None]
        Sfalse_w = Sfalse_w[:,None]

        Rtrue_e = Rwt_e[:, None] >= Rthresh
        Rfalse_e = ~Rtrue_e
        Rtrue_w = Rwt_w[:, None] >= Rthresh
        Rfalse_w = ~Rtrue_w

        print(f"Rtrue_e: {Rtrue_e.shape}")
        print(f"Strue_e: {Strue_e.shape}")
        tp_e = np.sum(Strue_e & Rtrue_e, axis=0)
        tn_e = np.sum(Sfalse_e & Rfalse_e, axis=0)
        fp_e = np.sum(Sfalse_e & Rtrue_e, axis=0)
        fn_e = np.sum(Strue_e & Rfalse_e, axis=0)

        tp_w = np.sum(Strue_w & Rtrue_w, axis=0)
        tn_w = np.sum(Sfalse_w & Rfalse_w, axis=0)
        fp_w = np.sum(Sfalse_w & Rtrue_w, axis=0)
        fn_w = np.sum(Strue_w & Rfalse_w, axis=0)
        
        true_agree_e = tp_e / (tp_e + fn_e)
        false_agree_e = tn_e / (tn_e + fp_e)

        true_agree_w = tp_w / (tp_w + fn_w)
        false_agree_w = tn_w / (tn_w + fp_w)
        
        balanced_score_e = 0.5 * (true_agree_e + false_agree_e)
        balanced_score_w = 0.5 * (true_agree_w + false_agree_w)

        print(f"true agree east: {true_agree_e}")
        print(f"false agree east: {false_agree_e}")
        print(f"true agree west: {true_agree_w}")
        print(f"false agree west: {false_agree_w}")
        print(f"balanced score east: {balanced_score_e}")
        print(f"balanced score west: {balanced_score_w}")
        print(f"total balanced score: {balanced_score_e + balanced_score_w}")

        ds_out = xr.Dataset()
        ds_out['X'] = xr.DataArray(Xgrid.astype('float64'), dims=("nx","ny"))
        ds_out['Y'] = xr.DataArray(Ygrid.astype('float64'), dims=("nx","ny"))
        ds_out['spec'] = xr.DataArray(spec_remapped.astype('float64'), dims=("nx","ny"))
        ds_out['W'] = xr.DataArray(W_remapped.astype('float64'), dims=("nx","ny"))
        ds_out['H'] = xr.DataArray(H_remapped.astype('float64'), dims=("nx","ny"))
        ds_out['Z'] = xr.DataArray(Z_remapped.astype('float64'), dims=("nx","ny"))
        ds_out.to_netcdf('remapped.nc')

def main():
    run = validateWithSpec()
    
    run.interpolate_to_common_grid()

if __name__ == "__main__":
    main()

                            
