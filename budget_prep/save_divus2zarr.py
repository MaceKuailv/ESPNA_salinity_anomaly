import xarray as xr
import numpy as np
import os

import dask
a = 's'
dask.config.set(scheduler='threads')
path = f'/sciserver/filedb05-03/ocean/wenrui_temp/divu{a}_seas/'
os.listdir(path)


datasets_2 = sorted([path+i for i in os.listdir(path) if '.nc' in i])
print('before opening')
ugm = xr.open_mfdataset(datasets_2)
# ugm[f'divuj{a}j'] = ugm[f'divuj{a}j'].isel(time = slice(366))
print('finished opening')

out = ugm.reset_coords()[[f'divuj{a}j',f'divuy{a}j',f'divuj{a}y',f'divuy{a}y']]

import zarr

# zarr.blosc.list_compressors()

compressor = zarr.Blosc(cname='zlib')
opts = {}
for varname in out.data_vars:
    out[varname] = out[varname].chunk((1,50,13,90,90))
    opts[varname] = {'compressor': compressor}

from dask.diagnostics import ProgressBar
output_path = f'/sciserver/filedb02-02/ocean/wenrui_temp/heat/divu{a}_seas.zarr'
with ProgressBar():
    out.to_zarr(output_path, encoding = opts,mode = 'w')