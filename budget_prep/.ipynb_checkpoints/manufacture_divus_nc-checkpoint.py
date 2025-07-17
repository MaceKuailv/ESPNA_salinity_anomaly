import seaduck as sd
import xarray as xr
import matplotlib.pyplot as plt
import numpy  as np
# import tys
import os
import dask
import warnings
warnings.filterwarnings("ignore")
dask.config.set(scheduler='threads')
os.listdir('/sciserver/filedb01-02')
os.listdir('/sciserver/filedb02-02')
os.listdir('/sciserver/filedb03-02')

path = '/sciserver/filedb02-02/ocean/wenrui_temp/heat/'
path_to_output = '/sciserver/filedb05-03/ocean/wenrui_temp/'

# arg = tys.argv[-1]

mean = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/mean*',engine = 'zarr')
snap = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/snap*',engine = 'zarr')
grid = xr.open_zarr('~/ECCO_transport')
bc = xr.open_zarr('/sciserver/filedb09-01/ocean/GM_vel.zarr') #bolus correct
ds = xr.merge([mean, snap])
ds = ds.reset_coords().assign_coords(grid.coords).astype(float)


ut_mean = xr.open_zarr(path+'/utrans_j.zarr')
# ws_mean = xr.open_zarr(path+'/walls_m.zarr')
xxx = xr.open_zarr(path+'txj.zarr')
yyy = xr.open_zarr(path+'tyj.zarr')
zzz = xr.open_zarr(path+'tzj.zarr')
wt_mean = xr.merge([xxx,yyy,zzz])

# ws = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_salt.zarr')
wt = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_theta.zarr')

ds['utrans'] = ((ds['UVELMASS']+bc['u_gm'])*ds.drF*ds.dyG).transpose('time','Z','face','Y','Xp1')
ds['vtrans'] = ((ds['VVELMASS']+bc['v_gm'])*ds.drF*ds.dxG).transpose('time','Z','face','Yp1','X')
ds['wtrans'] = ((ds['WVELMASS']+bc['w_gm'])*ds.rA).transpose('time','Zl','face','Y','X')

from seaduck.eulerian_budget import *
xgcmgrd = create_ecco_grid(ds)
tub = sd.OceData(ds)

# Constants
R = 0.62
zeta1 = 0.6
zeta2 = 20.0
rhoconst = 1029
c_p = 3994

Depth = ds.Depth
dxG = ds.dxG
dyG = ds.dyG
drF = ds.drF
rA = ds.rA
hFacC = ds.HFacC.load()
vol = (rA*drF*hFacC).transpose('face','Z','Y','X')

xfluxname = 'ADVx'
yfluxname = 'ADVy'
zfluxname = 'ADVz'

# import zarr
    
# # zarr.blosc.list_compressors()

# compressor = zarr.Blosc(cname='zlib')
# opts = {}
# for arg in ['divus','divupsp','divupsm','divumsp','divumsm']:
#     opts[arg] = {'compressor': compressor}

for it in range(9497):
    sl = ds.isel(time = slice(it,it+1))
    wsl = wt.isel(time = slice(it,it+1))
    strtime = str(ds['time'][it].values)[:10]
    day_of_year = np.datetime64(ds['time'][it].values, 'D').astype('datetime64[D]').astype('O').timetuple().tm_yday
    day = day_of_year - 1
    if it % 1 ==0:
        print(strtime,day)
    tub._ds = sl
    for arg in ['divuyty','divuytj','divujty','divujtj']:
        use_most_common_chunking = True
        
        # if arg == 'divut':
        #     sl['ADVx'] = sl['utrans']*wsl['tx']
        #     sl['ADVy'] = sl['vtrans']*wsl['ty']
        #     sl['ADVz'] = sl['wtrans']*wsl['tz']
        if arg == 'divuyty':
            sl['ADVx'] = (sl['utrans']-ut_mean['utrans_j'][day])*(wsl['tx'] - wt_mean['txj'][day])
            sl['ADVy'] = (sl['vtrans']-ut_mean['vtrans_j'][day])*(wsl['ty'] - wt_mean['tyj'][day])
            sl['ADVz'] = (sl['wtrans']-ut_mean['wtrans_j'][day])*(wsl['tz'] - wt_mean['tzj'][day])
        if arg == 'divujty':
            sl['ADVx'] = (ut_mean['utrans_j'][day])*(wsl['tx'] - wt_mean['txj'][day])
            sl['ADVy'] = (ut_mean['vtrans_j'][day])*(wsl['ty'] - wt_mean['tyj'][day])
            sl['ADVz'] = (ut_mean['wtrans_j'][day])*(wsl['tz'] - wt_mean['tzj'][day])
        if arg == 'divuytj':
            sl['ADVx'] = (sl['utrans']-ut_mean['utrans_j'][day])*(wt_mean['txj'][day])
            sl['ADVy'] = (sl['vtrans']-ut_mean['vtrans_j'][day])*(wt_mean['tyj'][day])
            sl['ADVz'] = (sl['wtrans']-ut_mean['wtrans_j'][day])*(wt_mean['tzj'][day])
        if arg == 'divujtj':
            sl['ADVx'] = xr.ones_like(sl['utrans'])*(ut_mean['utrans_j'][day])*(wt_mean['txj'][day])
            sl['ADVy'] = xr.ones_like(sl['vtrans'])*(ut_mean['vtrans_j'][day])*(wt_mean['tyj'][day])
            sl['ADVz'] = xr.ones_like(sl['wtrans'])*(ut_mean['wtrans_j'][day])*(wt_mean['tzj'][day])
            # use_most_common_chunking = False
        sl[arg] = total_div(tub, xgcmgrd, xfluxname, yfluxname,zfluxname)
    output_list = ['divuyty','divuytj','divujty','divujtj']
    out = sl[output_list]

    for varname in out.data_vars:
        out[varname] = out[varname].transpose('time','Z','face','Y','X').chunk((1,50,13,90,90))
    
    out.to_netcdf(path_to_output+'divut_seas/'+strtime+'.nc',mode = 'w')