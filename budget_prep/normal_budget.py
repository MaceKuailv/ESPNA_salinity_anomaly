import seaduck as sd
import xarray as xr
import matplotlib.pyplot as plt
import numpy  as np
import sys
import os
import dask
dask.config.set(scheduler='threads')
os.listdir('/sciserver/filedb01-02')
os.listdir('/sciserver/filedb02-02')
os.listdir('/sciserver/filedb03-02')

path = '/sciserver/filedb02-02/ocean/wenrui_temp/heat/'

arg = sys.argv[-1]

mean = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/mean*',engine = 'zarr')
snap = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/snap*',engine = 'zarr')
grid = xr.open_zarr('~/ECCO_transport')
ecco_grid = grid
wecco_grid = xr.open_dataset('~/ECCO-grid/ECCO-GRID.nc')
bc = xr.open_zarr('/sciserver/filedb09-01/ocean/GM_vel.zarr') #bolus correct
ds = xr.merge([mean, snap])
ds = ds.reset_coords().assign_coords(grid.coords).astype(float)

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

dt = [float(t)/10**9 for t in np.diff(ds.time_midp)]
dt = xr.DataArray(data=dt, dims=['time'], coords={'time':ds.time[1:-1]})
use_most_common_chunking = True

if arg == 'adv':
    ds['ADVx_TH'] = ds.ADVx_TH.where(ecco_grid.HFacW.values > 0,0)
    ds['ADVy_TH'] = ds.ADVy_TH.where(ecco_grid.HFacS.values > 0,0)
    ADVxy_diff = xgcmgrd.diff_2d_vector({'X' : ds.ADVx_TH, 'Y' : ds.ADVy_TH}, boundary = 'fill')
    adv_hConvH = (-(ADVxy_diff['X'] + ADVxy_diff['Y']))
    ds['ADVr_TH'] = ds.ADVr_TH.where(ecco_grid.HFacC.values > 0,0)
    ADVr_TH = ds.ADVr_TH
    adv_vConvH = xgcmgrd.diff(ADVr_TH, 'Z', boundary='fill')
    G_advection = (adv_hConvH + adv_vConvH)/vol
    ds['adv'] = G_advection
    # xfluxname = 'ADVx_TH'
    # yfluxname = 'ADVy_TH'
    # zfluxname = 'ADVr_TH'
    # ds['adv'] = total_div(tub, xgcmgrd, xfluxname, yfluxname,zfluxname).transpose('time','Z','face','Y','X')
    output_list = [arg]
elif arg == 'dif_h':
    # xfluxname = 'DFxE_TH'
    # yfluxname = 'DFyE_TH'
    # ds['dif_h'] = hor_div(tub, xgcmgrd, xfluxname, yfluxname).transpose('time','Z','face','Y','X')
    ds['DFxE_TH'] = ds.DFxE_TH.where(ecco_grid.HFacW.values > 0,0)
    ds['DFyE_TH'] = ds.DFyE_TH.where(ecco_grid.HFacS.values > 0,0)
    
    DFxyE_diff = xgcmgrd.diff_2d_vector({'X' : ds.DFxE_TH, 'Y' : ds.DFyE_TH}, boundary = 'fill')
    dif_hConvH = (-(DFxyE_diff['X'] + DFxyE_diff['Y']))
    ds['dif_h'] = dif_hConvH/vol
    output_list = [arg]
elif arg == 'dif_v':
    ds['DFrE_TH'] = ds.DFrE_TH.where(ecco_grid.HFacC.values > 0,0)
    ds['DFrI_TH'] = ds.DFrI_TH.where(ecco_grid.HFacC.values > 0,0)
    DFrE_TH = ds.DFrE_TH
    DFrI_TH = ds.DFrI_TH
    dif_vConvH = xgcmgrd.diff(DFrE_TH, 'Z', boundary='fill') + xgcmgrd.diff(DFrI_TH, 'Z', boundary='fill')
    ds['dif_v'] = dif_vConvH/vol
    # ds['DFr_TH'] = ds['DFrE_TH']+ ds['DFrI_TH']
    # zfluxname = 'DFr_TH'
    # ds['dif_v'] = ver_div(tub,xgcmgrd,zfluxname).transpose('time','Z','face','Y','X')
    output_list = [arg]
elif arg == 'forcH':
    # mskC = ds['HFacC'].copy(deep=True).load()
    # mskC.values[mskC.values>0] = 1
    
    # land_mask = mskC[0]
    # land_mask.values[land_mask.values==0] = np.nan
    # land_mask_3d = mskC.copy()
    # land_mask_3d.values[land_mask_3d.values==0] = np.nan
    
    # TFLUX = ds.TFLUX
    # oceQsw = ds.oceQsw
    
    # Z = ds.Z.load()
    # RF = np.concatenate([ds.Zl,[np.nan]])
    
    # q1 = R*np.exp(1.0/zeta1*RF[:-1]) + (1.0-R)*np.exp(1.0/zeta2*RF[:-1])
    # q2 = R*np.exp(1.0/zeta1*RF[1:]) + (1.0-R)*np.exp(1.0/zeta2*RF[1:])
    
    # zCut = np.where(Z < -200)[0][0]
    # q1[zCut:] = 0
    # q2[zCut-1:] = 0
    
    # q1 = xr.DataArray(q1,coords=[Z],dims=['Z']).persist()
    # q2 = xr.DataArray(q2,coords=[Z],dims=['Z']).persist()
    
    # forcH_subsurf = ((q1*(mskC==1)-q2*(mskC.shift(Z=-1)==1))*oceQsw).transpose('time','face','Z','Y','X')
    
    # # Surface heat flux (at the sea surface)
    # forcH_surf = ((TFLUX - (1-(q1[0]-q2[0]))*oceQsw)*mskC[0]).transpose('time','face','Y','X').assign_coords(Z=-5).expand_dims('Z')
    
    # # Full-depth forcing
    # forcH = xr.concat([forcH_surf,forcH_subsurf[:,:,1:]], dim='Z').transpose('time','face','Z','Y','X')
    
    # mskC_shifted = mskC.shift(Z=-1)
    
    # mskC_shifted.values[-1,:,:,:] = 0
    # mskb = mskC - mskC_shifted
    
    # geoflx_llc = ds.GEOFLX.isel(time = 0)
    
    # # Create 3d field of geothermal heat flux
    # geoflx3d = geoflx_llc * mskb.transpose('face','Z','Y','X')
    # GEOFLX = geoflx3d.transpose('face','Z','Y','X').persist()
    
    # # Add geothermal heat flux to forcing field and convert from W/m^2 to degC/s
    # forcH = (((forcH + GEOFLX)/(rhoconst*c_p))/(hFacC*drF)).transpose('time','Z','face','Y','X')
    Z = ecco_grid.Z.compute()
    RF = np.concatenate([wecco_grid.Zp1.values[:-1],[np.nan]])
    # RF
    
    q1 = R*np.exp(1.0/zeta1*RF[:-1]) + (1.0-R)*np.exp(1.0/zeta2*RF[:-1])
    q2 = R*np.exp(1.0/zeta1*RF[1:]) + (1.0-R)*np.exp(1.0/zeta2*RF[1:])
    
    zCut = np.where(Z < -200)[0][0]
    q1[zCut:] = 0
    q2[zCut-1:] = 0
    
    q1 = xr.DataArray(q1,dims=['Z'])
    q2 = xr.DataArray(q2,dims=['Z'])
    
    mskC = ecco_grid.HFacC.copy(deep=True).compute()
    
    # Change all fractions (ocean) to 1. land = 0
    mskC.values[mskC.values>0] = 1
    
    forcH_subsurf = ((q1*(mskC==1)-q2*(mskC.shift(Z=-1)==1))*ds.oceQsw).transpose('time','Z','face','Y','X')
    
    # forcH_surf = ((ds.TFLUX - (1-(q1[0]-q2[0]))*ds.oceQsw)*mskC[0]).assign_coords(Z=0).expand_dims('Z').transpose('time','Z','face','Y','X')
    forcH_surf = ((ds.TFLUX - (1-(q1[0]-q2[0]))*ds.oceQsw)*mskC[0]).transpose('time','face','Y','X').assign_coords(Z=-5).expand_dims('Z')
    
    forcH = xr.concat([forcH_surf,forcH_subsurf.isel(Z = slice(1,None))], dim='Z').transpose('time','face','Z','Y','X')
    
    geoflx_llc = ds.GEOFLX[0]
    mskC_shifted = mskC.shift(Z=-1)
    
    mskC_shifted.values[-1,:,:,:] = 0
    mskb = mskC - mskC_shifted
    
    # Create 3d field of geothermal heat flux
    geoflx3d = geoflx_llc * mskb.transpose('Z','face','Y','X')
    GEOFLX = geoflx3d.transpose('Z','face','Y','X')
    GEOFLX.attrs = {'standard_name': 'GEOFLX','long_name': 'Geothermal heat flux','units': 'W/m^2'}
    
    G_forcing = ((forcH+GEOFLX)/(rhoconst*c_p))/(ecco_grid.HFacC*ecco_grid.drF)
    # ds['forcH'] = forcH
    ds['forcH']  = G_forcing
    output_list = [arg]
elif arg == 'tendH':
    # THETAsnp = ds.THETA_snap
    # HCsnp = (THETAsnp*(1+ds.ETAN_snap/Depth)).transpose('time_midp','Z','face','Y','X')
    # tendH = xgcmgrd.diff(HCsnp, 'time', boundary='fill', fill_value=0.0)/dt
    # tendH['Depth'] = ds.Depth
    # tendH['XC'] = ds.XC
    # tendH['YC'] = ds.YC
    # tendH['Z'] = ds.Z
    # tendH['rA'] = ds.rA
    # tendH['PHrefC'] = ds.PHrefC
    # tendH['drF'] = ds.drF
    # tendH['hFacC'] = ds.HFacC
    
    # tendH = xr.concat([np.nan*xr.zeros_like(vol).assign_coords(time=ds.time[0]).expand_dims('time'),tendH,
    #                    np.nan*xr.zeros_like(vol).assign_coords(time=ds.time[-1]).expand_dims('time')],
    #                   dim='time').transpose('time','Z','face','Y','X')
    delta_t = xgcmgrd.diff(ds.time_midp, 'time', boundary='fill', fill_value=np.nan)

    # Convert to seconds
    delta_t = delta_t.astype('f4') / 1e9
    sTHETA = ds.THETA_snap*(1+ds.ETAN_snap/ecco_grid.Depth)
    G_total = xgcmgrd.diff(sTHETA, 'time', boundary='fill', fill_value=np.nan)/delta_t
    ds['tendH'] = G_total
    output_list = [arg]
elif arg == 'utrans_m':
    ds['utrans_m'] = ((bc['u_gm']+ds['UVELMASS'])*ds.drF*ds.dyG).mean(dim = 'time').transpose('Z','face','Y','Xp1').chunk((50,13,90,90))
    ds['vtrans_m'] = ((bc['v_gm']+ds['VVELMASS'])*ds.drF*ds.dxG).mean(dim = 'time').transpose('Z','face','Yp1','X').chunk((50,13,90,90))
    ds['wtrans_m'] = ((bc['w_gm']+ds['WVELMASS'])*ds.rA).mean(dim = 'time').transpose('Zl','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = ['utrans_m','vtrans_m','wtrans_m']
elif arg == 'utrans_j':# NEW
    ds['utrans_j'] = ((bc['u_gm']+ds['UVELMASS'])*ds.drF*ds.dyG).groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','Xp1').chunk((6,50,13,90,90))
    ds['vtrans_j'] = ((bc['v_gm']+ds['VVELMASS'])*ds.drF*ds.dxG).groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Yp1','X').chunk((6,50,13,90,90))
    ds['wtrans_j'] = ((bc['w_gm']+ds['WVELMASS'])*ds.rA).groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Zl','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = ['utrans_j','vtrans_j','wtrans_j']
elif arg == 'walls_m':
    smean = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_salt.zarr')
    ds['sxm'] = smean['sx'].mean(dim = 'time').transpose('Z','face','Y','Xp1').chunk((50,13,90,90))
    ds['sym'] = smean['sy'].mean(dim = 'time').transpose('Z','face','Yp1','X').chunk((50,13,90,90))
    ds['szm'] = smean['sz'].mean(dim = 'time').transpose('Zl','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = ['sxm','sym','szm']
elif arg == 'wallt_m':
    smean = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_theta.zarr')
    ds['txm'] = smean['tx'].mean(dim = 'time').transpose('Z','face','Y','Xp1').chunk((50,13,90,90))
    ds['tym'] = smean['ty'].mean(dim = 'time').transpose('Z','face','Yp1','X').chunk((50,13,90,90))
    ds['tzm'] = smean['tz'].mean(dim = 'time').transpose('Zl','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = ['txm','tym','szm']
elif arg == 'txj':
    smean = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_theta.zarr')
    ds['txj'] = smean['tx'].groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','Xp1').chunk((6,50,13,90,90))
    # ds['tyj'] = smean['ty'].groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Yp1','X').chunk((6,50,13,90,90))
    # ds['tzj'] = smean['tz'].groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Zl','face','Y','X' ).chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = ['txj']
elif arg == 'tyj':# e6s1300
    smean = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_theta.zarr')
    ds['tyj'] = smean['ty'].groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Yp1','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = ['tyj']
elif arg == 'tzj':
    smean = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_theta.zarr')
    ds['tzj'] = smean['tz'].groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Zl','face','Y','X' ).chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = ['tzj']
elif arg == 'smean':
    ds[arg] = ds.SALT.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'tmean':
    ds[arg] = ds.THETA.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'tjijie':
    ds[arg] = ds.THETA.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'divu':
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    ds[arg] = vol_budget.hConvV+vol_budget.vConvV
    output_list = [arg]
elif arg == 'divu_mean':
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    ds[arg] = (vol_budget.hConvV+vol_budget.vConvV).mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'divu_j': #NEWe4s4687
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    ds[arg] = (vol_budget.hConvV+vol_budget.vConvV).groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'sdivu':
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    ds[arg] = ds['SALT']*(vol_budget.hConvV+vol_budget.vConvV)
    output_list = [arg]
elif arg == 'tdivu':
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    ds[arg] = ds['THETA']*(vol_budget.hConvV+vol_budget.vConvV)
    output_list = [arg]
elif arg == 'spdivum':
    sm = xr.open_zarr(path+'smean.zarr')['smean']
    sp = ds['SALT']-sm
    divum = xr.open_zarr(path+'divu_mean.zarr')['divu_mean']
    ds[arg] = sp*divum
    output_list = [arg]
elif arg == 'smdivup':
    sm = xr.open_zarr(path+'smean.zarr')['smean']
    divum = xr.open_zarr(path+'divu_mean.zarr')['divu_mean']
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    divup = (vol_budget.hConvV+vol_budget.vConvV)-divum
    ds[arg] = sm*divup
    output_list = [arg]
elif arg == 'spdivup':
    sm = xr.open_zarr(path+'smean.zarr')['smean']
    sp = ds['SALT']-sm
    divum = xr.open_zarr(path+'divu_mean.zarr')['divu_mean']
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    divup = (vol_budget.hConvV+vol_budget.vConvV)-divum
    ds[arg] = sp*divup
    output_list = [arg]
elif arg == 'smdivum':
    sm = xr.open_zarr(path+'smean.zarr')['smean']
    divum = xr.open_zarr(path+'divu_mean.zarr')['divu_mean']
    ds[arg] = (sm*divum).transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]

elif arg == 'tpdivum':
    tm = xr.open_zarr(path+'tmean.zarr')['tmean']
    tp = ds['THETA']-tm
    divum = xr.open_zarr(path+'divu_mean.zarr')['divu_mean']
    ds[arg] = tp*divu
    output_list = [arg]
elif arg == 'tmdivup':
    tm = xr.open_zarr(path+'tmean.zarr')['tmean']
    divum = xr.open_zarr(path+'divu_mean.zarr')['divu_mean']
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    divup = (vol_budget.hConvV+vol_budget.vConvV)-divum
    ds[arg] = tm*divup
    output_list = [arg]
elif arg == 'tpdivup':
    tm = xr.open_zarr(path+'tmean.zarr')['tmean']
    tp = ds['THETA']-tm
    divum = xr.open_zarr(path+'divu_mean.zarr')['divu_mean']
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    divup = (vol_budget.hConvV+vol_budget.vConvV)-divum
    ds[arg] = tp*divup
    output_list = [arg]
elif arg == 'tmdivum':
    tm = xr.open_zarr(path+'tmean.zarr')['tmean']
    divum = xr.open_zarr(path+'divu_mean.zarr')['divu_mean']
    ds[arg] = (tm*divum).transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]

elif arg == 'tydivuj':
    tm = xr.open_zarr(path+'tjijie.zarr')['tjijie']
    tp = ds['THETA'].groupby('time.dayofyear')-tm
    divum = xr.open_zarr(path+'divu_j.zarr')['divu_j']
    ds[arg] = tp.groupby('time.dayofyear')*divum
    output_list = [arg]
elif arg == 'tjdivuy':
    tm = xr.open_zarr(path+'tjijie.zarr')['tjijie']
    divum = xr.open_zarr(path+'divu_j.zarr')['divu_j']
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    divup = (vol_budget.hConvV+vol_budget.vConvV).groupby('time.dayofyear')-divum
    ds[arg] = tm*divup.groupby('time.dayofyear')
    output_list = [arg]
elif arg == 'tydivuy':
    tm = xr.open_zarr(path+'tjijie.zarr')['tjijie']
    tp = ds['THETA'].groupby('time.dayofyear')-tm
    divum = xr.open_zarr(path+'divu_j.zarr')['divu_j']
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    divup = (vol_budget.hConvV+vol_budget.vConvV).groupby('time.dayofyear')-divum
    ds[arg] = tp*divup
    output_list = [arg]
elif arg == 'tjdivuj':
    tm = xr.open_zarr(path+'tjijie.zarr')['tjijie']
    divum = xr.open_zarr(path+'divu_j.zarr')['divu_j']
    ds[arg] = (tm*divum)*xr.ones_like(ds['THETA']).groupby('time.dayofyear')
    output_list = [arg]

elif arg == 'tend1N-1':
    adv = xr.open_zarr(path+'adv.zarr')
    dif_h = xr.open_zarr(path+'dif_h.zarr')
    dif_v = xr.open_zarr(path+'dif_v.zarr')
    forch = xr.open_zarr(path+'forcH.zarr')
    tendh = xr.open_zarr(path+'tendH.zarr')
    appro = (adv.adv+dif_h.dif_h+dif_v.dif_v+forch.forcH).transpose('time','Z','face','Y','X')
    ds['tendH_first'] = appro.isel(time = 0).chunk((50,13,90,90))
    ds['tendH_last'] = appro.isel(time = -1).chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = ['tendH_first','tendH_last']

elif arg == 'close_m':
    adv = xr.open_zarr(path+'adv.zarr')
    dif_h = xr.open_zarr(path+'dif_h.zarr')
    dif_v = xr.open_zarr(path+'dif_v.zarr')
    forch = xr.open_zarr(path+'forcH.zarr')
    tendh = xr.open_zarr(path+'tendH.zarr')
    clos1 = tendh.tendH - (adv.adv+dif_h.dif_h+dif_v.dif_v+forch.forcH)
    ds[arg] = clos1.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]

elif arg == 'tforcv':
    vol_budget = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/vol_budget')
    ds[arg] = vol_budget['forcV']*ds['THETA']
    output_list = [arg]
elif arg == 'tforcv_m':
    tforcv = xr.open_zarr(path+'tforcv.zarr')
    ds[arg] = tforcv.tforcv.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'tforcv_j': #NEWe4s469 FAIL
    tforcv = xr.open_zarr(path+'tforcv.zarr')
    ds[arg] = tforcv.tforcv.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'adv_m':
    adv = xr.open_zarr(path+'adv.zarr')
    ds[arg] = adv.adv.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'adv_j': #NEWe2s13693
    adv = xr.open_zarr(path+'adv.zarr')
    ds[arg] = adv.adv.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'dif_h_m':
    data = xr.open_zarr(path+'dif_h.zarr')
    ds[arg] = data.dif_h.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'dif_h_j': #NEWe2s13695
    data = xr.open_zarr(path+'dif_h.zarr')
    ds[arg] = data.dif_h.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'dif_v_m':
    data = xr.open_zarr(path+'dif_v.zarr')
    ds[arg] = data.dif_v.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'dif_v_j': #NEWe2s13699
    data = xr.open_zarr(path+'dif_v.zarr')
    ds[arg] = data.dif_v.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'tendH_m':
    data = xr.open_zarr(path+'tendH.zarr')
    fl = xr.open_zarr(path+'tend1N-1.zarr')
    data['tendH'][0] = fl['tendH_first']
    data['tendH'][-1] = fl['tendH_last']
    ds[arg] = data.tendH.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'tendH_j': #NEW e1s16394
    data = xr.open_zarr(path+'tendH.zarr')
    fl = xr.open_zarr(path+'tend1N-1.zarr')
    data['tendH'][0] = fl['tendH_first']
    data['tendH'][-1] = fl['tendH_last']
    ds[arg] = data.tendH.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'forcH_m':
    data = xr.open_zarr(path+'forcH.zarr')
    ds[arg] = data.forcH.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'forcH_j': #NEW e1s16397
    data = xr.open_zarr(path+'forcH.zarr')
    ds[arg] = data.forcH.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
    
elif arg == 'e_ua_s':
    advs = xr.open_zarr('/sciserver/filedb12-01/ocean/wenrui_temp/advNdif_normal')
    divus = xr.open_zarr(path+'divus.zarr')
    ds[arg] = advs['adv_hConvS']+advs['adv_vConvS'] - divus.divus
    output_list = [arg]
elif arg == 'e_ua_t':
    advs = xr.open_zarr(path+'adv.zarr')
    divut = xr.open_zarr(path+'divut.zarr')
    ds[arg] = advs['adv'] - divut.divut
    output_list = [arg]
elif arg == 'e_ua_s_m':
    eua = xr.open_zarr(path+'e_ua_s.zarr')
    ds[arg] = eua.e_ua_s.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'e_ua_t_m':
    eua = xr.open_zarr(path+'e_ua_t.zarr')
    ds[arg] = eua.e_ua_t.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'e_ua_t_j': #NEW e1s164
    eua = xr.open_zarr(path+'e_ua_t.zarr')
    ds[arg] = eua.e_ua_t.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]

elif arg == 'e_ssh_s':
    pe = xr.open_zarr('/sciserver/filedb10-01/ocean/wenrui_temp/RainForcing')
    sdivu = xr.open_zarr(path+'sdivu.zarr')
    ds[arg] = pe.pe+sdivu.sdivu
    output_list = [arg]
elif arg == 'e_ssh_s_m':
    e_ssh_s = xr.open_zarr(path+'e_ssh_s.zarr')
    ds[arg] = e_ssh_s.e_ssh_s.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'e_ssh_t':
    pe = xr.open_zarr(path+'tforcv.zarr')
    tdivu = xr.open_zarr(path+'tdivu.zarr')
    ds[arg] = pe.tforcv+tdivu.tdivu
    output_list = [arg]
elif arg == 'e_ssh_t_m':
    e_ssh_t = xr.open_zarr(path+'e_ssh_t.zarr')
    ds[arg] = e_ssh_t.e_ssh_t.mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'e_ssh_t_j':#NEW e6s12
    e_ssh_t = xr.open_zarr(path+'e_ssh_t.zarr')
    ds[arg] = e_ssh_t.e_ssh_t.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]

elif arg == 'tdivu_j':
    tdivu = xr.open_zarr(path+'tdivu.zarr')
    ds[arg] = tdivu.tdivu.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'divut_j':
    tdivu = xr.open_zarr(path+'divut.zarr')
    ds[arg] = tdivu.divut.groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]

elif arg == 'upgrdsp_m': 
    tot = xr.open_zarr(path+'divus.zarr')
    div = xr.open_zarr(path+'spdivup.zarr')
    ds[arg] = (tot.divupsp - div.spdivup).mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'upgrdtp_m':
    tot = xr.open_zarr(path+'divut.zarr')
    div = xr.open_zarr(path+'tpdivup.zarr')
    ds[arg] = (tot.divuptp - div.tpdivup).mean(dim = 'time').transpose('Z','face','Y','X').chunk((50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
elif arg == 'uygrdty_j':
    tot = xr.open_zarr(path+'divut_seas.zarr')
    div = xr.open_zarr(path+'tydivuy.zarr')
    ds[arg] = (tot.divuyty - div.tydivuy).groupby('time.dayofyear').mean(dim = 'time').transpose('dayofyear','Z','face','Y','X').chunk((6,50,13,90,90))
    use_most_common_chunking = False
    output_list = [arg]
else:
    raise ValueError(f'argument not supported {arg}, {type(arg)}')

out = ds[output_list]

import zarr

# zarr.blosc.list_compressors()

compressor = zarr.Blosc(cname='zlib')
opts = {}
for varname in out.data_vars:
    if use_most_common_chunking:
        out[varname] = out[varname].transpose('time','Z','face','Y','X').chunk((1,50,13,90,90))
    opts[varname] = {'compressor': compressor}

from dask.diagnostics import ProgressBar
with ProgressBar():
    out.to_zarr(path+arg+'.zarr', encoding = opts,mode = 'w')