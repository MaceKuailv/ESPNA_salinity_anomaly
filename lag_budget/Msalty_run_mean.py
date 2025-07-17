import numpy as np
# import oceanspy as ospy
import seaduck as sd
import xarray as xr
from otherToolsNeo import *
from seaduck.get_masks import which_not_stuck
import os 
import sys
print('got sys parameter:',sys.argv,type(sys.argv))

filedb_lst = []
for i in range(1,13):
    for j in range(1,4):
        filedb_lst.append(f'/sciserver/filedb{i:02}-0{j}')

seed = 2006

save_path = filedb_lst[19+int(sys.argv[-1])]+'/ocean/wenrui_temp/particle_file/saltyM/nc_mean/'
ds = xr.open_zarr('~/ECCO_transport')
big_ecco = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/mean*', engine = 'zarr')
snap_ecco = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/snap*', engine = 'zarr')
smean = xr.open_zarr('~/centerS_mean.zarr')
ecco_grid = xr.open_dataset('~/ECCO-grid/ECCO-GRID.nc')
agrid = ecco_grid.swap_dims({'k': 'Z', 'k_l': 'Zl', 'k_p1':'Zp1','k_u':'Zu'}).rename({'tile':'face',
               'i':'X','i_g':'Xp1',
               'j':'Y','j_g':'Yp1',
               
               'hFacC': 'HFacC','hFacS': 'HFacS','hFacW': 'HFacW'}).drop(['k','k_u','k_p1','k_l'])
agrid                   
ds = ds.assign_coords(coords=agrid.coords)
ds['time_midp'] = snap_ecco['time_midp']
ds['time'] = big_ecco['time']
oce = sd.OceData(ds)
oce['wtrans'][0] = 0
oce.time_midp

time = '2007-01-01'
t = sd.utils.convert_time(time)
end_time = t-365*86400*10
stops = np.array([end_time])

xrange = (-34,-10)
yrange = (47.5,65)
zrange = (-200,0)

lon_bool = np.logical_and(ds.XC[2]>xrange[0],ds.XC[2]<xrange[1])
lat_bool = np.logical_and(ds.YC[2]>yrange[0],ds.YC[2]<yrange[1])
dep_bool = np.logical_and(ds.Z>zrange[0],ds.Z<zrange[1])
pos_bool = np.logical_and(np.logical_and(lon_bool,lat_bool),dep_bool)

sp = big_ecco.SALT.sel(time = time)[0] - smean.smean
those = np.logical_and(sp[:,2]>0.1,pos_bool)
iz, iy, ix = (xr.DataArray(i,dims = 'stupid') for i in np.where(those))

XG = np.array(ds.XG[2])
YG = np.array(ds.YG[2])
Zl = np.array(ds.Zl)
np.random.seed(seed)
def random_pos(iz,iy,ix,num):
    xgs = XG[[iy,iy+1,iy+1,iy],[ix,ix,ix+1,ix+1]]
    ygs = YG[[iy,iy+1,iy+1,iy],[ix,ix,ix+1,ix+1]]
    zs = Zl[[iz+1,iz]]
    
    rx = np.random.random(num)-0.5
    ry = np.random.random(num)-0.5
    rz = np.random.random(num)
    
    w = sd.utils.weight_f_node(rx,ry)
    # print(zs)
    x = np.einsum('i,ji->j',xgs,w)
    y = np.einsum('i,ji->j',ygs,w)
    z = (1-rz)*zs[0]+rz*zs[1]
    return x,y,z

vols = np.array(ds.dxG[2,iy,ix])*np.array(ds.dyG[2,iy,ix])*np.array(ds.drF[iz])
Np = 5e4
num = np.round(vols*Np/np.sum(vols)).astype(int)
Np = np.sum(num)
xs = np.zeros(Np)
ys = np.zeros(Np)
zs = np.zeros(Np)
ts = t*np.ones_like(xs)

last_ind = 0
for i,(z,y,x,n) in enumerate(zip(iz,iy,ix,num)):
    x1,x2,x3 = random_pos(z,y,x,n)
    xs[last_ind: last_ind+n] = x1
    ys[last_ind: last_ind+n] = x2
    zs[last_ind: last_ind+n] = x3
    last_ind+=n

bins = Np//10+1
slc = slice(int(sys.argv[-1])*bins,(int(sys.argv[-1])+1)*bins)

p = sd.Particle(x = xs[slc],y = ys[slc], z=zs[slc], t = ts[slc],data = oce, 
                uname = 'utrans', vname = 'vtrans',wname = 'wtrans',transport = True,
                save_raw = True,
               )
p=p.subset(sd.get_masks.which_not_stuck(p))
p.empty_lists()
print('finished pre-calculating')

to_list_of_time(p,stops,update_stops = 'default',dump_prefix = save_path+f'Seed{seed}_')
print('success', p.N)