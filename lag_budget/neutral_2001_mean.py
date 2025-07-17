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
        os.listdir(f'/sciserver/filedb{i:02}-0{j}')

seed = 2001

save_path = filedb_lst[19+int(sys.argv[-1])]+'/ocean/wenrui_temp/particle_file/neutral2001/nc_mean/'
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

time = '2001-10-01'
t = sd.utils.convert_time(time)
end_time = t-365*86400*5
stops = np.array([end_time])

xrange = (-34,-10)
yrange = (47.5,65)
zrange = (-200,0)

xbounds = np.linspace(xrange[0],xrange[1],4)
ybounds = np.linspace(np.sin(yrange[0]*np.pi/180), np.sin(yrange[1]*np.pi/180),4)
xbnds = []
ybnds = []
for i in range(3):
    for j in range(3):
        xbnds.append((xbounds[i],xbounds[i+1]))
        ybnds.append((np.arcsin(ybounds[j])*180/np.pi,np.arcsin(ybounds[j+1])*180/np.pi))
    
xrange = xbnds[int(sys.argv[-1])]
yrange = ybnds[int(sys.argv[-1])]

Nx = 12
Ny = 12
Nz = 36

x = np.linspace(xrange[0],xrange[1],Nx+1)
x = (x[1:]+x[:-1])/2

all_levels = np.linspace(zrange[0],zrange[1],Nz+1)
all_levels = (all_levels[:-1]+all_levels[1:])/2
levels = np.array(all_levels)

sins = np.linspace(np.sin(yrange[0]*np.pi/180), np.sin(yrange[1]*np.pi/180),Ny+1)
sins = (sins[1:]+sins[:-1])/2
y= np.arcsin(sins)*180/np.pi

x,y = np.meshgrid(x,y)

small_shape = x.shape
x = x.ravel()
y = y.ravel()

x,z = np.meshgrid(x,levels)
y,z = np.meshgrid(y,levels)

x = x.ravel()
y = y.ravel()
z = z.ravel()

p = sd.Particle(x = x,y = y, z=z, t = t,data = oce, 
                uname = 'utrans', vname = 'vtrans',wname = 'wtrans',transport = True,
                save_raw = True,
               )
p=p.subset(sd.get_masks.which_not_stuck(p))
p.empty_lists()
print('finished pre-calculating')

to_list_of_time(p,stops,update_stops = 'default',dump_prefix = save_path+f'Seed{seed}_')
print('success', p.N)