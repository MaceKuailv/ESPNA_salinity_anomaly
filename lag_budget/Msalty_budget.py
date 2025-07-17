import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from post_process.performance import *
from post_process.estimate import *
import os
import seaduck as sd

from load_data.anomaly import *
from load_data.bolus_mask import separate_e_ua
from post_process.read import simple_read_nc
from post_process.estimate import separate_lhs

import os 
for i in range(1,9):
    os.listdir(f'/sciserver/filedb0{i}-02/')

particle_path = '/sciserver/filedb08-01/ocean/wenrui_temp/particle_file/saltyM/'
seed = 2006
map_path = particle_path+'maps/'
table_path = particle_path+'table/'
# output_path = particle_path+'output/' 
zarr_path = particle_path+'nc/'
vec0 = xr.open_zarr('sxsysz_mean')
oce = sd.OceData(vec0)

wrong = np.load('bolus_bug.npy')
bolus_mask = tuple(wrong)

path0 = '/sciserver/filedb10-01/ocean/wenrui_temp/'
path1 = '/sciserver/filedb11-01/ocean/wenrui_temp/'
path2 = '/sciserver/filedb12-01/ocean/wenrui_temp/'
all_file = []
first_patch = []
for path in [path0,path1,path2]:
    all_file+=[path + i for i in os.listdir(path)]
    for var in os.listdir(path):
        if 'particle' in var:
            continue
        elif '.zip' in var:
            continue
        elif 'centerS' in var:
            continue
        elif 'uall' in var:
            continue
        elif 'table' in var:
            continue
        elif 'walls_normal' in var:
            continue
        elif 'E_ua_mean' in var:
            E_ua_mean_name = path+var
        elif 'tendS_0N-1' in var:
            tends_fl_name  = path+var
        else:
            first_patch.append(path+var)
print('I want to print')
ds = xr.open_mfdataset(first_patch,engine = 'zarr', parallel = True)
ds = ds.drop_vars('E_ua_mean')

new = xr.open_zarr(E_ua_mean_name)
ds = xr.merge([ds,new])

tend_ = xr.open_zarr(tends_fl_name)
ds.tendS[0] = tend_.tendS_first
ds.tendS[-1]= tend_.tendS_last

ds['R'] = -ds['ubargradsprime'] - ds["uprimegradsprime"]
ds['e_ua'] = ds.E_ua-ds.E_ua_mean
ds['e_ssh'] = (ds.E_ssh-ds.E_ssh_mean)
ds['E'] = (-ds["u'grads'_mean"])
ds['dif_h'] = (ds.dif_hConvS-ds.dif_hConvS_mean).transpose('time','Z','face','Y','X')
ds['dif_v'] = (ds.dif_vConvS-ds.dif_vConvS_mean).transpose('time','Z','face','Y','X')
ds['I'] = (ds.forcS-ds.forcS_mean)
ds['A'] = ds["uprimegradsbar"]
ds['U'] = ds.tendS_mean - ds.tendS
ds['F'] = (ds.pe_mean - ds.pe)
print('I want to print!')

rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']
lhs_list = ['e_ssh', 'U']
termlist = rhs_list+lhs_list
for var in termlist:
    if 'time' in ds[var].dims:
        ds[var] = ds[var].transpose('time','Z','face','Y','X')
    else:
        ds[var] = ds[var].transpose('Z','face','Y','X')

target = 's'
indlist = ['ix', 'iy', 'izl_lin', 'face']
notrequired = ['rzl_lin', 'ry', 'rx','rt']
vellist = [ 'u', 'v', 'w', 'du', 'dv', 'dw']
morelist = ['lon', 'lat', 'dep', 'vs']

varlist = termlist+indlist+morelist+vellist+[target]

vec = xr.open_zarr(path1+'walls_normal')
vec_mean = xr.open_zarr('sxsysz_mean')
vec = xr.merge([vec,vec_mean])
vec['sxprime'] = vec['sx'] - vec['osx_mean']
vec['syprime'] = vec['sy'] - vec['osy_mean']
vec['szprime'] = vec['sz'] - vec['osz_mean']

filedb_lst = []
for i in range(1,13):
    for j in range(1,4):
        filedb_lst.append(f'/sciserver/filedb{i:02}-0{j}/')
path_mid = 'ocean/wenrui_temp/particle_file/saltyM/nc/'
which_node = list(range(19,29))
spread = len(which_node)
duration = len(os.listdir(filedb_lst[which_node[0]]+path_mid))
dates = [i[-19:] for i in sorted(os.listdir(filedb_lst[which_node[0]]+path_mid),reverse = True)]
all_files = []
for node in which_node:
    path = filedb_lst[node]+path_mid
    all_files.append([path+i for i in sorted(os.listdir(path),reverse = True)])
all_files = np.array(all_files)

files = all_files[:,0]
datasets = [xr.open_zarr(files[i]) for i in range(len(files))]
Np = sum([len(dataset.shapes) for dataset in datasets])

# peek = xr.open_zarr(zarr_path+f'Seed{seed}_2005-05-31T00:00:00')
# Np = len(peek.shapes)

particle_slc = slice(None)
dates = sorted([i[9:] for i in os.listdir(zarr_path) if 'zarr' not in i],reverse = True)
date_identifier = dates[particle_slc]
ds['index_of_time'] = xr.DataArray(np.arange(len(ds.time)),dims  ='time')
first_index = int(ds['index_of_time'].sel(time = date_identifier[0][:10]).values)
to_isel = np.arange(first_index, first_index - len(date_identifier),-1)
hovmoller = xr.Dataset()
for var in rhs_list+['lhs', 'sf', 'sl', 'lon', 'lat', 'dep']:
    hovmoller[var] = xr.DataArray(np.zeros((len(date_identifier),Np)),dims = ('time','space'))
print('I want to print!!')

def cumu_map_zarr(neo, contr_dic, varname, last):
    value = contr_dic[varname]
    ind = tuple(np.array(i)[:-1] for i in [neo.iz-1, neo.face, neo.iy, neo.ix])
    array = np.zeros((50,13,90,90))
    np.add.at(array, ind, value)
    return array

def cumu_map_array(neo, value):
    ind = tuple(np.array(i)[:-1] for i in [neo.iz-1, neo.face, neo.iy, neo.ix])
    array = np.zeros((50,13,90,90))
    np.add.at(array, ind, value)
    return array

maps = {}
for var in rhs_list+['count']:
    maps[var] = np.zeros((50,13,90,90))

# def term_indie(rhs_contr, tres_used, step_dic, rhs_list):
#     rhs_sum = np.zeros_like(tres_used)
#     dic = {}
#     for var in rhs_list:
#         dic[var] = step_dic[var][:-1]*tres_used
#         rhs_sum+=dic[var]
#     dic['error'] = rhs_contr-rhs_sum
#     return dic

t1 = time.time()
for it,(dataset_date, particle_date) in enumerate(zip(to_isel,date_identifier)):
    my = ds.isel(time = dataset_date)
    vec_i = vec.isel(time = dataset_date)

    prefetch_scalar = get_prefetch_scalar(my, termlist)
    prefetch_scalar = separate_e_ua(prefetch_scalar, bolus_mask, e_ua_name = 'e_ua')
    
    prefetch_vec = get_prefetch_vec(vec_i)
    more = np.array(prefetch_vec['s'][0])
    print('dataset_date:', particle_date, time.time()-t1)
    # for jt, date in enumerate(date_identifier):

    files = all_files[:,it]
    assert np.array([(particle_date in name) for name in files]).all()
    datasets = [xr.open_zarr(files[i]) for i in range(len(files))]
    ds0 = datasets[0]
    neo = xr.Dataset()
    neo['shapes'] = xr.concat([ds.shapes for ds in datasets], dim = 'shapes')
    nprof = [len(ds.nprof) for ds in datasets]
    prefix = [0]+list(accumulate(nprof))
    neo['wrong_ind'] = xr.concat([datasets[i].wrong_ind+prefix[i] for i in range(len(datasets))], dim = 'wrong_ind')
        
    for var in ['face','frac','ind1','ind2','ix','iy','iz','tres','tt','vs','xx','yy','zz']:
        neo[var] = xr.concat([ds[var] for ds in datasets], dim = 'nprof')
    
    s1 = more[tuple(neo['ind1'].data)]
    s2 = more[tuple(neo['ind2'].data)]
    
    frac = np.array(neo.frac)
    s_wall = s1*frac + (1-frac)*s2
    first, last = first_last(np.array(neo.shapes))
    tres = np.array(neo.tres)
    
    step_dic = simple_read_nc(neo,termlist,prefetch_scalar)
    
    nstep = len(frac)-1
    deltas = np.nan_to_num(np.diff(s_wall))
    deltas[last[:-1]] = 0
    tres_used = -tres[1:]
    tres_used[last[:-1]] = 0 

    correction = separate_lhs(neo, step_dic, last, lhs_name = ['U', 'e_ssh'])
    rhs_contr = deltas - correction

    contr_dic = term_p_relaxed(rhs_contr, tres_used, step_dic, rhs_list, neo.wrong_ind)
    # contr_dic = term_indie(rhs_contr, tres_used, step_dic, rhs_list)
    contr_dic['lhs'] = correction

    for var in rhs_list+['lhs']:
        cumsum = np.cumsum(np.nan_to_num(contr_dic[var]))
        cumsum = np.insert(cumsum[last-1],0,0)
        hovmoller[var][it,:] = np.diff(cumsum)
        # assert np.isclose(np.nansum(hovmoller[var][it]),np.nansum(contr_dic[var]))
    hovmoller['sl'][it,:] = s_wall[last]
    hovmoller['sf'][it,:] = s_wall[first]
    hovmoller['lon'][it,:] = np.array(neo.xx[first])
    hovmoller['lat'][it,:] = np.array(neo.yy[first])
    hovmoller['dep'][it,:] = np.array(neo.zz[first])

    for var in rhs_list:
        maps[var] += cumu_map_zarr(neo, contr_dic, var, last)
    maps['count'] += cumu_map_array(neo, np.ones_like(tres_used))

maps_ds = xr.Dataset()
for var in rhs_list+['count']:
    maps_ds[var] = xr.DataArray(maps[var],dims = ('Z','face','Y','X'))
maps_ds.to_zarr(map_path, mode = 'w')
hovmoller.to_zarr(table_path, mode = 'w')
print('success')
