import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from post_process.performance import *
from post_process.estimate import *
import os
import seaduck as sd

from load_data.anomaly import *
from load_data.bolus_mask import separate_e_ua
from post_process.read import simple_read_neo
from post_process.estimate import separate_lhs

import time

import sys

the_slice = slice(330,332)

region_names =['gulf','labr','gdbk','nace','egrl']

int_arg = int(sys.argv[-1])
bins = 2
dataset_slc = slice(bins*int_arg,bins*(int_arg+1))
table_path = '/sciserver/filedb10-01/ocean/wenrui_temp/table_dec11/table'+str(int_arg)+str(the_slice)
print('got sys parameter:',sys.argv,type(sys.argv),table_path[-10:])

import logging
logging.basicConfig(filename='~/parallel_run/myapp.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

# particle_path = '/sciserver/filedb10-01/ocean/wenrui_temp/particle_file/14400run_uniform/'
# output_path = particle_path+'output/' 
# zarr_path = particle_path+'zarr/'
vec = xr.open_zarr('sxsysz_mean')
oce = sd.OceData(vec)
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
        if 'particle_file' in var:
            pass
        elif 'uall' in var:
            pass
        elif '.zip' in var:
            pass
        elif 'centerS' in var:
            continue
        elif 'table' in var:
            pass
        elif 'walls_normal' in var:
            pass
        elif 'E_ua_mean' in var:
            E_ua_mean_name = path+var
        elif 'tendS_0N-1' in var:
            tends_fl_name  = path+var
        else:
            first_patch.append(path+var)

ds = xr.open_mfdataset(first_patch,engine = 'zarr')
ds = ds.drop_vars('E_ua_mean')

new = xr.open_zarr(E_ua_mean_name)
ds = xr.merge([ds,new])

tend_ = xr.open_zarr(tends_fl_name)
ds.tendS[0] = tend_.tendS_first
ds.tendS[-1]= tend_.tendS_last

ds['R'] = -ds['ubargradsprime']
ds['e_ua'] = ds.E_ua-ds.E_ua_mean
ds['e_ssh'] = (ds.E_ssh-ds.E_ssh_mean)
ds['E'] = (ds["uprimegradsprime"]-ds["u'grads'_mean"])
ds['dif_h'] = (ds.dif_hConvS-ds.dif_hConvS_mean).transpose('time','Z','face','Y','X')
ds['dif_v'] = (ds.dif_vConvS-ds.dif_vConvS_mean).transpose('time','Z','face','Y','X')
ds['I'] = (ds.forcS-ds.forcS_mean)
ds['A'] = ds["uprimegradsbar"]
ds['U'] = ds.tendS_mean - ds.tendS
ds['F'] = (ds.pe_mean - ds.pe)

ds['lhs'] = ds['U'] + ds['e_ssh']
print('I wanna print')

rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']
lhs_list = ['e_ssh', 'U']
termlist = rhs_list+['lhs']
for var in termlist:
    ds[var] = ds[var].transpose('time','Z','face','Y','X')

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
print('I wanna print!')

filedb_lst = []
for i in range(1,13):
    for j in range(1,4):
        filedb_lst.append(f'/sciserver/filedb{i:02}-0{j}/')
path_mid = 'ocean/wenrui_temp/particle_file/meanM/nc/'
which_node = list(range(9))
spread = len(which_node)
zarr_path = filedb_lst[which_node[0]]+path_mid
duration = len(os.listdir(zarr_path))
dates = [i[-19:] for i in sorted(os.listdir(filedb_lst[which_node[0]]+path_mid),reverse = True)][the_slice]
all_files = []
for node in which_node:
    path = filedb_lst[node]+path_mid
    all_files.append([path+i for i in sorted(os.listdir(path),reverse = True)][the_slice])
all_files = np.array(all_files)

print('I wanna print!!')

# dates = sorted(os.listdir(zarr_path),reverse = True)[the_slice]
dataset_date_id = np.array(ds.time[dataset_slc])
hovmoller = xr.Dataset()
for var in rhs_list+['lhs', 'sf', 'sl']:
    hovmoller[var] = xr.DataArray(np.zeros((len(dataset_date_id), len(dates))),dims = ('time','space'))
for var in rhs_list:
    for region in region_names:
        hovmoller[var+'_'+region] = xr.DataArray(
            np.zeros((len(dataset_date_id), len(dates))),
            dims = ('time','space')
        )
hovmoller['time'] = ds.time[dataset_slc]

print('let the thing begin')
try:
    t1 = time.time()
    for it,dataset_date in enumerate(dataset_date_id):
        my = ds.sel(time = dataset_date)
        vec_i = vec.sel(time = dataset_date)
    
        prefetch_scalar = get_prefetch_scalar(my, termlist)
        prefetch_scalar = separate_e_ua(prefetch_scalar, bolus_mask, e_ua_name = 'e_ua')
        
        prefetch_vec = get_prefetch_vec(vec_i)
        more = np.array(prefetch_vec['s'][0])
        # print('dataset_date:', dataset_date, time.time()-t1)
        for jt, date in enumerate(dates):
            if jt %50 ==0:
                print('dataset_date:', dataset_date, str(it)+'/'+str(bins),
                      str(jt)+'/'+str(len(dates)), 
                      time.time()-t1)
            files = all_files[:,jt]
            assert np.array([(date in name) for name in files]).all(), (date, files[0])
            datasets = [xr.open_zarr(files[i]) for i in range(len(files))]
            ds0 = datasets[0]
            neo = xr.Dataset()
            neo['shapes'] = xr.concat([ds.shapes for ds in datasets], dim = 'shapes')
            nprof = [len(ds.nprof) for ds in datasets]
            prefix = [0]+list(accumulate(nprof))
            neo['wrong_ind'] = xr.concat([datasets[i].wrong_ind+prefix[i] for i in range(len(datasets))], dim = 'wrong_ind')
                
            for var in ['face','frac','ind1','ind2','ix','iy','iz','tres','tt','vs','xx','yy','zz']:
                neo[var] = xr.concat([ds[var] for ds in datasets], dim = 'nprof')
            
            for region in region_names:
                neo[region] = xr.concat([ds[region] for ds in datasets], dim = 'nprof')
            # print('meta read', time.time() - t1)
            
            ind1 = tuple(neo['ind1'].values)
            ind2 = tuple(neo['ind1'].values)
            
            frac = np.array(neo.frac)
            shapes = np.array(neo.shapes)
            tres = np.array(neo.tres)
            tact = np.array(neo.tt)
            
            ind = tuple(np.array(i) for i in [neo.iz-1,neo.face,neo.iy,neo.ix])
            wrong_ind = np.array(neo.wrong_ind)
            region_ind = {}
            for region in region_names:
                region_ind[region] = np.where(neo[region][:-1])
            
            step_dic = simple_read_neo(ind,termlist,prefetch_scalar)
            first, last = first_last(shapes)
            s1 = more[ind1]
            s2 = more[ind2]
            
            s_wall = s1*frac + (1-frac)*s2
            nstep = len(frac)-1
            deltas = np.nan_to_num(np.diff(s_wall))
            deltas[last[:-1]] = 0
            tres_used = -tres[1:]
            tres_used[last[:-1]] = 0 
            
            # correction = separate_lhs(neo, step_dic, last, lhs_name = ['U', 'e_ssh'])
            correction = separate_lhs_one(tact, step_dic, last)
            rhs_contr = deltas - correction
            
            contr_dic = term_p_relaxed(rhs_contr, tres_used, step_dic, rhs_list, wrong_ind)
            contr_dic['lhs'] = correction
            
            for var in rhs_list+['lhs']:
                hovmoller[var][it,jt] = np.nansum(contr_dic[var])
            for var in rhs_list:
                for region in region_names:
                    hovmoller[var+'_'+region][it,jt] = np.nansum(contr_dic[var][region_ind[region]])
            
            hovmoller['sl'][it,jt] = np.nanmean(s_wall[last])
            hovmoller['sf'][it,jt] = np.nanmean(s_wall[first])
    # hovmoller.to_zarr(table_path, mode = 'w')
    for var in hovmoller.data_vars:
        print(var, hovmoller[var].values)
    print('success')
except Exception as err:
    print(it,jt)
    # logger.error(err)
    print('failure')
    raise err
    
print('finished')
