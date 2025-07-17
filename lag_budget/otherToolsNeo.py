import numpy as np
import xarray as xr
import zarr
from seaduck.get_masks import which_not_stuck
from post_process.performance import *

wrong = np.load('bolus_bug.npy')
lookup = np.load('fillin_index.npy')

def flatten(lstoflst,shapes = None):
    if shapes is None:
        shapes = [len(i) for i in lstoflst]
    suffix = np.cumsum(shapes)
    thething = np.zeros(suffix[-1])
    thething[:suffix[0]] = lstoflst[0]
    for i in range(1,len(lstoflst)):
        thething[suffix[i-1]:suffix[i]] = lstoflst[i]
    return thething

def bring_back(flt,shapes):
    suffix = np.cumsum(shapes)
    R = []
    for i in range(len(shapes)):
        R.append(flt[suffix[i]-shapes[i]:suffix[i]])
    return R

def particle2xarray(p):
    shapes = [len(i) for i in p.xxlist]
    # it = flatten(p.itlist,shapes = shapes)
    fc = flatten(p.fclist,shapes = shapes)
    iy = flatten(p.iylist,shapes = shapes)
    iz = flatten(p.izlist,shapes = shapes)
    ix = flatten(p.ixlist,shapes = shapes)
    rx = flatten(p.rxlist,shapes = shapes)
    ry = flatten(p.rylist,shapes = shapes)
    rz = flatten(p.rzlist,shapes = shapes)
    tt = flatten(p.ttlist,shapes = shapes)
    uu = flatten(p.uulist,shapes = shapes)
    vv = flatten(p.vvlist,shapes = shapes)
    ww = flatten(p.wwlist,shapes = shapes)
    du = flatten(p.dulist,shapes = shapes)
    dv = flatten(p.dvlist,shapes = shapes)
    dw = flatten(p.dwlist,shapes = shapes)
    xx = flatten(p.xxlist,shapes = shapes)
    yy = flatten(p.yylist,shapes = shapes)
    zz = flatten(p.zzlist,shapes = shapes)
    vs = flatten(p.vslist,shapes = shapes)
    
    ds = xr.Dataset(
        coords = dict(
            shapes = (['shapes'],shapes),
            nprof  = (['nprof'],np.arange(len(xx)))
        ),
        data_vars = dict(
            # it = (['nprof'],it),
            fc = (['nprof'],fc),
            iy = (['nprof'],iy),
            iz = (['nprof'],iz),
            ix = (['nprof'],ix),
            rx = (['nprof'],rx),
            ry = (['nprof'],ry),
            rz = (['nprof'],rz),
            tt = (['nprof'],tt),
            uu = (['nprof'],uu),
            vv = (['nprof'],vv),
            ww = (['nprof'],ww),
            du = (['nprof'],du),
            dv = (['nprof'],dv),
            dw = (['nprof'],dw),
            xx = (['nprof'],xx),
            yy = (['nprof'],yy),
            zz = (['nprof'],zz),
            vs = (['nprof'],vs)
        )
    )
    return ds

def dump_to_zarr(neo, oce, wrong, lookup, filename,use_region = False):

    if use_region:
        (ind1,ind2,frac, wrong_ind,
         [gulf_ind,labr_ind,gdbk_ind,nace_ind,egrl_ind],
         tres, last, first) = find_ind_frac_tres(neo,oce,wrong,lookup,use_region = use_region)
    else:
        ind1,ind2,frac, wrong_ind, tres, last, first = find_ind_frac_tres(neo,oce,wrong,lookup)
    
    neo['five'] = xr.DataArray(['iw','iz','face','iy','ix'], dims = 'five')
    neo = neo.assign_coords(wrong_ind = xr.DataArray(wrong_ind.astype('int32'),dims = 'wrong_ind'))
    if use_region:
        neo['gulf'] = xr.DataArray(gulf_ind.astype(bool),dims = 'nprof')
        neo['labr'] = xr.DataArray(labr_ind.astype(bool),dims = 'nprof')
        neo['gdbk'] = xr.DataArray(gdbk_ind.astype(bool),dims = 'nprof')
        neo['nace'] = xr.DataArray(nace_ind.astype(bool),dims = 'nprof')
        neo['egrl'] = xr.DataArray(egrl_ind.astype(bool),dims = 'nprof')
        # region_ind = np.concatenate([gulf_ind,labr_ind,gdbk_ind,nace_ind,egrl_ind]).astype('int32')
        # neo.attrs['region_shape'] = [len(i) for i in [gulf_ind,labr_ind,gdbk_ind,nace_ind,egrl_ind]]
        # if len(region_ind)>0:
        #     neo = neo.assign_coords(region_ind = xr.DataArray(region_ind,dims = 'region_ind'))

    neo['ind1'] = xr.DataArray(ind1.astype('int16'), dims = ['five','nprof'])
    neo['ind2'] = xr.DataArray(ind2.astype('int16'), dims = ['five','nprof'])
    neo['frac'] = xr.DataArray(frac, dims = 'nprof')
    neo['tres'] = xr.DataArray(tres, dims = 'nprof')
    # neo['last'] = xr.DataArray(last.astype('int64'), dims = 'shapes')
    # neo['first'] = xr.DataArray(first.astype('int64'), dims = 'shapes')
    
    neo['face'] = neo['fc'].astype('int16')
    neo['ix'] = neo['ix'].astype('int16')
    neo['iy'] = neo['iy'].astype('int16')
    neo['iz'] = neo['iz'].astype('int16')
    neo['vs'] = neo['vs'].astype('int16')
    
    neo = neo.drop_vars(['rx','ry','rz','uu','vv','ww','du','dv','dw','fc'])
    
    neo.to_zarr(filename, mode = 'w')
    zarr.consolidate_metadata(filename)
    
def store_lists(pt,name):
    neo = particle2xarray(pt)
    dump_to_zarr(neo, pt.ocedata, wrong, lookup, name,use_region = False)

def to_list_of_time(pt,normal_stops,update_stops = 'default',return_in_between  =True,dump_prefix = ''):
    t_min = np.minimum(np.min(normal_stops),pt.t[0])
    t_max = np.maximum(np.max(normal_stops),pt.t[0])

    if 'time' not in pt.ocedata[pt.uname].dims:
        pass
    else:
        data_tmin = pt.ocedata.ts.min()
        data_tmax = pt.ocedata.ts.max()
        if t_min<data_tmin or t_max>data_tmax:
            raise Exception(f'time range not within bound({data_tmin},{data_tmax})')
    if update_stops == 'default':
        update_stops = pt.ocedata.time_midp[np.logical_and(t_min<pt.ocedata.time_midp,
                                                     pt.ocedata.time_midp<t_max)]
    temp = (list(zip(normal_stops,np.zeros_like(normal_stops)))+
            list(zip(update_stops,np.ones_like(update_stops))))
    temp.sort(key = lambda x:abs(x[0]-pt.t[0]))
    stops,update = list(zip(*temp))
#         return stops,update
    pt.get_u_du()
#     R = []
    for i,tl in enumerate(stops):
        print()
        timestr = str(np.datetime64(round(tl),'s'))
        print(timestr)
        if pt.save_raw:
            # save the very start of everything. 
            pt.note_taking(stamp = 15)
        pt.to_next_stop(tl)
        # assert (~which_not_stuck(pt)).all()
        if update[i]:
            if not pt.too_large:
                pt.update_uvw_array()
            pt.get_u_du()
            if return_in_between:
#                 R.append(pt.deepcopy())
                store_lists(pt,dump_prefix+timestr)
        else:
            store_lists(pt,dump_prefix+timestr)
#             R.append(pt.deepcopy())
        if pt.save_raw:
            pt.empty_lists()
#     return stops,R