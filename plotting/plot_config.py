import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

dpi = 300
rerun = True
regen_talk = False

projection = ccrs.LambertConformal(central_longitude=-40, central_latitude=58.0)
extent = (-90, 10, 12, 90)

balance = cmocean.cm.balance
depth_cmap = "Greys_r"
depth_norm = mpl.colors.Normalize(vmin=-5000, vmax=5000)

fresh_time_cmap = plt.get_cmap('BuPu_r')
fresh_theme_color = 'teal'
fresh_idate = 8918

salty_time_cmap = plt.get_cmap('OrRd_r')
salty_theme_color = 'maroon'
salty_idate = 5479

mean_time_cmap = plt.get_cmap('YlGn_r')

s_cmap = plt.get_cmap('PuOr_r')
term_cmap = balance
term_cmap_r = cmocean.cm.balance_r

# a_palette5 = ["#df2935","#86ba90","#f5f3bb","#dfa06e","#412722"]
# a_palette5 = ["#1b4079","#4d7c8a","#7f9c96","#8fad88","#cbdf90"]
a_palette5 = ["#003049","#61988e","#f77f00","#7d1538","#345511"]
region_names =['gulf','labr','gdbk','nace','egrl']
region_longnames = ['Gulf Stream','Labrador Current','Grand Banks','NAC Extension','East Greenland Current']
region_longnames = dict(zip(region_names, region_longnames))
region_colors = dict(zip(region_names,a_palette5))

# rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']
rhs_list = ['A','F','dif_v','E','dif_h','e_ua','I']
term_colors = ['#fc8d62','#66c2a5','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494']
mean_term_colors = ['#fc8d62','#66c2a5','#8da0cb','#CE5FA4','#a6d854','#ffd92f','#e5c494']
color_dic = dict(zip(rhs_list,term_colors))
color_dic_mean = dict(zip(rhs_list,mean_term_colors))

error_color = 'r'

term_dic = {
    'A': "Anomalous Advection",
    'F': "Freshwater Forcing",
    'E': "Fluctuation Advection",
    # 'E': r"$-\left(u'\nabla s'\right)'$",
    # 'E': r"$\overline{u'\cdot \nabla s'}$",
    'dif_v': "Vertical Diffusion",
    'dif_h': "Horizontal diffusion",
    'e_ua': r"$f'_{ua}$",
    'I': r"$f'_{salt}$"
}
case_term_dic = {
    'A': "Anomalous Advection",
    'F': "Freshwater Forcing",
    # 'E': r"$(-u'\nabla s'-\overline{u'\nabla s'})$",
    'E': "Mean Fluctuation",
    'dif_v': "Vertical Diffusion",
    'dif_h': "Horizontal diffusion",
    'e_ua': r"$f'_{ua}$",
    'I': r"$f'_{salt}$"
}

NUMBER_OF_PARTICLE_domain4 = 139968
NUMBER_OF_PARTICLE_domain_all = 1224905

TOTAL_VOLUME_salty,NUMBER_OF_PARTICLE_salty,VOLUME_EACH_salty = (269320135376896.0, 999984, 269324444.5680091)
TOTAL_VOLUME_fresh,NUMBER_OF_PARTICLE_fresh,VOLUME_EACH_fresh = (131811648733184.0, 1000041, 131806244.67715223)
TOTAL_VOLUME_whole_domain = 558514510387999.06

fill_betweenx_kwarg = dict(
    color = 'grey',
    alpha = 0.5
)