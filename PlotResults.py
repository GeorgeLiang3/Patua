# %%
import sys
# local dir
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')
sys.path.append('/Volumes/GoogleDrive/My Drive/')

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from plotutils import *
import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF



result_dir = '/Users/zhouji/Documents/Cluster/Results/'
# filename = '/Users/zhouji/Documents/Cluster/Results/Patua-20230727-160137.json'
filename = None
if filename is None:
   filename = result_dir+ get_last_patua_file(result_dir)
else:
   filename = result_dir+filename
print(filename)
# %%


with open(filename) as f:
  data = json.load(f)
data = json.loads(data)

plot_result(data)
# %%
from run_script import *

# %%
model = ModelTF(P_model.geo_data)
model.activate_regular_grid()
model.create_tensorflow_graph(gradient=True,compute_gravity=True,max_slope = uq_P.max_slope)


model.compute_model()
# %%
gp.plot.plot_section(model, 
                        cell_number=18,
                        direction='y',
                        show_grid=True, 
                        show_data=True,
                        colorbar = True,)

# %%

def plot_result(mu, uq, ind = 18):
  model = ModelTF(uq.gp_model)
  model.activate_regular_grid()
  model.create_tensorflow_graph(gradient=True,compute_gravity=True,max_slope = uq.max_slope)

  sfp_xyz,properties = uq.parameter2input(mu,transformer= uq.transformer, DENSITIES_FLAG = True)
  
  model.compute_model(sfp_xyz)

  cmin = np.min(model.solutions.values_matrix)
  cmax = np.max(model.solutions.values_matrix)
  norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
  fig = gp.plot.plot_section(model, 
                        cell_number=ind,
                        block = model.solutions.values_matrix,
                        direction='y',
                        show_grid=True, 
                        show_data=True,
                        colorbar = True,
                        norm = norm,
                        cmap = 'viridis')
  

def plt_scatter(x, y, c, ax, title, norm, s=22, edgecolors='k'):

    cf = ax.scatter(x, y, c=c, cmap = 'jet', norm=norm, edgecolors=edgecolors,
                    s=s)
    plt.colorbar(cf, orientation = 'vertical', ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.ticklabel_format(style='sci', axis='x')
    # ax.set_xlim([P['xmin'], P['xmax']])
    # ax.set_ylim([P['ymin'], P['ymax']])

# %%
grav = uq_P.forward_function(mu)
# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plt_scatter(P_model.P['Grav']['xObs'] - P_model.P['xmin'], P_model.P['Grav']['yObs'] - P_model.P['ymin'],Data_obs, ax=ax[0], title='Observed Gravity', norm = None, s=22, edgecolors='k')

plt_scatter(P_model.P['Grav']['xObs'] - P_model.P['xmin'], P_model.P['Grav']['yObs'] - P_model.P['ymin'], grav,ax=ax[1], title='Simulated Gravity', norm = None, s=22, edgecolors='k')
# %%
mu = np.zeros(23)
plot_result(mu,uq_P, ind = 20)
# %%
xx = np.arange(model.resolution[0])
plt.plot(xx, model.solutions.values_matrix[:model.resolution[0]])