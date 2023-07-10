
# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')
sys.path.append('/Volumes/GoogleDrive/My Drive/')
# %%
from GemPhy.Stat.Bayes import *
from GemPhy.Geophysics.utils.util import *
from gempy.assets.geophysics import Receivers,GravityPreprocessing
from gempy.core.grid_modules.grid_types import CenteredRegGrid

import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF

from operator import add
import numpy as np
import tensorflow as tf
from Patua import PutuaModel

P_model = PutuaModel()
init_model = P_model.init_model()
# %%
# init_model.compute_model()
# gp.plot.plot_section(init_model, cell_number=18,
#                             direction='y', show_data=True)


# %%
args = dotdict({
    'learning_rate': 0.5,
    'Adam_iterations':200,
    'number_init': 2,
    'resolution':[20,20,20],
    'grav_res':5
})
model_extent = [None]*(6)
model_extent[::2] = P_model.P['xy_origin']
model_extent[1::2] = list( map(add, P_model.P['xy_origin'], P_model.P['xy_extent'] ) )

X_r = np.linspace(model_extent[0],model_extent[1],args.grav_res)
Y_r = np.linspace(model_extent[2],model_extent[3],args.grav_res)

r = []
for x in X_r:
    for y in Y_r:
        r.append(np.array([x,y]))

Z_r = model_extent[-1] # at the top surface
xyz = np.meshgrid(X_r, Y_r, Z_r)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

radius = [1000,1000,3000]

# %%
# @tf.function
def forward(sf,tz,model_,densities,sigmoid = True):
  final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav = model_.compute_gravity(tz,surface_points = sf,kernel = Reg_kernel,receivers = receivers,method = 'kernel_reg',gradient = sigmoid,LOOP_FLAG = False,values_properties =densities)
  return grav

# Normalization and rearranging to gempy input
# @tf.function
def forward_function(mu,model_,tz,static_xy,sfp_shape,sigmoid = True, transformer = None):
  '''This function rescale and manimulate the input data to the gravity function
        mu is scaled and need to be concatenate with other parameters
  '''

    
  ### Customized forward function, each surface has identical Z value for both 2 surface points
  ### Define the variable for the entire surface

  # Transform the normalized paramters to bounded truth range

  mu_norm = transformer.reverse_transform(mu)

  mu_sfp = tf.repeat(mu_norm[:-4], repeats=[2])

  mu0 = concat_xy_and_scale(mu_sfp,model_,static_xy,sfp_shape,sfp_shape[0])

  # model_.TFG.densities = mu_norm[-4:]
  # tf.print(mu_norm)
  densities = mu_norm[-4:]

  properties = tf.stack([model_.TFG.lith_label,densities],axis = 0)
  gravity = forward(mu0,tz,model_,properties,sigmoid)
  return gravity

###### Forward compute gravity #######
# %%
model_prior = init_model

model_extent = [None]*(6)
model_extent[::2] = P_model.P['xy_origin']
model_extent[1::2] = list( map(add, P_model.P['xy_origin'], P_model.P['xy_extent'] ) )
receivers = Receivers(radius,model_extent,xy_ravel,kernel_resolution = args.resolution)

Reg_kernel = CenteredRegGrid(receivers.xy_ravel,radius=receivers.model_radius,resolution=receivers.kernel_resolution)

# Define gravity kernel

# We define the regularization term to be the max of three dimension or diagnal distance, for regular grid
max_length = np.sqrt(Reg_kernel.dxyz[0]**2 + Reg_kernel.dxyz[1]**2 + Reg_kernel.dxyz[2]**2)
max_slope = 1.5*2/max_length * model_prior.rf

model_prior.activate_customized_grid(Reg_kernel)
gpinput = model_prior.get_graph_input()
# Create a tensorflow graph, here we add the regularization term for the slope for the step function
model_prior.create_tensorflow_graph(delta = 2.,gradient=True,compute_gravity=True,max_slope = max_slope)
g_center_regulargrid = GravityPreprocessing(Reg_kernel)
tz_center_regulargrid = tf.constant(g_center_regulargrid.set_tz_kernel(),model_prior.tfdtype)
tz = tf.constant(tz_center_regulargrid,model_prior.tfdtype)

# # Convert to tf.float64
# static_xy = constant64(model_prior.surface_points.df[['X','Y','Z']].to_numpy()[:,0:2])

# grav = forward_function(mean_prior_norm,model_prior,tz,static_xy,sfp_shape, transformer = ilt)
# grav

# %%
sf = 
grav = forward(sf,tz,model_prior,densities,sigmoid = True):