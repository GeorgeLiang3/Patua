
# %%
import sys
sys.path.append('/home/ib012512/Documents/GemPhy/GP_old')
sys.path.append('/home/ib012512/Documents/')
# sys.path.append('/Users/zhouji/Documents/github/PyNoddyInversion/code/')


# %%
from GemPhy.Stat.Bayes import Stat_model
from GemPhy.Geophysics.utils.util import constant64,dotdict,concat_xy_and_scale,calculate_slope_scale,NumpyEncoder
from gempy.assets.geophysics import Receivers,GravityPreprocessing
from GemPhy.Geophysics.utils.ILT import *
from gempy.core.grid_modules.grid_types import CenteredRegGrid

# import gempy as gp
# from gempy.core.tensor.modeltf_var import ModelTF

from MCMC import mcmc

import time
from operator import add
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import timeit

# from sklearn.metrics import mean_squared_error
import json

from Patua import PutuaModel
from LoadInputDataUtility import loadData
# from VisualizationUtilities import plt_scatter,get_norm

# suppress warinings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %%

print('load done')
Bayesargs = dotdict({
    'prior_sfp_std': 10,
    'likelihood_std':1, #the gravity data has an error range approximately between 0.5 mGal to 2.5 mGal. - Pollack, A, 2021
})

P_model = PutuaModel()
init_model = P_model.init_model()
loadData(P_model.P, number_data = 100)
# %%
# init_model.compute_model()
# gp.plot.plot_section(init_model, cell_number=18,
#                             direction='y', show_data=True)


# %%
args = dotdict({
    "foldername": "Patua",
    # 'learning_rate': 0.5,
    # 'Adam_iterations':200,
    # 'number_init': 2,
    'resolution':[16,16,12]
})
model_extent = [None]*(6)
model_extent[::2] = P_model.P['xy_origin']
model_extent[1::2] = list( map(add, P_model.P['xy_origin'], P_model.P['xy_extent'] ) )


X_r = P_model.P['Grav']['xObs']
Y_r = P_model.P['Grav']['yObs']
Z_r = [model_extent[-1]]*P_model.P['Grav']['nObsPoints']


xyz = np.stack((X_r,Y_r,Z_r)).T
radius = [2000,2000,3000]

# %%
# @tf.function
def forward(sf,tz,model_,densities,sigmoid = True):
  final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav = model_.compute_gravity(tz,surface_points = sf,kernel = Reg_kernel,receivers = receivers,method = 'kernel_reg',gradient = sigmoid,LOOP_FLAG = False,values_properties =densities)
  return grav



###### Forward compute gravity #######
# %%
model_prior = init_model

model_extent = [None]*(6)
model_extent[::2] = P_model.P['xy_origin']
model_extent[1::2] = list( map(add, P_model.P['xy_origin'], P_model.P['xy_extent'] ) )
receivers = Receivers(radius,model_extent,xyz,kernel_resolution = args.resolution)

Reg_kernel = CenteredRegGrid(receivers.xy_ravel,radius=receivers.model_radius,resolution=receivers.kernel_resolution)

# Define gravity kernel

# We define the regularization term to be the max of three dimension or diagnal distance, for regular grid
max_slope = calculate_slope_scale(Reg_kernel,rf = model_prior.rf)


model_prior.activate_customized_grid(Reg_kernel)

# Create a tensorflow graph, here we add the regularization term for the slope for the step function
model_prior.create_tensorflow_graph(delta = 2.,gradient=True,compute_gravity=True,max_slope = max_slope)
g_center_regulargrid = GravityPreprocessing(Reg_kernel)
tz_center_regulargrid = tf.constant(g_center_regulargrid.set_tz_kernel(),model_prior.tfdtype)
tz = tf.constant(tz_center_regulargrid,model_prior.tfdtype)


###### Convolutional method  (to save memory) ##########


###### ################## #######





# # %%
# sf = constant64(model_prior.geo_data.surface_points.df[['X_r','Y_r','Z_r']].to_numpy())
# densities = constant64(model_prior.geo_data.surfaces.df['densities'].to_numpy())
# # densities = tf.expand_dims(densities,0)
# properties = tf.stack([model_prior.TFG.lith_label,densities],axis = 0)

# # %%
# start = timeit.default_timer()
# grav = forward(sf,tz,model_prior,properties,sigmoid = True)
# end = timeit.default_timer()
# print('forward computing time: %.3f' % (end - start))
# # %%
# grav = -grav - min(-grav)
Data_obs = P_model.P['Grav']['Obs'] = P_model.P['Grav']['Obs'] - (min(P_model.P['Grav']['Obs']))
# %%
####### Plot the measurement data
# P = P_model.P
# fig, ax = plt.subplots()

# s=22
# edgecolors='k'
# cf = ax.scatter(P['Grav']['xObs'], P['Grav']['yObs'], c = P['Grav']['Obs'], cmap = 'jet', edgecolors=edgecolors,s=s)
# plt.colorbar(cf, orientation = 'vertical', ax=ax, fraction=0.046, pad=0.04)
# ax.set_xlim([P['xmin'], P['xmax']])
# ax.set_ylim([P['ymin'], P['ymax']])
# plt.show()

# # %%
# fig, ax = plt.subplots()

# cf = ax.scatter(P['Grav']['xObs'], P['Grav']['yObs'], c = grav, cmap = 'jet', edgecolors=edgecolors,s=s)
# plt.colorbar(cf, orientation = 'vertical', ax=ax, fraction=0.046, pad=0.04)
# ax.set_xlim([P['xmin'], P['xmax']])
# ax.set_ylim([P['ymin'], P['ymax']])
# # ax.set_title(title)
# # ax.set_aspect('equal', 'box')
# # ax.ticklabel_format(style='sci', axis='x')
# plt.show()

# print(mean_squared_error(P['Grav']['Obs'],grav))

# %%
######### DEFINE STATISTIC MODEL ###########

# define the static x y coordinates
all_points = model_prior.surface_points.df[['X','Y','Z']].to_numpy()
fault_and_intrusion_points = all_points[:28]
static_xy = all_points[:,0:2]
model_prior.static_xy = static_xy
strata_points = all_points[28:]
all_points_shape = all_points.shape

num_sf_para = strata_points.shape[0]
sfp_mean = strata_points[:,2]
sfp_std = constant64([Bayesargs.prior_sfp_std]*num_sf_para)

prior_mean = tf.concat([sfp_mean],axis = 0)
prior_std = tf.concat([sfp_std],axis = 0)


num_para_total = prior_mean.shape[0]

### Define the bounds for parameters, bounds has to be normalized first 
lowerBound = prior_mean - 3*prior_std
upperBound = prior_mean + 3*prior_std

# invertible logarithmic transform
ilt = ILT(lowerBound,upperBound)
mean_prior_norm = ilt.transform(prior_mean)

# %%
transformer = ilt
mu = transformer.transform(prior_mean)

# %%

# Normalization and rearranging to gempy input
# @tf.function
def forward_function(mu,model_,tz,fault_and_intrusion_points,all_points_shape,sigmoid = True, transformer = None):

  if transformer is not None:
    mu_norm = transformer.reverse_transform(mu)
  else:
    mu_norm = mu

  sfp = tf.concat([fault_and_intrusion_points[:,2],mu_norm],axis = -1)

  mu0 = concat_xy_and_scale(sfp,model_,model_.static_xy,all_points_shape)

  densities = constant64(model_prior.geo_data.surfaces.df['densities'].to_numpy())
  properties = tf.stack([model_prior.TFG.lith_label,densities],axis = 0)

  gravity = forward(mu0,tz,model_,properties,sigmoid)
  gravity = -gravity - tf.math.reduce_min(-gravity)
  return gravity
# %%

def log_likelihood(self,mu):
    # forward calculating gravity
    Gm_ = self.gravity_function(mu,self.model,self.tz,self.fault_and_intrusion_points,self.all_points_shape,transformer = self.transformer)

    mvn_likelihood = tfd.MultivariateNormalTriL(
        loc=Gm_,
        scale_tril=tf.cast(tf.linalg.cholesky(self.data_cov_matrix),self.tfdtype))

    likelihood_log_prob = tf.reduce_sum(mvn_likelihood.log_prob(self.Obs))
    return likelihood_log_prob


# %%
##### Stat model #####
stat_model = Stat_model(model_prior,forward_function,num_para_total, tz, transformer = ilt)
# customize rewrite the likelihood function
Stat_model.log_likelihood = log_likelihood
stat_model.fault_and_intrusion_points = fault_and_intrusion_points
stat_model.all_points_shape = all_points_shape
# %%
# Set Prior
stat_model.set_prior(Number_para = num_para_total)

# Manually define the total shape of surface points (including intrusion and faults here)
stat_model.sfp_shape = all_points_shape

# Set likelihood
Data_measurement = tf.cast(Data_obs,model_prior.dtype) # Observation data
Data_std = Bayesargs.likelihood_std
stat_model.set_likelihood(Data_measurement,Data_std)
stat_model.monitor=False

# %%
print(stat_model.log_likelihood(mu))
##########MCMC###########

MCMCargs = dotdict({
    'num_results': 10000,
    'number_burnin':0,
    'RMH_step_size': 0.02,
    'HMC_step_size': 0.003,
    'leapfrogs':4,
})

mu0_list = [mu]
samples_RMH_list = []
accept_RMH_list = []
samples_HMC_list = []
accept_HMC_list = []
for mu0 in mu0_list:
  samples_RMH,samples_HMC,accept_RMH,accept_HMC = mcmc(mu0,stat_model, RMH = True, HMC = True,MCMCargs = MCMCargs)
  samples_RMH_list.append(samples_RMH)
  accept_RMH_list.append(accept_RMH)
  samples_HMC_list.append(samples_HMC)
  accept_HMC_list.append(accept_HMC)


# %%

stat_model.set_result_path('/home/ib012512/Documents/Results/'+args.foldername+time.strftime("-%Y%m%d-%H%M%S"))
# %%
saving_dict = {'samples_RMH_list': samples_RMH_list,
              'accepted_rate_RMH':accept_RMH_list,
              'samples_HMC_list': samples_HMC_list,
              'accepted_rate_HMC':accept_HMC_list,
              # 'MAPs' : mu0_list,
              }
saving_dict.update(args)
saving_dict.update(MCMCargs)

json_dump = json.dumps(saving_dict, cls=NumpyEncoder)

with open(stat_model.path + '.json', 'w') as outfile:
    json.dump(json_dump, outfile)

print('Done')