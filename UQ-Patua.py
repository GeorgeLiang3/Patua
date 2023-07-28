
# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')
sys.path.append('/Volumes/GoogleDrive/My Drive/')
# sys.path.append('/Users/zhouji/Documents/github/PyNoddyInversion/code/')

%matplotlib inline
# %%
from GemPhy.Stat.Bayes import Stat_model
from GemPhy.Geophysics.utils.util import constant64,dotdict,concat_xy_and_scale,calculate_slope_scale
from gempy.assets.geophysics import Receivers,GravityPreprocessing
from GemPhy.Geophysics.utils.ILT import *
from gempy.core.grid_modules.grid_types import CenteredRegGrid

# import gempy as gp
# from gempy.core.tensor.modeltf_var import ModelTF

from MCMC import mcmc

from operator import add
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
import json

from Patua import PutuaModel
from LoadInputDataUtility import loadData
# from VisualizationUtilities import plt_scatter,get_norm
# %%
Bayesargs = dotdict({
    'prior_sfp_std': 50,
    'prior_den_std': 0.2,
    'likelihood_std':0.09, #the gravity data has an error range approximately between 0.5 mGal to 2.5 mGal. - Pollack, A, 2021
})

P_model = PutuaModel()
init_model = P_model.init_model()
loadData(P_model.P, number_data = 20)
# %%
# init_model.compute_model()
# gp.plot.plot_section(init_model, cell_number=18,
#                             direction='y', show_data=True)


# %%
args = dotdict({
    "foldername": "Patua",
    'learning_rate': 0.5,
    'Adam_iterations':200,
    'number_init': 2,
    'resolution':[16,16,12]
})
model_extent = [None]*(6)
model_extent[::2] = P_model.P['xy_origin']
model_extent[1::2] = list( map(add, P_model.P['xy_origin'], P_model.P['xy_extent'] ) )

# X_r = np.linspace(model_extent[0],model_extent[1],args.grav_res)
# Y_r = np.linspace(model_extent[2],model_extent[3],args.grav_res)

# r = []
# for x in X_r:
#     for y in Y_r:
#         r.append(np.array([x,y]))

X_r = P_model.P['Grav']['xObs']
Y_r = P_model.P['Grav']['yObs']
Z_r = [model_extent[-1]]*P_model.P['Grav']['nObsPoints']

# Z_r = model_extent[-1] # at the top surface
# xyz = np.meshgrid(X_r, Y_r, Z_r)
# xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

xyz = np.stack((X_r,Y_r,Z_r)).T
radius = [2000,2000,3000]

# %%
###### Base gravity forward function #######
# @tf.function
def forward(sf,tz,model_,densities,sigmoid = True):
  final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav = model_.compute_gravity(tz,surface_points = sf,kernel = Reg_kernel,receivers = receivers,method = 'kernel_reg',gradient = sigmoid,LOOP_FLAG = False,values_properties =densities)
  return grav

# %%
###### Wrapper gravity forward function #######
# @tf.function
def forward_function(mu,model_,tz,fix_points,all_points_shape,sigmoid = True, transformer = None, densities = True):
  print('densities:', densities)
  if transformer is None:
    mu_norm = mu
  else:
    mu_norm = transformer.reverse_transform(mu)

  if not densities: # use default densities defined in the model
    properties = constant64(model_.geo_data.surfaces.df['densities'].to_numpy())
    sfp_z = tf.concat([fix_points[:,2],mu_norm],axis = -1)

  else:
    properties = mu_norm[-5:]
    sfp_z = tf.concat([fix_points[:,2],mu_norm[:-5]],axis = -1)
    # concatenate the auxiliary densities
    auxiliary_densities = constant64([-1]*12)
    properties = tf.concat([properties[:1],auxiliary_densities,properties[1:]],axis = -1)

  sfp_xyz = concat_xy_and_scale(sfp_z,model_,model_.static_xy,all_points_shape)
  properties = tf.stack([model_.TFG.lith_label,properties],axis = 0)

  gravity = forward(sfp_xyz,tz,model_,properties,sigmoid)
  gravity = -gravity - tf.math.reduce_min(-gravity)
  return gravity




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


# %%
sf = constant64(model_prior.geo_data.surface_points.df[['X_r','Y_r','Z_r']].to_numpy())
densities = constant64(model_prior.geo_data.surfaces.df['densities'].to_numpy())
# densities = tf.expand_dims(densities,0)
properties = tf.stack([model_prior.TFG.lith_label,densities],axis = 0)

# %%
start = timeit.default_timer()
grav = forward(sf,tz,model_prior,properties,sigmoid = True)
end = timeit.default_timer()
print('forward computing time: %.3f' % (end - start))
# %%
grav = -grav - min(-grav)
Data_obs = P_model.P['Grav']['Obs'] = P_model.P['Grav']['Obs'] - (min(P_model.P['Grav']['Obs']))
# %%
####### Plot the measurement data
P = P_model.P
fig, ax = plt.subplots()

s=22
edgecolors='k'
cf = ax.scatter(P['Grav']['xObs'], P['Grav']['yObs'], c = P['Grav']['Obs'], cmap = 'jet', edgecolors=edgecolors,s=s)
plt.colorbar(cf, orientation = 'vertical', ax=ax, fraction=0.046, pad=0.04)
ax.set_xlim([P['xmin'], P['xmax']])
ax.set_ylim([P['ymin'], P['ymax']])
plt.show()

# %%
fig, ax = plt.subplots()

cf = ax.scatter(P['Grav']['xObs'], P['Grav']['yObs'], c = grav, cmap = 'jet', edgecolors=edgecolors,s=s)
plt.colorbar(cf, orientation = 'vertical', ax=ax, fraction=0.046, pad=0.04)
ax.set_xlim([P['xmin'], P['xmax']])
ax.set_ylim([P['ymin'], P['ymax']])
# ax.set_title(title)
# ax.set_aspect('equal', 'box')
# ax.ticklabel_format(style='sci', axis='x')
plt.show()

print(mean_squared_error(P['Grav']['Obs'],grav))

# %%
######### DEFINE STATISTIC MODEL ###########

# define the fix points coordinates
all_points = model_prior.surface_points.df[['X','Y','Z']].to_numpy()
df = model_prior.geo_data.surface_points.df
num_fault_points = len(df[df['surface'].str.startswith('fault')])
num_intrusion_points = len(df[df['surface'] == 'intrusion'])
num_GT_points = len(df[df['surface'] == 'GT'])

num_fix_points = num_fault_points + num_intrusion_points + num_GT_points # keep all the intrusion, faults and GT points fixed
fix_points = all_points[:num_fix_points] 
static_xy = all_points[:,0:2]
model_prior.static_xy = static_xy
strata_points = all_points[num_fix_points:]
all_points_shape = all_points.shape

num_sf_var = strata_points.shape[0]
sfp_mean = strata_points[:,2]
sfp_std = constant64([Bayesargs.prior_sfp_std]*num_sf_var)

num_den_var = 5
den_mean = constant64([2.8,2.3,2.53,2.39,2.6])
den_std = constant64([0.2,0.17,0.1,0.14,0.1])

prior_mean = tf.concat([sfp_mean,den_mean],axis = 0)
prior_std = tf.concat([sfp_std,den_std],axis = 0)


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

def log_likelihood(self,mu):
    # forward calculating gravity
    Gm_ = self.gravity_function(mu,self.model,self.tz,self.fix_points,self.all_points_shape,transformer = self.transformer,densities = True )

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
stat_model.fix_points = fix_points
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
mu = constant64(([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
       0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
       0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
       0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
       0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 4.4408921e-16,
       0.0000000e+00, 0.0000000e+00, 0.0000000e+00]))
print(ilt.reverse_transform(mu))
stat_model.gravity_function(mu,stat_model.model,stat_model.tz,stat_model.fix_points,stat_model.all_points_shape,transformer = stat_model.transformer, densities = True)
# %%
##########MCMC###########

MCMCargs = dotdict({
    'num_results': 10,
    'number_burnin':0,
    'RMH_step_size': 0.03,
    'HMC_step_size': 0.005,
    'leapfrogs':4,
})

mu0_list = [mu]
samples_RMH_list = []
accept_RMH_list = []
# samples_HMC_list = []
# accept_HMC_list = []
for mu0 in mu0_list:
  samples_RMH,samples_HMC,accept_RMH,accept_HMC = mcmc(mu0,stat_model, RMH = True, HMC = True,MCMCargs = MCMCargs)
  samples_RMH_list.append(samples_RMH)
  accept_RMH_list.append(accept_RMH)
  # samples_HMC_list.append(samples_HMC)
  # accept_HMC_list.append(accept_HMC)


# %%

stat_model.set_result_path('/home/ib012512/Documents/Results/'+args.foldername+time.strftime("-%Y%m%d-%H%M%S"))
# %%
saving_dict = {'samples_RMH_list': samples_RMH_list,
              'accepted_rate_RMH':accept_RMH_list,
              # 'samples_HMC_list': samples_HMC_list,
              # 'accepted_rate_HMC':accept_HMC_list,
              'MAPs' : mu0_list,
              }
saving_dict.update(args)
saving_dict.update(MCMCargs)

json_dump = json.dumps(saving_dict, cls=NumpyEncoder)

with open(stat_model.path + '.json', 'w') as outfile:
    json.dump(json_dump, outfile)

print('Done')