
# %%
import sys
# local dir
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')
sys.path.append('/Volumes/GoogleDrive/My Drive/')
# cluster dir
sys.path.append('/home/ib012512/Documents/GemPhy/GP_old')
sys.path.append('/home/ib012512/Documents/')


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%

from GemPhy.Geophysics.utils.util import constant64,dotdict
from gempy.assets.geophysics import Receivers
from GemPhy.Geophysics.utils.ILT import *
from gempy.core.grid_modules.grid_types import CenteredRegGrid

# import gempy as gp
# from gempy.core.tensor.modeltf_var import ModelTF


from UQ_sodf import UQ_Patua
from UQ_sodf import Gravity_forward

from operator import add
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



from Patua import PutuaModel
from LoadInputDataUtility import loadData


#########################
# %%
args = dotdict({
    "foldername": "Patua",
    'resolution':[20,20,20],
    'num_data': 50,
})

Bayesargs = dotdict({
    'prior_sfp_std': 20,
    'prior_dip_std': 3,
    'prior_den_std': 0.2,
    'likelihood_std': 2.5,
    # 'likelihood_std':0.09, #the gravity data has an error range approximately between 0.5 mGal to 2.5 mGal. - Pollack, A, 2021
})

MCMCargs = dotdict({
    'RMH':False,
    'HMC':True,
    'NUTS':False,
    'num_results': 300,
    'number_burnin':0,
    'RMH_step_size': 0.2,
    'HMC_step_size': 0.1,
    'leapfrogs':4,
})
# %%
# Load the data
P_model = PutuaModel()
init_model = P_model.init_model()
# %%
init_model.compute_model()# TODO: Check if necessary. precompute the model to order the surfaces

ObsData = loadData(P_model.P, number_data = args.num_data)
Data_obs = P_model.P['Grav']['Obs'] - (P_model.P['Grav']['Obs'])[0]

Data_measurement = tf.cast(Data_obs,init_model.dtype) 

# Define the receivers for gravity
model_extent = [None]*(6)
model_extent[::2] = P_model.P['xy_origin']
model_extent[1::2] = list( map(add, P_model.P['xy_origin'], P_model.P['xy_extent'] ) )

X_r = P_model.P['Grav']['xObs']
Y_r = P_model.P['Grav']['yObs']
Z_r = [model_extent[-1]]*P_model.P['Grav']['nObsPoints']

xyz = np.stack((X_r,Y_r,Z_r)).T
radius = [1000,1000,2000]

receivers = Receivers(radius,model_extent,xyz,kernel_resolution = args.resolution)

Reg_kernel = CenteredRegGrid(receivers.xy_ravel,radius=receivers.model_radius,resolution=receivers.kernel_resolution)

# %%
all_points = init_model.surface_points.df[['X','Y','Z']].to_numpy()

df = init_model.geo_data.surface_points.df
num_fault_points = len(df[df['surface'].str.startswith('fault')])
num_intrusion_points = len(df[df['surface'] == 'intrusion'])
num_GT_points = len(df[df['surface'] == 'Volconic_felsic'])

num_fix_points = num_fault_points + num_intrusion_points + num_GT_points # keep all the intrusion, faults and GT points fixed
############################################
# Define the statistic problem
############################################

# Be very careful here
# Define the fixed surface points, which are all the intrusion points and Granite top points in this case
fix_points = tf.concat([all_points[:num_intrusion_points+num_fault_points],all_points[-num_GT_points:]],axis = 0)
# Only get the xy coordinates of the fixed points
static_xy = all_points[:,0:2]
# Get the shape of all the variable strata surface points
strata_points = all_points[num_intrusion_points+num_fault_points:-num_GT_points]
all_points_shape = all_points.shape

# Define the fixed dip angles 
fix_dip_angles = tf.concat([init_model.dip_angles[:4], init_model.dip_angles[16:]],axis = 0)

num_sf_var = strata_points.shape[0]
# Define the mean and std for surface points
sfp_mean = strata_points[:,2]
sfp_std = constant64([Bayesargs.prior_sfp_std]*num_sf_var)

#Define the mean and std for dip angles
dip_mean = init_model.dip_angles[4:16]
dip_std = constant64([Bayesargs.prior_dip_std]*dip_mean.numpy().shape[0])

num_den_var = 5
den_mean = constant64([2.9,2.1,2.2,2.3,2.8])
den_std = constant64([0.2,0.17,0.1,0.14,0.1])

# The input vector has shape [18 surface points, 5 density values,  12 dip angles]
prior_mean = tf.concat([sfp_mean,den_mean,dip_mean],axis = 0)
prior_std = tf.concat([sfp_std,den_std,dip_std],axis = 0)

num_para_total = prior_mean.shape[0]

### Define the bounds for parameters, bounds has to be normalized first 
lowerBound = prior_mean - 3*prior_std
upperBound = prior_mean + 3*prior_std

# invertible logarithmic transform
ilt = ILT(lowerBound,upperBound)


# %%
gf = Gravity_forward(gp_model    = init_model,
                Reg_kernel  = Reg_kernel,
                receivers   = receivers)

uq_P = UQ_Patua(gravity_forward_model    = gf,
                transformer = ilt,
                num_para_total = num_para_total,
                delta = 2.,
                fix_points = fix_points,
                static_xy = static_xy,
                fix_dips = fix_dip_angles,
                Data_Obs = Data_measurement,
                args = args,
                Bayesargs = Bayesargs,
                num_fault_points = num_fault_points,
                num_intrusion_points = num_intrusion_points, 
                num_GT_points = num_GT_points,
                )
# %%
mu = ilt.transform(prior_mean)
# %%
# uq_P.forward_function(mu)
# %%
# uq_P.stat_model.log_likelihood(mu)

# %%
mu_list = uq_P.stat_model.mvn_prior.sample(5)
uq_P.set_initial_status(mu_list)
if __name__ == '__main__':
    # uq_P.forward_function(mu)
    uq_P.run_mcmc(MCMCargs)

# %%
