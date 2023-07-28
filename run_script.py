
# %%
import sys
# local dir
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')
sys.path.append('/Volumes/GoogleDrive/My Drive/')
# cluster dir
sys.path.append('/home/ib012512/Documents/GemPhy/GP_old')
sys.path.append('/home/ib012512/Documents/')



# %%
from GemPhy.Stat.Bayes import Stat_model
from GemPhy.Geophysics.utils.util import constant64,dotdict,concat_xy_and_scale
from gempy.assets.geophysics import Receivers,GravityPreprocessing
from GemPhy.Geophysics.utils.ILT import *
from gempy.core.grid_modules.grid_types import CenteredRegGrid

# import gempy as gp
# from gempy.core.tensor.modeltf_var import ModelTF

from MCMC import mcmc
from UQ import UQ_Patua

from operator import add
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt


from Patua import PutuaModel
from LoadInputDataUtility import loadData


#########################
# %%
args = dotdict({
    "foldername": "Patua",
    'resolution':[16,16,12]
})

Bayesargs = dotdict({
    'prior_sfp_std': 50,
    'prior_den_std': 0.2,
    'likelihood_std': 10
    # 'likelihood_std':0.09, #the gravity data has an error range approximately between 0.5 mGal to 2.5 mGal. - Pollack, A, 2021
})

MCMCargs = dotdict({
    'num_results': 1000,
    'number_burnin':0,
    'RMH_step_size': 0.2,
    'HMC_step_size': 0.1,
    'leapfrogs':4,
})
# %%
# Load the data
P_model = PutuaModel()
init_model = P_model.init_model()
loadData(P_model.P, number_data = 20)
Data_obs = P_model.P['Grav']['Obs'] = P_model.P['Grav']['Obs'] - (min(P_model.P['Grav']['Obs']))
Data_measurement = tf.cast(Data_obs,init_model.dtype) 

# Define the receivers for gravity
model_extent = [None]*(6)
model_extent[::2] = P_model.P['xy_origin']
model_extent[1::2] = list( map(add, P_model.P['xy_origin'], P_model.P['xy_extent'] ) )

X_r = P_model.P['Grav']['xObs']
Y_r = P_model.P['Grav']['yObs']
Z_r = [model_extent[-1]]*P_model.P['Grav']['nObsPoints']

xyz = np.stack((X_r,Y_r,Z_r)).T
radius = [2000,2000,3000]

receivers = Receivers(radius,model_extent,xyz,kernel_resolution = args.resolution)

Reg_kernel = CenteredRegGrid(receivers.xy_ravel,radius=receivers.model_radius,resolution=receivers.kernel_resolution)

# %%
all_points = init_model.surface_points.df[['X','Y','Z']].to_numpy()

df = init_model.geo_data.surface_points.df
num_fault_points = len(df[df['surface'].str.startswith('fault')])
num_intrusion_points = len(df[df['surface'] == 'intrusion'])
num_GT_points = len(df[df['surface'] == 'GT'])

num_fix_points = num_fault_points + num_intrusion_points + num_GT_points # keep all the intrusion, faults and GT points fixed

fix_points = all_points[:num_fix_points] 
static_xy = all_points[:,0:2]
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


# %%

uq_P = UQ_Patua(gp_model    = init_model,
                Reg_kernel  = Reg_kernel,
                receivers   = receivers,
                transformer = ilt,
                num_para_total = num_para_total,
                fix_points = fix_points,
                static_xy = static_xy,
                Data_Obs = Data_measurement,
                args = args,
                Bayesargs = Bayesargs
                )
# %%
mu = ilt.transform(prior_mean)
# %%
uq_P.forward_function(mu)
# %%
uq_P.stat_model.log_likelihood(mu)

# %%
uq_P.set_initial_status([mu])
if __name__ is '__main__':
    uq_P.run_mcmc(MCMCargs,RMH = True, HMC = True)
# %%
