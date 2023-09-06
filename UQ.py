import sys
# local dir
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')
sys.path.append('/Volumes/GoogleDrive/My Drive/')
# cluster dir
sys.path.append('/home/ib012512/Documents/GemPhy/GP_old')
sys.path.append('/home/ib012512/Documents/')

import json
import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from MCMC import mcmc
from GemPhy.Geophysics.utils.util import constant64,concat_xy_and_scale,calculate_slope_scale,NumpyEncoder
from gempy.assets.geophysics import GravityPreprocessing
from GemPhy.Stat.Bayes import Stat_model

class UQ_Patua():
  def __init__(self,gp_model,Reg_kernel,receivers,transformer,num_para_total,delta = 2.,fix_points = None, static_xy = None, args = None,Bayesargs = None,Data_Obs = None,
               num_fault_points = None,
               num_intrusion_points = None, 
               num_GT_points = None,) -> None:
    self.gp_model = gp_model
    self.kernel = Reg_kernel
    self.receivers = receivers
    self.transformer = transformer
    self.num_para_total = num_para_total
    self.fix_points = fix_points
    self.static_xy = static_xy
    self.args = args
    self.Bayesargs = Bayesargs
    self.num_fault_points = num_fault_points
    self.num_intrusion_points = num_intrusion_points
    self.num_GT_points = num_GT_points

    self.all_points = self.gp_model.surface_points.df[['X','Y','Z']].to_numpy()
    self.all_points_shape = self.all_points.shape

    self.gp_model.activate_customized_grid(self.kernel)
    self.calculate_slope()
    self.create_graph(delta = delta)
    self.init_stat_model(Data_Obs)


  def set_initial_status(self, mu):
    self.init_stat = mu
  def calculate_slope(self):
    self.max_slope = calculate_slope_scale(self.kernel,rf = self.gp_model.rf)

  def create_graph(self,delta = 2.):
    self.gp_model.create_tensorflow_graph(delta = delta,gradient=True,compute_gravity=True,max_slope = self.max_slope)
    

    g_center_regulargrid = GravityPreprocessing(self.kernel)
    tz_center_regulargrid = tf.constant(g_center_regulargrid.set_tz_kernel(),self.gp_model.tfdtype)
    self.tz = tf.constant(tz_center_regulargrid,self.gp_model.tfdtype)

  def init_stat_model(self,Data_measurement):
    
    self.stat_model = Stat_model(self.gp_model,self.forward_function,self.num_para_total, self.tz, transformer = self.transformer )
    Stat_model.log_likelihood = log_likelihood
    self.stat_model.fix_points = self.fix_points
    self.stat_model.all_points_shape = self.all_points_shape
    # Set Prior
    self.stat_model.set_prior(Number_para = self.num_para_total)
    # Manually define the total shape of surface points (including intrusion and faults here)
    self.stat_model.sfp_shape = self.all_points_shape

    # Set likelihood

    Data_std = self.Bayesargs.likelihood_std
    self.stat_model.set_likelihood(Data_measurement,Data_std)
    self.stat_model.monitor=False


  def parameter2input(self,mu,transformer = None,DENSITIES_FLAG = True):
    '''
      This function convert the normalized flattened parameters to gempy forward input function
    '''
    if transformer is None:
      mu_norm = mu
    else:
      mu_norm = transformer.reverse_transform(mu)

    if not DENSITIES_FLAG: # use default densities defined in the model
      properties = constant64(self.gp_model.geo_data.surfaces.df['densities'].to_numpy())
      # concatenate the surface points with order (intrusion, faults, strata, GT)
      # TODO: this is a bit hacky, need to be improved. 
      sfp_z = tf.concat([self.fix_points[:self.num_intrusion_points+self.num_fault_points,2],mu_norm,self.fix_points[-self.num_GT_points:,2]],axis = -1)

    else:
      properties = mu_norm[-5:]
      # concatenate the surface points with order (intrusion, faults, strata, GT)
      sfp_z = tf.concat([self.fix_points[:self.num_intrusion_points+self.num_fault_points,2],mu_norm[:-5],self.fix_points[-self.num_GT_points:,2]],axis = -1)
      # concatenate the auxiliary densities
      auxiliary_densities = constant64([-1]*12)
      properties = tf.concat([properties[:1],auxiliary_densities,properties[1:]],axis = -1)

    sfp_xyz = concat_xy_and_scale(sfp_z,self.gp_model,self.static_xy,self.all_points_shape)
    properties = tf.stack([self.gp_model.TFG.lith_label,properties],axis = 0)

    return sfp_xyz,properties

  def forward(self,sf,all_properties,sigmoid = True):
    final_block,final_property,block_matrix,block_mask,size,scalar_field,sfai,grav = self.gp_model.compute_gravity(self.tz,surface_points = sf,kernel = self.kernel,receivers = self.receivers,method = 'kernel_reg',LOOP_FLAG = False,all_properties =all_properties, DEBUG_FLAG = True)
    return grav
  
  def forward_function(self,mu,sigmoid = True, DENSITIES_FLAG = True):

    sfp_xyz,all_properties = self.parameter2input(mu,transformer= self.transformer, DENSITIES_FLAG = DENSITIES_FLAG)

    gravity = self.forward(sfp_xyz,all_properties,sigmoid)
    # reverse the axis and deduce the min
    gravity = -gravity
    gravity = gravity - tf.math.reduce_mean(gravity)
    return gravity
  
  def run_mcmc(self,MCMCargs, save = True):
    self.MCMCargs = MCMCargs
    mu0_list = self.init_stat
    samples_RMH_list = []
    accept_RMH_list = []
    samples_HMC_list = []
    accept_HMC_list = []
    samples_NUTS_list = []
    accept_NUTS_list = []

    for mu0 in mu0_list:
        samples_RMH,samples_HMC,samples_NUTS,accept_RMH,accept_HMC,accept_NUTS = mcmc(mu0,self.stat_model,MCMCargs = MCMCargs)
        samples_RMH_list.append(samples_RMH)
        accept_RMH_list.append(accept_RMH)
        samples_HMC_list.append(samples_HMC)
        accept_HMC_list.append(accept_HMC)
        samples_NUTS_list.append(samples_NUTS)
        accept_NUTS_list.append(accept_NUTS)
    if save:
      self.save_results(samples_RMH_list = samples_RMH_list,
                        samples_HMC_list = samples_HMC_list,
                        accept_RMH_list = accept_RMH_list,
                        accept_HMC_list = accept_HMC_list,
                        samples_NUTS_list = samples_NUTS_list,
                        accept_NUTS_list = accept_NUTS_list)
    return samples_RMH_list,samples_HMC_list,samples_NUTS_list,accept_RMH_list,accept_HMC_list,accept_NUTS_list

  def save_results(self,**kwargs):

    self.stat_model.set_result_path('/home/ib012512/Documents/Results/'+self.args.foldername+time.strftime("-%Y%m%d-%H%M%S"))

    # %%
    # saving_dict = {}
    # for key in kwargs:
    #   print('key:',key)
    #   if kwargs[key] is not None:
    #     saving_dict.update(zip(key,arg))
    saving_dict = kwargs
    print('saving_dict:',saving_dict)
    # saving_dict = {'samples_RMH_list': samples_RMH_list,
    #               'accepted_rate_RMH':accept_RMH_list,
    #               'samples_HMC_list': samples_HMC_list,
    #               'accepted_rate_HMC':accept_HMC_list,
    #               # 'MAPs' : mu0_list,
    #               }
    saving_dict.update(self.args)
    saving_dict.update(self.Bayesargs)
    saving_dict.update(self.MCMCargs)

    json_dump = json.dumps(saving_dict, cls=NumpyEncoder)

    with open(self.stat_model.path + '.json', 'w') as outfile:
        json.dump(json_dump, outfile)

    print('Saved')
    
def log_likelihood(self,mu):
    # forward calculating gravity
    Gm_ = self.gravity_function(mu,sigmoid = True, DENSITIES_FLAG = True )

    mvn_likelihood = tfd.MultivariateNormalTriL(
        loc=Gm_,
        scale_tril=tf.cast(tf.linalg.cholesky(self.data_cov_matrix),self.tfdtype))

    likelihood_log_prob = tf.reduce_sum(mvn_likelihood.log_prob(self.Obs))
    return likelihood_log_prob