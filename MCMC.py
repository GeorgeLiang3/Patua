import tensorflow as tf

def mcmc(mu,stat_model,MCMCargs = None):
  if 'RMH' in MCMCargs:
    RMH = MCMCargs.RMH
  else:
    RMH = False

  if 'HMC' in MCMCargs:
    HMC = MCMCargs.HMC
  else:
    HMC = False
  
  if 'NUTS' in MCMCargs:
    NUTS = MCMCargs.NUTS
  else:
    NUTS = False
  
  initial_chain_state = [mu]
  print('initial_chain_state:',initial_chain_state)
  # RMH
  if RMH:
    states_RMH = stat_model.run_MCMC('RMH',
                                    num_results = MCMCargs.num_results,
                                    number_burnin = MCMCargs.number_burnin,
                                    step_size = MCMCargs.RMH_step_size,
                                    initial_chain_state =initial_chain_state
                                      )
    accept_RMH = (tf.reduce_mean(tf.cast(states_RMH.trace.is_accepted,stat_model.model.dtype)))*100
    print(f'acceptance rate Random-walk Matroplis {MCMCargs.num_results:d} samples: {accept_RMH:.2f}%')
    samples_RMH = states_RMH.all_states[0].numpy()
    accept_RMH = accept_RMH.numpy()
  else:
    samples_RMH = None
    accept_RMH = None

  # HMC
  if HMC:
    states_HMC = stat_model.run_MCMC(method='HMC',
                                    num_results = MCMCargs.num_results,
                                    number_burnin=MCMCargs.number_burnin,
                                    step_size=MCMCargs.HMC_step_size,
                                    num_leapfrog_steps=MCMCargs.leapfrogs,
                                    initial_chain_state =initial_chain_state)
    accept_HMC = (tf.reduce_mean(tf.cast(states_HMC.trace.is_accepted,stat_model.model.dtype)))*100
    print(f'acceptance rate Hamiltonian Monte Carlo {MCMCargs.num_results:d} samples: {accept_HMC:.2f}%')


    # convert result to numpy
    samples_HMC = states_HMC.all_states[0].numpy()
    accept_HMC = accept_HMC.numpy()
  else:
    samples_HMC = None
    accept_HMC = None

  if NUTS:
    states_NUTS = stat_model.run_MCMC(method='NUTS',
                                    num_results = MCMCargs.num_results,
                                    number_burnin=MCMCargs.number_burnin,
                                    step_size=MCMCargs.HMC_step_size,
                                    initial_chain_state =initial_chain_state)
    accept_NUTS = (tf.reduce_mean(tf.cast(states_NUTS.trace.is_accepted,stat_model.model.dtype)))*100
    print(f'acceptance rate NUTS {MCMCargs.num_results:d} samples: {accept_NUTS:.2f}%')


    # convert result to numpy
    samples_NUTS = states_NUTS.all_states[0].numpy()
    accept_NUTS = accept_NUTS.numpy()
  else:
    samples_NUTS = None
    accept_NUTS = None


  return  samples_RMH,samples_HMC,samples_NUTS,accept_RMH,accept_HMC,accept_NUTS