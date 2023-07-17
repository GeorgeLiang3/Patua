import tensorflow as tf

def mcmc(mu,stat_model,RMH = True,HMC = True,MCMCargs = None):
  initial_chain_state = [mu]

  # RMH
  if RMH:
    states_RMH = stat_model.run_MCMC('RMH',
                                    num_results = MCMCargs.num_results,
                                    number_burnin = MCMCargs.number_burnin,
                                    step_size = MCMCargs.RMH_step_size,
                                    initial_chain_state =initial_chain_state
                                      )
    accept_RMH = (tf.reduce_mean(tf.cast(states_RMH.trace.is_accepted,stat_model.model.dtype)))*100
    print(f'acceptance rate Random-walk Matroplis {MCMCargs.num_results:d} {accept_RMH:.2f}%')
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
    print(f'acceptance rate Hamiltonian Monte Carlo {MCMCargs.num_results:d} {accept_HMC:.2f}%')


    # convert result to numpy
    samples_HMC = states_HMC.all_states[0].numpy()
    accept_HMC = accept_HMC.numpy()
  else:
    samples_HMC = None
    accept_HMC = None


  return  samples_RMH,samples_HMC,accept_RMH,accept_HMC