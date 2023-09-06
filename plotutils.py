import numpy as np
import matplotlib.pyplot as plt
import os

def get_last_patua_file(directory):
    # Get a list of all files that start with 'Patua'
    patua_files = [f for f in os.listdir(directory) if f.startswith('Patua')]

    # Sort the list in alphabetical order
    patua_files.sort()

    # Return the last file in the sorted list
    if patua_files:
        return patua_files[-1]
    else:
        return None

def plot_result(data):
  sampling_list = {}
  if data['samples_RMH_list'][0] is not None:
    samples_RMH = np.asarray(data['samples_RMH_list'])[0].T
    sampling_list['RMH'] = samples_RMH
  if data['samples_HMC_list'][0] is not None:
    samples_HMC = np.asarray(data['samples_HMC_list'])[0].T
    sampling_list['HMC'] = samples_HMC
  if data['samples_NUTS_list'][0] is not None:
    samples_NUTS = np.asarray(data['samples_NUTS_list'])[0].T
    sampling_list['NUTS'] = samples_NUTS

  def plot_traces(trace):
    num_para = trace.shape[0]
    fig = plt.figure(figsize=(20, 12))
    for i, one_trace in enumerate(trace):
          ax = plt.subplot(num_para//5+1, 5, i + 1)
          ax.plot(one_trace)

  for sampling_trace in sampling_list:

    plot_traces(sampling_list[sampling_trace])
    # plot_traces(samples_RMH)
    # plot_traces(samples_HMC)