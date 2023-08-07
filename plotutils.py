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

  samples_RMH = np.asarray(data['samples_RMH_list'])[0].T
  samples_HMC = np.asarray(data['samples_HMC_list'])[0].T

  def plot_traces(trace):
    num_para = trace.shape[0]
    fig = plt.figure(figsize=(20, 12))
    for i, one_trace in enumerate(trace):
          ax = plt.subplot(num_para//5+1, 5, i + 1)
          ax.plot(one_trace)

  plot_traces(samples_RMH)
  plot_traces(samples_HMC)