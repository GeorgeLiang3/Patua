# %%
import json
import numpy as np
import matplotlib.pyplot as plt

with open('/Users/zhouji/Documents/Cluster/Results/Patua-20230724-115131.json') as f:
  data = json.load(f)
data = json.loads(data)


samples_RMH = np.asarray(data['samples_RMH_list'])[0]
samples_HMC = np.asarray(data['samples_HMC_list'])[0]

fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(15, 8))
print(samples_RMH.shape)
for i in range(5):
  for j in range(10):
    axes[i][j].plot(samples_RMH[:,i*10+j])
plt.show()
# %%
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(15, 8))
print(samples_HMC.shape)
for i in range(5):
  for j in range(10):
    axes[i][j].plot(samples_HMC[:,i*10+j])
plt.show()
# %%
