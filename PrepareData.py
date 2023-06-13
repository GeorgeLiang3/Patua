# %%
import LoadInputDataUtility as DI
import VisualizationUtilities as Viz
import pandas as pd
import numpy as np
# import sys
# sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')

# import gempy as gp
# from gempy.core.tensor.modeltf_var import ModelTF

# %matplotlib inline  

P = {}
P['HypP'] = {}
P['HypP']['jupyter'] = False
P['HypP']['ErrorType'] = 'Global'
P['DataTypes'] = ['Grav']

P['xy_origin']=[317883,4379246, 1200-4000]
P['xy_extent'] = [9000,9400,4000]

# Define the model limits
P['xmin'] = P['xy_origin'][0]
P['xmax'] = P['xy_origin'][0]+P['xy_extent'][0]

P['ymin'] = P['xy_origin'][1]
P['ymax'] = P['xy_origin'][1]+P['xy_extent'][1]

P['zmin'] = P['xy_origin'][2]
P['zmax'] = P['xy_origin'][2]+P['xy_extent'][2]

DI.loadData(P)
figfilename = 'test.png'
Viz.visualize_opt_step(figfilename, P)

# %%
surfacepoints_df = pd.DataFrame(P['GT'])
surfacepoints_df = surfacepoints_df.drop('nObsPoints',axis=1)
surfacepoints_df['formation'] = 'GT'
surfacepoints_df.columns = ['X','Y','Z','formation']

#Use a uniform mesh as the input and use a uniform thickness to define theZ
# %%
layers_surfacepoints_df = pd.DataFrame(columns=['X','Y','Z','formation'])

# create a 3x3 mesh as the coordinates for intermediate layers surface points
nx, ny = (3, 3)
x = np.linspace(P['xy_origin'][0]+1500, P['xy_origin'][0]+P['xy_extent'][0]-1500, nx)
y = np.linspace(P['xy_origin'][1]+1500, P['xy_origin'][1]+P['xy_extent'][1]-1500, ny)
xx, yy = np.meshgrid(x,y)
# sedimentary_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
Volcanic_mafic_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
Volconic_felsic_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})



# sedimentary_df[['Z','formation']] = [200,'sedimentary']
Volcanic_mafic_df[['Z','formation']] = [1000,'Volcanic_mafic']
Volconic_felsic_df[['Z','formation']] = [600,'Volconic_felsic']

surfacepoints_df = pd.concat([surfacepoints_df,
                              Volcanic_mafic_df,
                              Volconic_felsic_df])

## Orientations
orientation_df = pd.DataFrame(columns=['X','Y','Z','azimuth','dip','polarity','formation'])
nx, ny = (2, 1)
x = np.linspace(P['xy_origin'][0], P['xy_origin'][0]+P['xy_extent'][0], nx)
y = np.linspace(P['xy_origin'][1], P['xy_origin'][1]+P['xy_extent'][1], ny)
xx, yy = np.meshgrid(x,y)
sedimentary_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
Volcanic_mafic_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
Volconic_felsic_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
# sedimentary_df[['Z','azimuth','dip','polarity','formation']] = [200,90,0,1,'sedimentary']
Volcanic_mafic_df[['Z','azimuth','dip','polarity','formation']] = [1000,90,0,1,'Volcanic_mafic']
Volconic_felsic_df[['Z','azimuth','dip','polarity','formation']] = [600,90,0,1,'Volconic_felsic']

orientation_df = pd.concat([orientation_df,
                            Volcanic_mafic_df,
                            Volconic_felsic_df])

# %%
#get the fault data from Smith et. al. (2023)
fault_df = pd.read_csv('Data/Patua_fault_data.csv', header=0)
# convert the coordinates
fault_df['X'] = fault_df['X'] + 320000
fault_df['Y'] = fault_df['Y'] + 4380000

fault_df['polarity'] = 1
# %%
def convert_strike_to_azimuth(strike):
    azimuth = strike 
    if azimuth < 0:
        azimuth += 360
    return azimuth

#split the surface points and orientation points. Here use the same coordinates for surface and orientation point
fault_surface_df = fault_df[['X','Y','Z','formation']]


fault_df['azimuth'] = fault_df['strike'].apply(convert_strike_to_azimuth)
fault_orientation_df = fault_df[['azimuth','dip','polarity','X','Y','Z','formation']]

orientation_df = pd.concat([orientation_df,fault_orientation_df])
surfacepoints_df = pd.concat([surfacepoints_df,fault_surface_df])
surfacepoints_df.to_csv('./Data/Patua_surface_points.csv',index=False)

orientation_df.to_csv('./Data/Patua_orientations.csv',index=False)


