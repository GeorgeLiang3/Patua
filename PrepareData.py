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

P['xy_origin']=[318000,4379246, 1200-4000]
P['xy_extent'] = [7000,9000,4000]

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
# In GemPy the surface is defined as the bottom of the stratigraphic unit
surfacepoints_df['formation'] = 'Volconic_felsic'
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
Sedimentary_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
Volcanic_mafic_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
# %%


y_coord_sed = Sedimentary_df['Y'] - P['xy_origin'][1] - P['xy_extent'][1]/2
z_sed = -300*np.exp(-((y_coord_sed)**2/ 1000**2))

y_coord_vm = Sedimentary_df['Y'] - P['xy_origin'][1] - P['xy_extent'][1]/2
z_vm = -300*np.exp(-((y_coord_vm)**2/ 1000**2))

Sedimentary_df['Z'] = z_sed+1000
Sedimentary_df['formation'] = 'Sedimentary'
Volcanic_mafic_df['Z'] = z_vm+600
Volcanic_mafic_df['formation'] = 'Volcanic_mafic'

# Sedimentary_df[['Z','formation']] = [1000,'Sedimentary']
# Volcanic_mafic_df[['Z','formation']] = [600,'Volcanic_mafic']

surfacepoints_df = pd.concat([surfacepoints_df,
                              Sedimentary_df,
                              Volcanic_mafic_df])

## Orientations
## Because I don't have orientation data, I will make pseudo orientation data at the corner of the model. All pointing upwards.

orientation_df = pd.DataFrame(columns=['X','Y','Z','azimuth','dip','polarity','formation'])
nx, ny = (2, 2)
x = np.linspace(P['xy_origin'][0], P['xy_origin'][0]+P['xy_extent'][0], nx)
y = np.linspace(P['xy_origin'][1], P['xy_origin'][1]+P['xy_extent'][1], ny)
xx, yy = np.meshgrid(x,y)
sedimentary_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
Sedimentary_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
Volcanic_mafic_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})
GT_df = pd.DataFrame({'X': xx.flatten(), 'Y': yy.flatten()})

Sedimentary_df[['Z','azimuth','dip','polarity','formation']] = [1000,90,0,1,'Sedimentary']
Volcanic_mafic_df[['Z','azimuth','dip','polarity','formation']] = [600,90,0,1,'Volcanic_mafic']
GT_df[['Z','azimuth','dip','polarity','formation']] = [-200,90,0,1,'Volconic_felsic']

# Manually add an orientation point to constraint the structure
GT_df = GT_df.append({'X': 320000, 'Y': 4382000, 'Z': -300, 'azimuth': 90, 'dip': 60, 'polarity': 1, 'formation': 'Volconic_felsic'},ignore_index=True)


orientation_df = pd.concat([orientation_df,
                            Sedimentary_df,
                            Volcanic_mafic_df,
                            GT_df])

# %%
## Faults
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

def angle2vector(dip,azimuth):
    x = np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(azimuth))
    y = np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(azimuth))
    z = np.cos(np.deg2rad(dip)) 
    return x,y,z


def find_points_on_perpendicular_plane(origin,pole_vector, d):
    # x = origin[0] - (y - origin[1])*pole_vector[1]/pole_vector[0]
    x = np.sqrt(d/(1+pole_vector[0]**2/pole_vector[1]**2))+origin[0]
    y = np.sqrt(d-(x-origin[0])**2) + origin[1]
    # y = np.array(y)
    z = np.array(origin[2])
    return np.array([x,y,z])


def extrapolate_surfacepoints_along_plane(origin, pole_vector, number_points_2_create = 2, distance = 400):
    ds = np.linspace(origin[1]-distance,origin[1]+distance,number_points_2_create)
    points =np.array([find_points_on_perpendicular_plane(origin,pole_vector,d) for d in ds])
    return points




# %%
# Process the fault data

fault_df['azimuth'] = fault_df['strike'].apply(convert_strike_to_azimuth)

new_fault_data_df = pd.DataFrame(columns=['X','Y','Z','azimuth','dip','polarity','formation'])
for index, row in fault_df.iterrows():
    V =np.array(angle2vector(row['dip'],row['azimuth']))
    origin = row[['X','Y','Z']].to_numpy()
    points = extrapolate_surfacepoints_along_plane(origin, pole_vector=V)

    for point in points:
        new_row = {'X':point[0],'Y':point[1],'Z':point[2],'azimuth':row['azimuth'],'dip':row['dip'],'polarity':row['polarity'],'formation':row['formation']}
        new_fault_data_df = new_fault_data_df.append(new_row, ignore_index=True)
   
# %%
# Intrusion
intrusion_data_df = pd.DataFrame(columns=['X','Y','Z','azimuth','dip','polarity','formation'])
intrusion_center = [P['xmin']+6000,P['ymin']+3480,P['zmax'] - 2000]
intrusion_radius = 400

azimuth_dict = {'south': 180,
                'west': 270,
                'north':0,
                'east': 90}

point_south = {'X':intrusion_center[0],'Y':intrusion_center[1]-intrusion_radius,'Z':intrusion_center[2],'azimuth':azimuth_dict['north'],'dip':90,'polarity':1,'formation':'intrusion'}
point_west = {'X':intrusion_center[0]-intrusion_radius,'Y':intrusion_center[1],'Z':intrusion_center[2],'azimuth':azimuth_dict['east'],'dip':90,'polarity':1,'formation':'intrusion'}
point_north = {'X':intrusion_center[0],'Y':intrusion_center[1]+intrusion_radius,'Z':intrusion_center[2],'azimuth':azimuth_dict['south'],'dip':90,'polarity':1,'formation':'intrusion'}
point_east = {'X':intrusion_center[0]+intrusion_radius,'Y':intrusion_center[1],'Z':intrusion_center[2],'azimuth':azimuth_dict['west'],'dip':90,'polarity':1,'formation':'intrusion'}
intrusion_data_df = intrusion_data_df.append([point_south,point_west,point_north,point_east], ignore_index=True)
intrusion_surface_df = intrusion_data_df[['X','Y','Z','formation']]
intrusion_orientation_df = intrusion_data_df[['X','Y','Z','azimuth','dip','polarity','formation']]

# %%
#split the surface points and orientation points. Here use the same coordinates for surface and orientation point
fault_surface_df = new_fault_data_df[['X','Y','Z','formation']]
# Put faults and surface data together
fault_orientation_df = fault_df[['X','Y','Z','azimuth','dip','polarity','formation']]

orientation_df = pd.concat([orientation_df,fault_orientation_df,intrusion_orientation_df])
surfacepoints_df = pd.concat([surfacepoints_df,fault_surface_df,intrusion_surface_df])




# save
surfacepoints_df.to_csv('./Data/Patua_surface_points.csv',index=False)

orientation_df.to_csv('./Data/Patua_orientations.csv',index=False)



# %%
