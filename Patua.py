
# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')

import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF


# %%

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

# %%
data_path = './Data/'
geo_data = gp.create_data( extent=[P['xmin'], P['xmax'], P['ymin'], P['ymax'], P['zmin'], P['zmax']], resolution=[50, 50, 50],
                          path_o=data_path + "Patua_orientations.csv",
                          path_i=data_path + "Patua_surface_points.csv")


gp.map_series_to_surfaces(geo_data, {"Fault_Series1": 'fault1',
                                     "Fault_Series2": 'fault2',
                                     "Fault_Series3": 'fault3',
                                     "Fault_Series4": 'fault4',
                                     "Fault_Series5": 'fault5',
                                     "Fault_Series6": 'fault6',
                                     "Fault_Series7": 'fault7',
                                     "Fault_Series8": 'fault8',
                                     "Fault_Series9": 'fault9',
                                     "Fault_Series10": 'fault10',
                                     "Fault_Series11": 'fault11',
                                     "Fault_Series12": 'fault12',
                                    "Strat_Series": ('Volcanic_mafic','Volconic_felsic',
                                    'GT')})
geo_data.set_is_fault(['Fault_Series1',
                       'Fault_Series2',
                       'Fault_Series3',
                       'Fault_Series4',
                       'Fault_Series5',
                       'Fault_Series6',
                       'Fault_Series7',
                       'Fault_Series8',
                       'Fault_Series9',
                       'Fault_Series10',
                       'Fault_Series11',
                       'Fault_Series12',
                       ])


# %%
## Initialize the model
model = ModelTF(geo_data)
model.activate_regular_grid()
model.create_tensorflow_graph(gradient = False)

# %%
model.compute_model()
# %%
gp._plot.plot_3d(model)
# %%
gp.plot.plot_section(model, cell_number=15,
                         direction='y', show_data=True)