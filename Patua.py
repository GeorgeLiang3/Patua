
# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/GemPhy/GP_old/')

import gempy as gp
from gempy.core.tensor.modeltf_var import ModelTF

import numpy as np
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

# # for experiment
# del_surfaces = ['fault2','fault3', 'fault4', 'fault5', 'fault6', 'fault7', 'fault8', 'fault9', 'fault10', 'fault11', 'fault12','GT' ,'Volcanic_mafic','Volconic_felsic']
# del_surfaces = ['intrusion']
# geo_data.delete_surfaces(del_surfaces, remove_data=True)
# "Intrusion": 'intrusion',
gp.map_series_to_surfaces(geo_data, {"Intrusion": 'intrusion',
                                     "Fault_Series1": 'fault1',
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
                                    "Sedimentary_Series": ('Volcanic_mafic','Volconic_felsic',
                                    'GT'),
                                    "Basement":'basement'
                                    }
                                    )

order_series = ['Fault_Series1',
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
                'Sedimentary_Series',
                'Basement']

# geo_data.reorder_series(order_series)

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

mapping_object = {"Fault_Series1":np.array([1,1,1]),
                  "Fault_Series2":np.array([1,1,1]),
                  "Fault_Series3":np.array([1,1,1]),
                  "Fault_Series4":np.array([1,1,1]),
                  "Fault_Series5": np.array([1,1,1]),
                  "Fault_Series6":np.array([1,1,1]),
                  "Fault_Series7":np.array([1,1,1]),
                  "Fault_Series8": np.array([1,1,1]),
                  "Fault_Series9": np.array([1,1,1]),
                  "Fault_Series10":np.array([1,1,1]),
                  "Fault_Series11":np.array([1,1,1]),
                  "Fault_Series12":np.array([1,1,1]),
                  "Intrusion":np.array([1,1,0.1]),
                  "Sedimentary_Series": np.array([1,1,1]),
                  }
gp.assign_global_anisotropy(geo_data,mapping_object)

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
gp.plot.plot_section(model, cell_number=18,
                         direction='y', show_data=True)
# %%
